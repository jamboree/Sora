//===--- Driver.cpp ---------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Driver/Driver.hpp"
#include "Sora/AST/ASTScope.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "Sora/Diagnostics/DiagnosticsDriver.hpp"
#include "Sora/Driver/DiagnosticVerifier.hpp"
#include "Sora/Driver/Options.hpp"
#include "Sora/Parser/Parser.hpp"
#include "Sora/Sema/Sema.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"

using namespace sora;
using namespace llvm::opt;

Driver::Driver(raw_ostream &out)
    : driverDiags(driverDiagsSrcMgr), optTable(createSoraOptTable()) {
  driverDiags.createConsumer<PrintingDiagnosticConsumer>(llvm::outs());
}

InputArgList Driver::parseArgs(ArrayRef<const char *> args, bool &hadError) {
  hadError = false;
  unsigned missingArgIndex = 0, missingArgCount = 0;
  // Parse the arguments, set the argList.
  llvm::opt::InputArgList argList =
      optTable->ParseArgs(args, missingArgIndex, missingArgCount);
  // Check for unknown arguments & diagnose them
  for (const Arg *arg : argList.filtered(opt::OPT_UNKNOWN)) {
    hadError = true;
    diagnose(diag::unknown_arg, arg->getAsString(argList));
  }
  // Diagnose missing arguments values
  if (missingArgCount) {
    diagnose(diag::missing_argv, argList.getArgString(missingArgIndex),
             missingArgCount);
    hadError = true;
  }
  return argList;
}

bool Driver::handleImmediateArgs(InputArgList &argList) {
  // handle -h/-help
  if (argList.hasArg(opt::OPT_help)) {
    optTable->PrintHelp(llvm::outs(), "sorac (options | inputs)",
                        "Sora Compiler");
    return false;
  }
  return true;
}

std::unique_ptr<CompilerInstance>
Driver::tryCreateCompilerInstance(llvm::opt::InputArgList &argList) {
  // To make make_unique work with the private constructor, we must
  // use a small trick.
  struct CompilerInstanceCreator : public CompilerInstance {
    using CompilerInstance::CompilerInstance;
  };
  auto CI = std::make_unique<CompilerInstanceCreator>();
  if (!CI->handleOptions(argList))
    return nullptr;
  if (!CI->loadInputs(argList))
    return nullptr;
  return std::move(CI);
}

bool CompilerInstance::handleOptions(InputArgList &argList) {
  bool success = true;
  options.dumpParse = argList.hasArg(opt::OPT_dump_parse);
  options.dumpAST = argList.hasArg(opt::OPT_dump_ast);
  options.parseOnly = argList.hasArg(opt::OPT_parse_only);
  options.verifyModeEnabled = argList.hasArg(opt::OPT_verify);
  options.printMemUsage = argList.hasArg(opt::OPT_print_mem_usage);
  if (Arg *arg = argList.getLastArg(opt::OPT_dump_scope_maps)) {
    StringRef value = arg->getValue();
    if (value == "lazy")
      options.scopeMapPrintingMode = ScopeMapPrintingMode::Lazy;
    else if (value == "expanded")
      options.scopeMapPrintingMode = ScopeMapPrintingMode::Expanded;
    else {
      success = false;
      diagnose(diag::unknown_argv_for, value, arg->getSpelling());
    }
  }
  return success;
}

bool CompilerInstance::loadInputs(llvm::opt::InputArgList &argList) {
  bool success = true;
  for (const Arg *arg : argList.filtered(opt::OPT_INPUT)) {
    success &= !loadFile(arg->getValue()).isNull();
    arg->claim();
  }
  return success;
}

void CompilerInstance::dump(raw_ostream &out) const {
#define DUMP_BOOL(EXPR)                                                        \
  out << #EXPR << ": " << ((EXPR) ? "true" : "false") << '\n';
  /// Dump state
  DUMP_BOOL(ran);
  out << "inputBuffers: " << inputBuffers.size() << '\n';
  for (auto buffer : inputBuffers) {
    out << "  Buffer #" << buffer.getRawValue() << " (";
    if (buffer) {
      StringRef name = srcMgr.getBufferIdentifier(buffer);
      if (name.empty())
        out << "<unnamed>";
      else
        out << name;
    }
    else
      out << "invalid";
    out << ")\n";
  }
  /// Dump options
  DUMP_BOOL(options.dumpParse);
  DUMP_BOOL(options.dumpAST);
  DUMP_BOOL(options.parseOnly);
  DUMP_BOOL(options.verifyModeEnabled);
  DUMP_BOOL(options.printMemUsage);
  out << "options.scopeMapPrintingMode: ";
  switch (options.scopeMapPrintingMode) {
  case ScopeMapPrintingMode::None:
    out << "None";
    break;
  case ScopeMapPrintingMode::Lazy:
    out << "Lazy";
    break;
  case ScopeMapPrintingMode::Expanded:
    out << "Expanded";
    break;
  }
  out << "\n";
#undef DUMP_BOOL
}

BufferID CompilerInstance::loadFile(StringRef filepath) {
  if (auto result = llvm::MemoryBuffer::getFile(filepath)) {
    auto buffer = srcMgr.giveBuffer(std::move(*result));
    assert(buffer && "invalid buffer id returned by giveBuffer");
    inputBuffers.push_back(buffer);
    return buffer;
  }
  diagnose(diag::couldnt_load_input_file, filepath);
  hadFileLoadError = true;
  return BufferID();
}

bool CompilerInstance::isLoaded(StringRef filePath, bool isAbsolute) {
  SmallVector<char, 16> path(filePath.begin(), filePath.end());
  if (!isAbsolute) {
    auto errc = llvm::sys::fs::make_absolute(path);
    // TODO: handle the error_code properly
    assert(!errc && "make_absolute failed");
  }
  // TODO
  return false;
}

bool CompilerInstance::run() {
  assert(!ran && "already ran this CompilerInstance!");
  Step stopAfter = options.parseOnly ? Step::Parsing : Step::Last;
  ran = true;
  // Check if we have input files
  if (inputBuffers.empty()) {
    if (!hadFileLoadError)
      diagnose(diag::no_input_files);
    return false;
  }
  // Currently, we only accept a single input file.
  if (inputBuffers.size() != 1) {
    diagnose(diag::only_one_file_accepted);
    return false;
  }

  // In verify mode, install the Diagnostic Verifier
  DiagnosticVerifier *verifier = installDiagnosticVerifierIfNeeded();
  if (verifier) {
    bool verifParsingOk = true;
    // Parse the input files if verify mode has been enabled
    for (auto inputBuffer : inputBuffers)
      verifParsingOk &= verifier->parseFile(inputBuffer);
    // If an error occured during DV parsing, stop now.
    if (!verifParsingOk)
      return false;
  }

  bool success = true;

  // Helper function, returns true if we can continue, false otherwise.
  auto canContinue = [&](Step currentStep) {
    return success && stopAfter != currentStep;
  };

  // Helper function to finish processing. Returns true on success, false on
  // failure.
  auto finish = [&]() {
    // When the verifier is active, its output will be our return value.
    return verifier ? verifier->finish() : success;
  };

  // Create the ASTContext
  createASTContext();
  assert(astContext && "no ASTContext?");

  // Create the source file
  SourceFile &sf = createSourceFile(inputBuffers[0]);

  // Perform Parsing
  success = doParsing(sf);
  if (!canContinue(Step::Parsing))
    return finish();
  // Perform Semantic Analysis
  success = doSema(sf);
  if (!canContinue(Step::Sema))
    return finish();
  // TODO: Other steps
  return finish();
}

ArrayRef<BufferID> CompilerInstance::getInputBuffers() const {
  return inputBuffers;
}

DiagnosticVerifier *CompilerInstance::installDiagnosticVerifierIfNeeded() {
  if (!options.verifyModeEnabled)
    return nullptr;
  auto verifier = std::make_unique<DiagnosticVerifier>(llvm::outs(), srcMgr);
  verifier->setConsumer(diagEng.takeConsumer());
  auto ptr = verifier.get();
  diagEng.setConsumer(std::move(verifier));
  return ptr;
}

SourceFile &CompilerInstance::createSourceFile(BufferID buffer) {
  assert(astContext && "No ASTContext?");
  return *SourceFile::create(*astContext, buffer, nullptr);
}

void CompilerInstance::dumpScopeMaps(raw_ostream &out, SourceFile &file) {
  auto mode = options.scopeMapPrintingMode;
  if (mode == ScopeMapPrintingMode::None)
    return;
  SourceFileScope *scopeMap = file.getScopeMap();
  assert(scopeMap && "no scope map?!");
  if (mode == ScopeMapPrintingMode::Expanded)
    scopeMap->fullyExpand();
  else
    assert((mode == ScopeMapPrintingMode::Lazy) &&
           "Unknown ScopeMapPrintingMode");
  scopeMap->dump(out);
}

void CompilerInstance::createASTContext() {
  if (!astContext)
    astContext = ASTContext::create(srcMgr, diagEng);
}

void CompilerInstance::printASTContextMemoryUsage(Step step) const {
  debug_os << "ASTContext memory usage after ";
  switch (step) {
  case Step::Parsing:
    debug_os << "parsing";
    break;
  case Step::Sema:
    debug_os << "semantic analysis";
    break;
  }
  debug_os << ": ";
  llvm::write_integer(debug_os, astContext->getTotalMemoryUsed(), 0,
                      llvm::IntegerStyle::Number);
  debug_os << " bytes\n";
}

bool CompilerInstance::doParsing(SourceFile &file) {
  Parser parser(*astContext, file);
  parser.parseSourceFile();
  if (options.dumpParse)
    file.dump(llvm::outs());
  if (options.printMemUsage)
    printASTContextMemoryUsage(Step::Parsing);
  if (options.parseOnly)
    dumpScopeMaps(llvm::outs(), file);
  return !diagEng.hadAnyError();
}

bool CompilerInstance::doSema(SourceFile &file) {
  performSema(file);
  if (options.dumpAST)
    file.dump(llvm::outs());
  if (options.printMemUsage)
    printASTContextMemoryUsage(Step::Parsing);
  dumpScopeMaps(llvm::outs(), file);
  return !diagEng.hadAnyError();
}