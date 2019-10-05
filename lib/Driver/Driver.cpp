//===--- Driver.cpp ---------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Driver/Driver.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "Sora/Diagnostics/DiagnosticsDriver.hpp"
#include "Sora/Driver/DiagnosticVerifier.hpp"
#include "Sora/Driver/Options.hpp"
#include "Sora/Parser/Parser.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"

using namespace sora;
using namespace llvm::opt;

Driver::Driver(raw_ostream &out)
    : driverDiags(driverDiagsSrcMgr,
                  std::make_unique<PrintingDiagnosticConsumer>(out)),
      optTable(createSoraOptTable()) {}

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
  CI->handleOptions(argList);
  CI->loadInputs(argList);
  return std::move(CI);
}

void CompilerInstance::handleOptions(InputArgList &argList) {
  options.dumpParse = argList.hasArg(opt::OPT_dump_parse);
  options.parseOnly = argList.hasArg(opt::OPT_parse_only);
  options.verifyModeEnabled = argList.hasArg(opt::OPT_verify);
}

bool CompilerInstance::loadInputs(llvm::opt::InputArgList &argList) {
  bool result = true;
  for (const Arg *arg : argList.filtered(opt::OPT_INPUT)) {
    result &= !loadFile(arg->getValue()).isNull();
    arg->claim();
  }
  return result;
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
  DUMP_BOOL(options.parseOnly);
  DUMP_BOOL(options.verifyModeEnabled);
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

bool CompilerInstance::run(Step stopAfter) {
  assert(!ran && "already ran this CompilerInstance!");
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
  // failure. Always call this before returning from this function past this
  // point.
  auto finish = [&]() {
    if (verifier)
      return verifier->finish() && success;
    return success;
  };

  // Create the ASTContext
  createASTContext();

  // Create the source file
  SourceFile *sf = createSourceFile(inputBuffers[0]);

  // Perform parsing
  success = doParsing(*sf);
  if (canContinue(Step::Parsing))
    return finish();
  // TODO: Perform Sema/IRGen/etc.
  return finish();
}

ArrayRef<BufferID> CompilerInstance::getInputBuffers() const {
  return inputBuffers;
}

DiagnosticVerifier *CompilerInstance::installDiagnosticVerifierIfNeeded() {
  if (!options.verifyModeEnabled)
    return nullptr;
  auto verifier = std::make_unique<DiagnosticVerifier>(llvm::outs(), srcMgr,
                                                       diagEng.takeConsumer());
  auto ptr = verifier.get();
  diagEng.setConsumer(std::move(verifier));
  return ptr;
}

void CompilerInstance::createASTContext() {
  if (!astContext)
    astContext = ASTContext::create(srcMgr, diagEng);
}

SourceFile *CompilerInstance::createSourceFile(BufferID buffer) {
  assert(astContext && "No ASTContext?");
  return SourceFile::create(*astContext, buffer, nullptr);
}

bool CompilerInstance::doParsing(SourceFile &file) {
  assert(astContext && "No ASTContext?");
  Parser parser(*astContext, file);
  parser.parseSourceFile();
  if (options.dumpParse)
    file.dump(llvm::outs());
  return true;
}