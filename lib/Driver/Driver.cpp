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
#include "Sora/EntryPoints.hpp"
#include "Sora/IR/Dialect.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"

using namespace sora;
using namespace llvm::opt;

Driver::Driver(raw_ostream &out)
    : driverDiags(driverDiagsSrcMgr), optTable(createSoraOptTable()) {
  driverDiags.createConsumer<PrintingDiagnosticConsumer>(llvm::outs());

  // Register the Sora MLIR Dialect
  mlir::registerDialect<SoraDialect>();
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

  // "x-only" options, from the least to the most restrictive
  if (argList.hasArg(opt::OPT_parse_only))
    setStopAfterStep(Step::Parsing);
  if (argList.hasArg(opt::OPT_sema_only))
    setStopAfterStep(Step::Sema);

  Arg *emitArg = nullptr;
  auto checkCanUseEmitArg = [&](Arg *arg) {
    if (!emitArg) {
      emitArg = arg;
      return true;
    }

    Arg *arg1 = arg;
    Arg *arg2 = emitArg;
    if (arg1->getIndex() < arg2->getIndex())
      std::swap(arg1, arg2);
    diagnose(diag::cannot_use_arg1_with_arg2, arg1->getSpelling(),
             arg2->getSpelling());
    success = false;
    return false;
  };

  if (Arg *arg = argList.getLastArg(opt::OPT_emit_raw_ir)) {
    if (checkCanUseEmitArg(arg)) {
      options.desiredOutput = CompilerOutputType::IR;
      setStopAfterStep(Step::IRGen);
    }
  }

  if (Arg *arg = argList.getLastArg(opt::OPT_emit_ir)) {
    if (checkCanUseEmitArg(arg)) {
      options.desiredOutput = CompilerOutputType::IR;
      setStopAfterStep(Step::IRTransform);
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
  DUMP_BOOL(options.verifyModeEnabled);
  DUMP_BOOL(options.printMemUsage);
  out << "options.scopeMapPrintingMode: ";
  switch (options.scopeMapPrintingMode) {
  case ScopeMapPrintingMode::None:
    out << "None\n";
    break;
  case ScopeMapPrintingMode::Lazy:
    out << "Lazy\n";
    break;
  case ScopeMapPrintingMode::Expanded:
    out << "Expanded\n";
    break;
  }
  out << "options.stopAfterStep: ";
  switch (options.stopAfterStep) {
  case Step::Parsing:
    out << "Parsing\n";
    break;
  case Step::Sema:
    out << "Sema\n";
    break;
  case Step::IRGen:
    out << "IRGen\n";
    break;
  case Step::IRTransform:
    out << "IRTransform\n";
    break;
  case Step::LLVMGen:
    out << "LLVMGen\n";
    break;
  }
  out << "options.desiredOutput: ";
  switch (options.desiredOutput) {
  case CompilerOutputType::IR:
    out << "IR\n";
    break;
  case CompilerOutputType::Executable:
    out << "Executable\n";
    break;
  }
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
  auto isDone = [&](Step lastStep) {
    return !success || options.stopAfterStep == lastStep;
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
  if (isDone(Step::Parsing)) {
    dumpScopeMaps(debug_os, sf);
    return finish();
  }

  // Perform Semantic Analysis
  success = doSema(sf);
  dumpScopeMaps(debug_os, sf);
  if (isDone(Step::Sema))
    return finish();

  // Perform IRGen
  mlir::MLIRContext mlirCtxt;
  mlir::ModuleOp mlirModule = createMLIRModule(mlirCtxt, sf);
  success = doIRGen(mlirCtxt, mlirModule, sf);
  if (isDone(Step::IRGen)) {
    if (success && options.desiredOutput == CompilerOutputType::IR)
      emitIRModule(mlirModule);
    return finish();
  }

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
  case Step::IRGen:
    debug_os << "ir generation";
    break;
  case Step::IRTransform:
    debug_os << "ir lowering/optimization";
    break;
  case Step::LLVMGen:
    debug_os << "llvm ir generation";
    break;
  }
  debug_os << ": ";
  llvm::write_integer(debug_os, astContext->getTotalMemoryUsed(), 0,
                      llvm::IntegerStyle::Number);
  debug_os << " bytes\n";
}

bool CompilerInstance::doParsing(SourceFile &file) {
  parseSourceFile(file);
  if (options.dumpParse)
    file.dump(debug_os);
  if (options.printMemUsage)
    printASTContextMemoryUsage(Step::Parsing);
  return !diagEng.hadAnyError();
}

bool CompilerInstance::doSema(SourceFile &file) {
  performSema(file);
  if (options.dumpAST)
    file.dump(debug_os);
  if (options.printMemUsage)
    printASTContextMemoryUsage(Step::Sema);
  return !diagEng.hadAnyError();
}

bool CompilerInstance::doIRGen(mlir::MLIRContext &mlirContext,
                               mlir::ModuleOp &mlirModule, SourceFile &file) {
  performIRGen(mlirContext, mlirModule, file);
  return !diagEng.hadAnyError();
}

void CompilerInstance::emitIRModule(mlir::ModuleOp &mlirModule) {
  // FIXME: Once I have a working -o option, write it to a file.
  mlirModule.print(llvm::outs());
}