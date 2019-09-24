//===--- Driver.cpp ---------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Driver/Driver.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/DiagnosticsDriver.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Driver/Options.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/FileSystem.h"

using namespace sora;
using namespace llvm::opt;

Driver::Driver(raw_ostream &out)
    : driverDiags(driverDiagsSrcMgr, out), optTable(createSoraOptTable()) {}

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
    optTable->PrintHelp(llvm::outs(), "sorac [options] <inputs>",
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
  options.dumpAST = false; /*TODO*/
  options.dumpParse = argList.hasArg(opt::OPT_dump_parse);
  options.parseOnly = argList.hasArg(opt::OPT_parse_only);
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
    } else
      out << "invalid";
    out << ")\n";
  }
  /// Dump options
  DUMP_BOOL(options.dumpParse);
  DUMP_BOOL(options.dumpAST);
  DUMP_BOOL(options.parseOnly);
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
    diagnose(diag::no_input_files);
    return false;
  }

  bool success = true;
  // Helper function, returns true if we can continue, false otherwise.
  auto canContinue = [&](Step currentStep) {
    return success && stopAfter != currentStep;
  };
  // Perform parsing
  success = doParsing();
  if (canContinue(Step::Parsing))
    return success;
  // Perform Sema
  //  TODO
  return success;
}

ArrayRef<BufferID> CompilerInstance::getInputBuffers() const {
  return inputBuffers;
}

void CompilerInstance::createASTContext() {
  if (!astContext)
    astContext = ASTContext::create(srcMgr, diagEng);
}

bool CompilerInstance::doParsing() {
  assert(!astContext && "ASTContext already created?");
  // Create the root of the AST we'll create.
  createASTContext();
  // TODO
  if (options.dumpParse) {
    // TODO: Dump Raw AST
  }
  return true;
}

bool CompilerInstance::doSema() {
  // TODO
  if (options.dumpAST) {
    // TODO: Dump Raw AST
  }
  return true;
}