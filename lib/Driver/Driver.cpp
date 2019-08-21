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

using namespace sora;
using namespace llvm::opt;

Driver::Driver(DiagnosticEngine &driverDiags)
    : driverDiags(driverDiags), optTable(createSoraOptTable()) {}

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
  // Diagnose missing arguments
  if (missingArgCount) {
    diagnose(diag::missing_argv, argList.getArgString(missingArgIndex),
             missingArgCount);
    hadError = true;
  }
  return argList;
}

bool Driver::handleImmediateArgs(InputArgList &options) {
  // handle -h/-help
  if (options.hasArg(opt::OPT_help)) {
    optTable->PrintHelp(llvm::outs(), "sorac [options] <inputs>",
                        "Sora Compiler");
    return false;
  }
  return true;
}

std::unique_ptr<CompilerInstance>
Driver::createCompilerInstance(llvm::opt::InputArgList &options) {
  // To make make_unique work with the private constructor, we must
  // use a small trick.
  struct CompilerInstanceCreater : public CompilerInstance {
    using CompilerInstance::CompilerInstance;
  };
  // TODO: Handle command-line options like
  //    -dump-ast, -dump-ast=(raw | checked)
  return llvm::make_unique<CompilerInstanceCreater>();
}

/*
  TODO:
    - unit-test options stuff
    - class CompilerInstance (friend class Driver)
      - bool run() // return true if success, false otherwise.
          - doParse
          - doSema
          - doIRGen
          - doIROpt
          - doLLVMIRGen
          - doLLVMIROpt
          - doLLVMCodeGen
*/

BufferID CompilerInstance::loadFile(StringRef filepath) {
  if (auto result = llvm::MemoryBuffer::getFile(filepath)) {
    if (auto buffer = srcMgr.giveBuffer(std::move(*result))) {
      inputBuffers.push_back(buffer);
      return buffer;
    }
  }
  return BufferID();
}

bool CompilerInstance::run(Step stopAfter) {
  assert(!ran && "already ran this CompilerInstance!");
  ran = true;
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

bool CompilerInstance::doParsing() {
  // TODO
  if (options.dumpRawAST) {
    // TODO: Dump Raw AST
  }
  return true;
}

bool CompilerInstance::doSema() {
  // TODO
  if (options.dumpCheckedAST) {
    // TODO: Dump Raw AST
  }
  return true;
}