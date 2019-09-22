//===--- main.cpp - Entry point of the Sora Executable ----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/DiagnosticsCommon.hpp"
#include "Sora/Common/InitLLVM.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Driver/Driver.hpp"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

using namespace sora;
using namespace llvm::opt;

int main(int argc, char **argv) {
  PROGRAM_START(argc, argv);
  // Create the driver
  Driver driver(llvm::outs());
  // Parse the arguments
  bool hadError;
  ArrayRef<const char *> rawArgs = ArrayRef<const char *>(argv, argv + argc);
  // Remove the first argument from the array, as it's the executable's path
  // and we aren't interested in it. Plus, the Driver will consider it as an
  // input if we don't remove it.
  rawArgs = rawArgs.slice(1);
  // Ask the driver to parse the argument
  InputArgList inputArgs = driver.parseArgs(rawArgs, hadError);
  /// Stop here if an error occured during the parsing of the arguments (because
  /// the arguments cannot be trusted)
  if (hadError)
    return EXIT_FAILURE;
  // Handle immediate arguments and return if we don't have anything else
  // to do after that.
  if (!driver.handleImmediateArgs(inputArgs))
    return EXIT_SUCCESS;
  // Try to create the CompilerInstance
  auto compilerInstance = driver.tryCreateCompilerInstance(inputArgs);
  if (!compilerInstance)
    return EXIT_FAILURE;
  compilerInstance->dump(llvm::outs());
  return compilerInstance->run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
