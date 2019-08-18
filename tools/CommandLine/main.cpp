//===--- main.cpp - Entry point of the Sora Executable ----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/DiagnosticsCommon.hpp"
#include "Sora/Common/InitLLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Driver/Driver.hpp"
#include <cstdio>

using namespace sora;
using namespace llvm::opt;

int main(int argc, char **argv) {
  PROGRAM_START(argc, argv);
  // Create the driver
  SourceManager srcMgr;
  DiagnosticEngine diags(srcMgr, llvm::outs());
  Driver driver(diags);
  // Parse the arguments
  bool hadError;
  InputArgList &inputArgs =
      driver.parseArgs(ArrayRef(argv, argv + argc), hadError);
  // Stop here if an error occured during the parsing of the arguments
  // (because the arguments cannot be trusted)
  if (hadError)
    return EXIT_FAILURE;
  // Handle immediate arguments and return if we don't have anything else
  // to do after that.
  if (!driver.handleImmediateArgs(inputArgs))
    return EXIT_SUCCESS;

  return EXIT_SUCCESS;
}
