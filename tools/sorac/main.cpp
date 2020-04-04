//===--- main.cpp - Entry point of the Sora Executable ----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/InitLLVM.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticsCommon.hpp"
#include "Sora/Driver/Driver.hpp"
#include "Sora/Driver/Options.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

using namespace sora;
using namespace llvm::opt;

int main(int argc, char **argv) {
  PROGRAM_START(argc, argv);
  // FIXME: It'd be great if the DiagEngine worked without a SourceManager, so
  // we don't have to create one for nothing.
  SourceManager driverDiagsSrcMgr;
  DiagnosticEngine driverDiagEngine(driverDiagsSrcMgr);
  driverDiagEngine.createConsumer<PrintingDiagnosticConsumer>(llvm::errs());

  // Create the driver
  Driver driver(driverDiagEngine, "sorac", "Sora Compiler");

  // Prepare the arguments
  ArrayRef<const char *> rawArgs = ArrayRef<const char *>(argv, argv + argc);
  // Remove the first argument from the array, as it's the executable's path
  // and the Driver isn't interested in it.
  rawArgs = rawArgs.slice(1);

  std::unique_ptr<llvm::opt::InputArgList> argList = driver.parseArgs(rawArgs);

  if (driver.hadAnyError())
    return EXIT_FAILURE;

  if (argList->hasArg(opt::OPT_help)) {
    driver.getOptTable().PrintHelp(llvm::outs(), "sorac (options | inputs)",
                                   "Sora Compiler");
    return EXIT_SUCCESS;
  }

  // Try to create the CompilerInstance
  auto compilerInstance = driver.tryCreateCompilerInstance(std::move(argList));
  return (compilerInstance && compilerInstance->run()) ? EXIT_SUCCESS
                                                       : EXIT_FAILURE;
}
