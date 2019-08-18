//===--- Driver.hpp - Compiler Driver ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Sora Driver and CompilerInstance classes.
//
// Note: This is a pretty simple Driver. As it was written for the first version
// of Sora, it should be completely rewritten when the compiler gets
// bigger (e.g. to support parallel/incremental compilation, and to be more
// modular overall).
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/Option/ArgList.h"
#include <memory>

namespace sora {
/// The Sora Driver
///
/// This is a high-level compiler driver. It handles command-line options and
/// handles creation of CompilerInstances.
class Driver {
public:
  /// \param driverDiags the DiagnosticEngine that should be used by the Driver
  /// to report errors. This will never be used with valid SourceLocs, so you
  /// can pass a DiagnosticEngine constructed with an empty SourceManager.
  /// Please note that the DiagnosticEngine/SourceManager combo that will be
  /// used by the various components of the compiler (e.g. Parser, Sema, etc.)
  /// will be created by the Driver itself when compiling files.
  Driver(DiagnosticEngine &driverDiags);

  /// Parses the arguments \p args.
  /// This also diagnoses unknown/ill-formed arguments.
  /// This will not change the state of the driver, but it may emit diagnostics.
  /// \param args the arguments
  /// \param hadError set to true if an error occured, false otherwise.
  /// \returns the InputArgList created.
  llvm::opt::InputArgList parseArgs(ArrayRef<const char *> args,
                                    bool &hadError);

  /// Handles options which should be considered before any compilation
  /// occurs.
  /// \returns true if we need to compile something (build a CompilerInstance),
  /// false otherwise.
  bool handleImmediateArgs(llvm::opt::InputArgList &options);

  /// Utility function to emit driver diagnostics.
  template<typename ... Args>
  InFlightDiagnostic diagnose(TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return driverDiags.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  /// \returns the option table
  const llvm::opt::OptTable &getOptTable() const {
    assert(optTable && "no option table");
    return *optTable;
  }

private:
  /// The option table
  std::unique_ptr<llvm::opt::OptTable> optTable;
  /// Driver Diagnostics
  DiagnosticEngine &driverDiags;
};
} // namespace sora