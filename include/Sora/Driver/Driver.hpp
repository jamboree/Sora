//===--- Driver.hpp - Compiler Driver ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the Driver and CompilerInstance classes.
// Essentially, the driver is the glue holding all the parts of the compiler
// together. It orchestrates compilation, handles command-line options, etc.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTContext.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/ArgList.h"
#include <memory>

namespace sora {
/// Represents an instance of the compiler. This owns the main singletons
/// (SourceManager, ASTContext, DiagnosticEngine, etc.) and orchestrates
/// the compilation process.
///
/// Currently, you can't run the same compiler instance more than once
/// (simply because it's not needed)
class CompilerInstance {
  friend class Driver;
  CompilerInstance() = default;

public:
  /// A step in the compilation process
  enum class Step : uint8_t {
    /// Parsing, done by the Lexer and Parser components
    Parsing,
    /// Semantic Analysis, done by Sema
    Sema,
    /// The last step of the process
    Last = Sema
  };

  struct {
    /// If "true", dumps the AST after parsing.
    /// Honored by doParsing()
    bool dumpParse = false;
    /// If "true", dumps the AST after semantic analysis.
    /// Honored by doSema()
    bool dumpAST = false;
    /// If "true", the compiler will stop after the parsing step.
    /// Honored by run()
    bool parseOnly = false;
  } options;

  /// Handles command-line options
  void handleOptions(llvm::opt::InputArgList &argList);

  /// Loads each input file in \p argList
  /// \returns false if an error occured while loading a file.
  bool loadInputs(llvm::opt::InputArgList &argList);

  /// Dumps the state of this CompilerInstance
  void dump(raw_ostream &out) const;

  /// Loads an file into the SourceManager. If the file can't be loaded, a
  /// diagnostic is emitted.
  /// \param the absolute path of the file
  /// \returns a valid BufferID if the file was loaded successfully, false
  /// otherwise.
  BufferID loadFile(StringRef filepath);

  /// Runs this CompilerInstance.
  ///
  /// This can only be called once per CompilerInstance.
  ///
  /// \param stopAfter stops the compilation process after that step (default =
  /// Step::Last)
  /// \returns true if compilation was successful, false otherwise.
  bool run(Step stopAfter = Step::Last);

  /// \returns the set of input buffers
  ArrayRef<BufferID> getInputBuffers() const;

  /// Utility function to emit CompilerInstance diagnostics.
  template <typename... Args>
  InFlightDiagnostic
  diagnose(TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagEng.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> astContext = nullptr;

private:
  /// Creates the ASTContext (if needed)
  void createASTContext();

  /// Whether this CompilerInstance was ran at least once.
  bool ran = false;
  /// The BufferIDs of the input files
  SmallVector<BufferID, 4> inputBuffers;

  /// Performs the parsing step
  /// \returns true if parsing was successful, false otherwise.
  bool doParsing();

  /// Performs the semantic analysis step
  /// \returns true if sema was successful, false otherwise.
  bool doSema();
};

/// This is a high-level compiler driver. It handles command-line options and
/// creation of CompilerInstances.
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

  /// (Tries to) create a compiler instance
  /// \returns the created compiler instance, or nullptr on error.
  std::unique_ptr<CompilerInstance>
  tryCreateCompilerInstance(llvm::opt::InputArgList &options);

  /// Utility function to emit driver diagnostics.
  template <typename... Args>
  InFlightDiagnostic
  diagnose(TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return driverDiags.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  /// \returns the option table
  const llvm::opt::OptTable &getOptTable() const {
    assert(optTable && "no option table");
    return *optTable;
  }

private:
  /// Driver Diagnostics
  DiagnosticEngine &driverDiags;
  /// The option table
  std::unique_ptr<llvm::opt::OptTable> optTable;
};
} // namespace sora