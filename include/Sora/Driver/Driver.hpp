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
class DiagnosticVerifier;
class SourceFile;

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
    /// If "true", the compiler will stop after the parsing step.
    /// Honored by run()
    bool parseOnly = false;
    /// Whether verify mode is enabled.
    /// Honored by run()
    bool verifyModeEnabled = true;
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

  /// \returns true if the file located at \p filePath is already loaded.
  ///         If \p isAbsolute is set to true, the path will not be converted to
  ///         an absolute path before the check if performed.
  bool isLoaded(StringRef filePath, bool isAbsolute = false);

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
  void diagnose(TypedDiag<Args...> diag,
                typename detail::PassArgument<Args>::type... args) {
    diagEng.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> astContext;

private:
  bool hadFileLoadError = false;

  /// Installs the DiagnosticVerifier if verification mode is enabled.
  /// \returns the installed DV, or nullptr if no DV was installed.
  DiagnosticVerifier *installDiagnosticVerifierIfNeeded();

  /// Creates the ASTContext (if needed)
  void createASTContext();

  /// Whether this CompilerInstance was ran at least once.
  bool ran = false;

  /// The BufferIDs of the input files.
  SmallVector<BufferID, 1> inputBuffers;

  SourceFile *createSourceFile(BufferID buffer);

  /// Performs the parsing step on \p file
  /// \returns true if parsing was successful, false otherwise.
  bool doParsing(SourceFile &file);
};

/// This is a high-level compiler driver. It handles command-line options and
/// creation of CompilerInstances.
class Driver {
public:
  /// \param out the stream where the driver will print diagnostics
  /// NOTE: This will not be used to print compilation-related diagnostics
  /// (those are always printed to llvm::outs()).
  Driver(raw_ostream &out);

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
  void diagnose(TypedDiag<Args...> diag,
                typename detail::PassArgument<Args>::type... args) {
    driverDiags.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  /// \returns the option table
  const llvm::opt::OptTable &getOptTable() const {
    assert(optTable && "no option table");
    return *optTable;
  }

private:
  /// Driver Diagnostics
  /// FIXME: It'd be great if the DiagnosticEngine could work without a
  /// SourceManager, so we don't have to create one for nothing.
  SourceManager driverDiagsSrcMgr;
  DiagnosticEngine driverDiags;
  /// The option table
  std::unique_ptr<llvm::opt::OptTable> optTable;
};
} // namespace sora