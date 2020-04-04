//===--- Driver.hpp - Compiler Driver ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the Driver and CompilerInstance classes.
// Essentially, the driver is the glue holding all the parts of the compiler
// together. It orchestrates compilation, handles command-line options, etc.
//
// The driver is still in a prototyping stage. It's rapidly changing, and thus
// the code is clearly not ideal. The driver should be completely rewritten once
// the full compiler pipeline is implemented.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTContext.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace mlir {
class ModuleOp;
class MLIRContext;
} // namespace mlir

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
  CompilerInstance() : diagEng(srcMgr), debug_os(llvm::outs()) {
    diagEng.createConsumer<PrintingDiagnosticConsumer>(llvm::outs());
  }
  CompilerInstance(const CompilerInstance &) = delete;
  CompilerInstance &operator=(const CompilerInstance &) = delete;

public:
  enum class Step : uint8_t {
    /// Parsing, done by the Lexer and Parser components
    Parsing,
    /// Semantic Analysis, done by Sema
    Sema,
    /// IR Generation, done by IRGen
    IRGen,
    /// IR Transformations (optimization & dialect lowering passes)
    /// (Not implemented yet)
    IRTransform,
    /// Generation of LLVM IR from the lowered IR
    /// (Not implemented yet)
    LLVMGen,
    /// The last step of the process
    Last = IRGen
  };

  enum class ScopeMapPrintingMode : uint8_t {
    /// Don't print scope maps
    None,
    /// Print scope maps as they are
    Lazy,
    /// Fully expand the scope map and print it
    Expanded
  };

  enum class CompilerOutputType : uint8_t {
    /// Emit IR (after IRGen or IRTransform, depending on options.stopAfterStep)
    IR,
    /// Emit a linked executable file
    Executable
  };

  struct {
    /// If "true", dumps the raw AST after parsing.
    /// Honored by doParsing()
    bool dumpParse = false;
    /// If "true", dumps the typechecked AST after Sema
    /// Honored by doSema()
    bool dumpAST = false;
    /// The step after which compilation should stop.
    /// By default, stops after the last step.
    Step stopAfterStep = Step::Last;
    /// The desired output of the compiler.
    /// Defaults to a linked executable file.
    CompilerOutputType desiredOutput = CompilerOutputType::Executable;
    /// Whether verify mode is enabled.
    /// Honored by run()
    bool verifyModeEnabled = false;
    /// Whether we should regularly print the memory usage of the ASTContext &
    /// other datastructures.
    bool printMemUsage = false;
    /// Scope Maps printing mode.
    /// Scope maps are printed by dumpScopeMaps().
    /// dumpScopeMaps is called by doSema or by doParsing if parseOnly is true.
    ScopeMapPrintingMode scopeMapPrintingMode = ScopeMapPrintingMode::None;
  } options;

  /// Dumps the state of this CompilerInstance
  void dump(raw_ostream &out) const;


  /// Sets the options.stopAfterStep option if \p step is before the current
  /// value of the option.
  void setStopAfterStep(Step step) {
    if (step < options.stopAfterStep)
      options.stopAfterStep = step;
  }

  /// Runs this CompilerInstance.
  ///
  /// This can only be called once per CompilerInstance.
  ///
  /// \returns true if compilation was successful, false otherwise.
  bool run();

  /// \returns the set of input buffers
  ArrayRef<BufferID> getInputBuffers() const;

  /// Utility function to emit CompilerInstance diagnostics.
  template <typename... Args>
  void diagnose(TypedDiag<Args...> diag,
                typename detail::PassArgument<Args>::type... args) {
    diagEng.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng;
  std::unique_ptr<ASTContext> astContext;

private:
  bool hadFileLoadError = false;

  /// Handles command-line options
  /// \param Driver the Driver to use to emit option parsing-related
  /// diagnostics.
  /// \param argList the argument list.
  /// \returns false if an error occured while handling the options
  bool handleOptions(llvm::opt::InputArgList &argList);

  /// Loads each input file in \p argList
  /// \returns false if an error occured while loading a file.
  bool loadInputs(llvm::opt::InputArgList &argList);

  /// Loads an file into the SourceManager. If the file can't be loaded, a
  /// diagnostic is emitted.
  /// \param the absolute path of the file
  /// \returns a valid BufferID if the file was loaded successfully, false
  /// otherwise.
  BufferID loadFile(StringRef filepath);

  /// Installs the DiagnosticVerifier if verification mode is enabled.
  /// \returns the installed DV, or nullptr if no DV was installed.
  DiagnosticVerifier *installDiagnosticVerifierIfNeeded();

  /// Creates a SourceFile instance for \p buffer
  SourceFile &createSourceFile(BufferID buffer);

  /// Honors options.scopeMapPrintingMode for \p file.
  void dumpScopeMaps(raw_ostream &out, SourceFile &file);

  /// Creates the ASTContext (if needed)
  void createASTContext();

  /// Whether this CompilerInstance was ran at least once.
  bool ran = false;

  /// The BufferIDs of the input files.
  SmallVector<BufferID, 1> inputBuffers;

  /// The output stream used to print debug message/statistics.
  /// Usually llvm::outs();
  raw_ostream &debug_os;

  /// Prints the memory usage of the ASTContext after \p step to debug_os.
  void printASTContextMemoryUsage(Step step) const;

  /// Performs the parsing step on \p file
  /// \returns false if errors were emitted during parsing
  bool doParsing(SourceFile &file);

  /// Performs the semantic analysis step on \p file
  /// \returns false if errors were emitted during semantic analysis
  bool doSema(SourceFile &file);

  /// Performs the IR Generation step on \p file
  /// \returns false if errors were emitted during IR Generation
  bool doIRGen(mlir::MLIRContext &mlirContext, mlir::ModuleOp &mlirModule,
               SourceFile &file);

  /// Emits mlirModule as a product of the compilation process.
  void emitIRModule(mlir::ModuleOp &mlirModule);
};

/// This is a high-level compiler driver. It handles command-line options and
/// creation of CompilerInstances.
class Driver {
  Driver(const Driver &) = delete;
  Driver &operator=(const Driver &) = delete;

public:
  /// \param out the stream where the driver will print diagnostics
  /// \param diagEngine the DiagnosticEngine to be used by the driver. This will
  ///     not be used by the CompilerInstance!
  /// \param driverName the driver name. This should match the executable's
  ///     name.
  /// \param driverDesc the driver's description.
  Driver(raw_ostream &out, DiagnosticEngine &diagEngine, StringRef driverName,
         StringRef driverDesc);

  /// Parses \p args, diagnosing ill-formed arguments and returning the
  /// InputArgList.
  std::unique_ptr<llvm::opt::InputArgList>
  parseArgs(ArrayRef<const char *> args);

  /// \returns true if the Driver encountered any error.
  bool hadAnyError() const { return diagEngine.hadAnyError(); }

  void printHelp(raw_ostream &out, bool showHidden);

  /// Attempts to create a compiler instance using \p args.
  /// \returns the created compiler instance, or nullptr on error.
  std::unique_ptr<CompilerInstance>
  tryCreateCompilerInstance(std::unique_ptr<llvm::opt::InputArgList> args);

  /// Utility function to emit driver diagnostics.
  template <typename... Args>
  void diagnose(TypedDiag<Args...> diag,
                typename detail::PassArgument<Args>::type... args) {
    diagEngine.diagnose<Args...>(SourceLoc(), diag, args...);
  }

  /// \returns the Sora option table
  const llvm::opt::OptTable &getOptTable() const {
    assert(optTable && "no option table");
    return *optTable;
  }

  DiagnosticEngine &diagEngine;
  const StringRef name;
  const StringRef description;

private:
  /// The option table
  std::unique_ptr<llvm::opt::OptTable> optTable;
};
} // namespace sora