//===--- EntryPoints.hpp - Sora Libraries Entry Points ----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//
// This file contains entry points for most of the Sora libraries.
// Some libraries are only accessible through this file (e.g. Sema) as they have
// a limited number of entry points, but other libraries, such as Parser, are
// accessible through this file *and* offer additional APIs in
// /include/Sora/(libname).
//
// In general, prefer using those entry points unless you have a good reason to
// use the library's API directly.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace sora {
class SourceFile;

//===- AST - Abstract Syntax Tree Library ---------------------------------===//

/// In debug mode, verifies the source file \p sf and exits if the AST is not
/// well-formed.
///
/// \param sf the SourceFile to check.
/// \param isChecked whether the AST is type-checked.
void verify(SourceFile &sf, bool isChecked);

//===- Parser - Parsing Library -------------------------------------------===//

/// Parses the content of \p sf
///
/// Memory will be allocated using the SourceFile's ASTContext, and Diagnostics
/// will be emitted using the ASTContext's DiagnosticEngine.
void parseSourceFile(SourceFile &sf);

//===- Sema - Semantic Analysis Library -----------------------------------===//

/// Performs Semantic Analysis on \p sf
///
/// Memory will be allocated using the SourceFile's ASTContext, and Diagnostics
/// will be emitted using the ASTContext's DiagnosticEngine.
void performSema(SourceFile &sf);

//===- IRGen - IR Generation Library --------------------------------------===//

/// Adds the MLIR Dialects necessary for IRGen to MLIR's Dialect Registry.
/// Currently, this adds the Sora and LLVM Dialects.
///
/// This function must be called once before attempting any IR Generation
/// operation. Ideally it should only be called once, but calling it multiple
/// times is not an issue (it'll just be a no-op).
void registerMLIRDialects();

/// Creates a MLIR Module for \p sf using \p mlirCtxt
mlir::ModuleOp createMLIRModule(mlir::MLIRContext &mlirCtxt, SourceFile &sf);

/// Performs IR Generation on \p sf using \p mlirCtxt.
///   \param mlirCtxt the MLIRContext to use
///   \param mlirModule the MLIR Module in which the contents of \p sf will be
///     emitted.
///   \param sf the target SourceFile
///   \param enableDebugInfo Whether Debug information will be emitted.
void performIRGen(mlir::MLIRContext &mlirCtxt, mlir::ModuleOp &mlirModule,
                  SourceFile &sf, bool enableDebugInfo);

//===----------------------------------------------------------------------===//

} // namespace sora