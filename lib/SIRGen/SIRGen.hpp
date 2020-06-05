//===--- SIRGen.hpp - Sora IR Generator Interface ---------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/SIR/Dialect.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/Error.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
class OpBuilder;
} // namespace mlir

namespace sora {
class ASTContext;
class BlockStmt;
class DiagnosticEngine;
class FuncDecl;
class SourceFile;
class SourceManager;
class Type;
class VarDecl;

class SIRGen {
  bool debugInfoEnabled;

  /// A Cache of Type -> mlir Types
  llvm::DenseMap<TypeBase *, mlir::Type> typeCache;
  /// A Cache fo Function -> mlir Function Operation
  llvm::DenseMap<FuncDecl *, mlir::FuncOp> funcCache;
  /// Maps VarDecls to the Value containing their addresses.
  llvm::ScopedHashTable<VarDecl *, mlir::Value> varAddresses;

public:
  //===- Helper Classes ---------------------------------------------------===//

  /// Represents a BlockStmt's scope. This doubles as a ScopedHashTableScope for
  /// the var values.
  class BlockScope : llvm::ScopedHashTableScope<VarDecl *, mlir::Value> {
  public:
    BlockScope(SIRGen &irGen) : ScopedHashTableScope(irGen.varAddresses) {}
  };

  //===- Constructor ------------------------------------------------------===//

  SIRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt,
         bool enableDebugInfo);

  //===- Generation Entry Points ------------------------------------------===//

  /// Generates a memory allocation operation for \p decl.
  /// \returns the Value containing the address of \p decl. This value always
  /// has a sir::PointerType, and can be used with sir.load/sir.store.
  mlir::Value genVarDeclAlloc(mlir::OpBuilder &builder, VarDecl *decl);

  /// Generates SIR for \p sf, returning the MLIR Module.
  void genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule);

  /// Generates SIR for a function.
  mlir::FuncOp genFunction(FuncDecl *func);

  /// Generates SIR for an Expression.
  mlir::Value genExpr(mlir::OpBuilder &builder, Expr *expr);

  /// Generates IR for a Pattern \p pattern.
  ///
  /// VarPatterns will call \c genVarDeclAlloc to generate the allocation
  /// operations.
  ///
  /// If \p value has a value, it will be destructured and assigned to the
  /// VarPatterns in \p pattern.
  void genPattern(mlir::OpBuilder &builder, Pattern *pattern,
                  Optional<mlir::Value> value = llvm::None);

  /// Generates IR for a function's body \p stmt.
  /// Note that this does not consider \p stmt to be free, so it won't generate
  /// a sir.block operation.
  void genFunctionBody(mlir::OpBuilder &builder, BlockStmt *stmt);

  /// Generates IR for a Declaration \p decl.
  void genDecl(mlir::OpBuilder &builder, Decl *decl);

  //===- Helpers/Conversion Functions -------------------------------------===//

  /// \returns the Value that contains the address of \p decl.
  /// This value always has a sir::PointerType, and can be used with
  /// sir.load/sir.store.
  mlir::Value getVarDeclAddress(VarDecl *decl);

  /// \returns the MLIR FuncOp for \p func, creating it if needed.
  /// Note that this does not generate the body of the function. For that, see
  /// \c genFunctionBody.
  mlir::FuncOp getFuncOp(FuncDecl *func);

  /// \returns the MLIR Location for \p expr
  mlir::Location getNodeLoc(Expr *expr);

  /// \returns the MLIR Location for \p stmt
  mlir::Location getNodeLoc(Stmt *stmt);

  /// \returns the MLIR Location for \p decl
  mlir::Location getNodeLoc(Decl *decl);

  /// \returns the MLIR Location for \p pattern
  mlir::Location getNodeLoc(Pattern *pattern);

  /// Converts a SourceLoc \p loc to a mlir::FileLineCol loc.
  /// Do not use this for an Op's location when generating IR for a node, use \c
  /// getNodeLoc instead.
  /// This returns mlir::UnknownLoc if debug info is disabled.
  mlir::Location getFileLineColLoc(SourceLoc loc);

  /// \returns the MLIR Type equivalent of \p type.
  /// Note that this can NOT lower types that contain Null types.
  mlir::Type getType(Type type);

  /// \returns the MLIR Type equivalent of \p expr's type.
  /// Note that this can NOT lower types that contain Null types.
  mlir::Type getType(Expr *expr);

  /// \returns the MLIR Identifier for \p str
  mlir::Identifier getIRIdentifier(StringRef str);

  /// \returns the MLIR Identifier for \p str
  mlir::Identifier getIRIdentifier(const char *str) {
    return getIRIdentifier(StringRef(str));
  }

  /// \returns the MLIR Identifier for \p ident
  mlir::Identifier getIRIdentifier(Identifier ident) {
    return getIRIdentifier(ident.c_str());
  }

  //===--------------------------------------------------------------------===//

  ASTContext &astCtxt;
  const SourceManager &srcMgr;
  mlir::MLIRContext &mlirCtxt;
  sir::SIRDialect &sirDialect;
};

/// A small common base class for SIR Generators of different AST hierarchies
/// (Expr, Stmt, etc.)
class SIRGeneratorBase {
public:
  SIRGeneratorBase(SIRGen &sirGen)
      : sirGen(sirGen), astCtxt(sirGen.astCtxt), mlirCtxt(sirGen.mlirCtxt) {}

  template <typename Ty> mlir::Location getNodeLoc(Ty &&value) {
    return sirGen.getNodeLoc(value);
  }

  template <typename Ty> mlir::Type getType(Ty &&value) {
    return sirGen.getType(value);
  }

  SIRGen &sirGen;
  ASTContext &astCtxt;
  mlir::MLIRContext &mlirCtxt;
};
} // namespace sora
