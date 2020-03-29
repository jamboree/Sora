//===--- IRGen.hpp - IR Generator Interface ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//
// The MLIR IR Generator - a Work In Progress!
//  TODO:
//    - Scoped ValueDecl -> mlir::Value map to know about alive/dead values
//      (see visitBlockStmt in IRGenStmt.cpp)
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTNode.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/IR/Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
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

class IRGen {
  bool debugInfoEnabled;

  /// A Cache of Type -> mlir Types
  llvm::DenseMap<TypeBase *, mlir::Type> typeCache;
  /// A Cache fo Function -> mlir Function Operation
  llvm::DenseMap<FuncDecl *, mlir::FuncOp> funcCache;

public:
  IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt, bool enableDebugInfo);

  //===- Generation Entry Points ------------------------------------------===//

  /// Generates IR for \p sf, returning the MLIR Module.
  void genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule);

  /// Generates IR for a function's body.
  /// This does nothing if the function's body has already been generated.
  mlir::FuncOp genFunctionBody(FuncDecl *func);

  /// Generates IR for an ASTNode.
  void genNode(ASTNode node, mlir::OpBuilder builder) {
    if (Expr *expr = node.dyn_cast<Expr *>())
      return (void)genExpr(expr, builder);
    if (Stmt *stmt = node.dyn_cast<Stmt *>())
      return genStmt(stmt, builder);
    if (Decl *decl = node.dyn_cast<Decl *>())
      return genDecl(decl, builder);
    llvm_unreachable("Unknown ASTNode kind");
  }

  /// Generates IR for an Expression.
  mlir::Value genExpr(Expr *expr, mlir::OpBuilder builder);

  /// Generates IR for a Statement.
  void genStmt(Stmt *stmt, mlir::OpBuilder builder);

  /// Generates IR for a Block Statement.
  /// This is simply an extra entry point so files don't have to include
  /// Stmt.hpp just to implicitly convert BlockStmt into Stmts.
  void genStmt(BlockStmt *stmt, mlir::OpBuilder builder);

  /// Generates IR for a Declaration.
  void genDecl(Decl *decl, mlir::OpBuilder builder);

  //===- Helpers/Conversion Functions -------------------------------------===//

  /// \returns the MLIR FuncOp for \p func, creating it if needed.
  /// Note that this does not generate the body of the function. For that, see
  /// \c genFunctionBody.
  mlir::FuncOp getFuncOp(FuncDecl *func);

  /// \returns the MLIR Location for \p node's SourceRange, or mlir::UnknownLoc
  /// if debug info is disabled.
  mlir::Location getMLIRLoc(ASTNode node);

  /// \returns the MLIR Location for \p loc, or mlir::UnknownLoc if debug info
  /// is disabled.
  mlir::Location getMLIRLoc(SourceLoc loc);

  /// \returns the MLIR Location for \p range, or mlir::UnknownLoc if debug info
  /// is disabled.
  mlir::Location getMLIRLoc(SourceRange range);

  /// \returns the MLIR Type equivalent of \p type.
  /// Note that this can NOT lower types that contain Null types.
  mlir::Type getMLIRType(Type type);

  /// \returns the MLIR Type equivalent of \p expr's type.
  /// Note that this can NOT lower types that contain Null types.
  mlir::Type getMLIRType(Expr *expr);

  /// \returns the MLIR Identifier for \p str
  mlir::Identifier getMLIRIdentifier(StringRef str);

  /// \returns the MLIR Identifier for \p str
  mlir::Identifier getMLIRIdentifier(const char *str) {
    return getMLIRIdentifier(StringRef(str));
  }

  /// \returns the MLIR Identifier for \p ident
  mlir::Identifier getMLIRIdentifier(Identifier ident) {
    return getMLIRIdentifier(ident.c_str());
  }

  //===--------------------------------------------------------------------===//

  ASTContext &astCtxt;
  const SourceManager &srcMgr;
  mlir::MLIRContext &mlirCtxt;
  ir::SoraDialect &soraDialect;
};

/// A small common base between IR Generators for different AST hierarchies
/// (Expr, Stmt, etc.)
class IRGeneratorBase {
public:
  IRGeneratorBase(IRGen &irGen)
      : irGen(irGen), astCtxt(irGen.astCtxt), mlirCtxt(irGen.mlirCtxt) {}

  IRGen &irGen;
  ASTContext &astCtxt;
  mlir::MLIRContext &mlirCtxt;
};
} // namespace sora