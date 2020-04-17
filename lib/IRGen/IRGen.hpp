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
  mlir::Type getIRType(Type type);

  /// \returns the MLIR Type equivalent of \p expr's type.
  /// Note that this can NOT lower types that contain Null types.
  mlir::Type getIRType(Expr *expr);

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
  ir::SoraDialect &soraDialect;
};

/// A small common base between IR Generators for different AST hierarchies
/// (Expr, Stmt, etc.)
class IRGeneratorBase {
public:
  IRGeneratorBase(IRGen &irGen)
      : irGen(irGen), astCtxt(irGen.astCtxt), mlirCtxt(irGen.mlirCtxt) {}

  template <typename Ty> mlir::Location getNodeLoc(Ty &&value) {
    return irGen.getNodeLoc(value);
  }

  template <typename Ty> mlir::Type getIRType(Ty &&value) {
    return irGen.getIRType(value);
  }

  IRGen &irGen;
  ASTContext &astCtxt;
  mlir::MLIRContext &mlirCtxt;
};
} // namespace sora