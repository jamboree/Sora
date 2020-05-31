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

class IRGen {
  bool debugInfoEnabled;

  /// A Cache of Type -> mlir Types
  llvm::DenseMap<TypeBase *, mlir::Type> typeCache;
  /// A Cache fo Function -> mlir Function Operation
  llvm::DenseMap<FuncDecl *, mlir::FuncOp> funcCache;
  /// Maps VarDecls to their current value.
  llvm::ScopedHashTable<VarDecl *, mlir::Value> vars;

public:
  //===- Helper Classes ---------------------------------------------------===//

  /// Represents a BlockStmt's scope. This doubles as a ScopedHashTableScope for
  /// the var values.
  class BlockScope : llvm::ScopedHashTableScope<VarDecl *, mlir::Value> {
  public:
    BlockScope(IRGen &irGen) : ScopedHashTableScope(irGen.vars) {}
  };

  //===- Constructor ------------------------------------------------------===//

  IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt, bool enableDebugInfo);

  //===- Generation Entry Points ------------------------------------------===//

  /// Generates a variable declaration \p decl, optionally using \p value as its
  /// initial value.
  ///
  /// If \p value is null, then a sora.create_default_value is used as the value
  /// of the variable.
  void genVarDecl(mlir::OpBuilder &builder, VarDecl *decl,
                  mlir::Value value = {});

  /// Generates IR for \p sf, returning the MLIR Module.
  void genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule);

  /// Generates IR for a function.
  mlir::FuncOp genFunction(FuncDecl *func);

  /// Generates IR for an Expression.
  mlir::Value genExpr(mlir::OpBuilder &builder, Expr *expr);

  /// Generates IR for a Pattern \p pattern. \p value is an optional argument to
  /// assign a value to the pattern.
  void genPattern(mlir::OpBuilder &builder, Pattern *pattern,
                  mlir::Value value = {});

  /// Generates IR for a function's body \p stmt.
  /// This considers that \p stmt is not a free block, and will not emit a
  /// sora.block operation for it.
  void genFunctionBody(mlir::OpBuilder &builder, BlockStmt *stmt);

  /// Generates IR for a Declaration.
  void genDecl(mlir::OpBuilder &builder, Decl *decl);

  //===- Helpers/Conversion Functions -------------------------------------===//

  /// Sets the value of \p decl to \p value.
  void setVarValue(VarDecl *decl, mlir::Value value);

  /// \returns the current value of \p decl
  mlir::Value getVarValue(VarDecl *decl);

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

  template <typename Ty> mlir::Type getType(Ty &&value) {
    return irGen.getType(value);
  }

  IRGen &irGen;
  ASTContext &astCtxt;
  mlir::MLIRContext &mlirCtxt;
};
} // namespace sora