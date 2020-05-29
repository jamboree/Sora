//===--- IRGenStmt.cpp - Statement IR Generation ----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Stmt.hpp"

using namespace sora;

//===- StmtIRGenerator ----------------------------------------------------===//

namespace {
class StmtIRGenerator : public IRGeneratorBase,
                        public StmtVisitor<StmtIRGenerator> {
  using Visitor = StmtVisitor<StmtIRGenerator>;

public:
  StmtIRGenerator(IRGen &irGen, mlir::OpBuilder &builder)
      : IRGeneratorBase(irGen), builder(builder) {}

  mlir::OpBuilder &builder;

  using Visitor::visit;

  void visit(BlockStmtElement node) {
    if (Stmt *stmt = node.dyn_cast<Stmt *>())
      return visit(stmt);
    if (Expr *expr = node.dyn_cast<Expr *>())
      return (void)irGen.genExpr(builder, expr);
    if (Decl *decl = node.dyn_cast<Decl *>())
      return irGen.genDecl(builder, decl);
  }

  void visitContinueStmt(ContinueStmt *stmt) {
    llvm_unreachable("Unimplemented - visitContinueStmt");
  }

  void visitBreakStmt(BreakStmt *stmt) {
    llvm_unreachable("Unimplemented - visitBreakStmt");
  }

  void visitReturnStmt(ReturnStmt *stmt) {
    llvm_unreachable("Unimplemented - visitReturnStmt");
  }

  void visitBlockStmt(BlockStmt *stmt);

  void visitIfStmt(IfStmt *stmt) {
    llvm_unreachable("Unimplemented - visitIfStmt");
  }

  void visitWhileStmt(WhileStmt *stmt) {
    llvm_unreachable("Unimplemented - visitWhileStmt");
  }
};
} // namespace

void StmtIRGenerator::visitBlockStmt(BlockStmt *stmt) {
  // FIXME: Shouldn't this have a dedicated region or something?
  for (BlockStmtElement elem : stmt->getElements())
    visit(elem);
}

//===- IRGen --------------------------------------------------------------===//

void IRGen::genStmt(mlir::OpBuilder &builder, Stmt *stmt) {
  builder.createBlock(builder.getBlock()->getParent());
  StmtIRGenerator(*this, builder).visit(stmt);
}

void IRGen::genStmt(mlir::OpBuilder &builder, BlockStmt *stmt) {
  return genStmt(builder, (Stmt *)stmt);
}

mlir::Location IRGen::getNodeLoc(Stmt *stmt) {
  return mlir::OpaqueLoc::get(stmt, getFileLineColLoc(stmt->getLoc()));
}