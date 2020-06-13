//===--- SIRGenStmt.cpp - Statement SIR Generation --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Stmt.hpp"

using namespace sora;

//===- StmtGenerator ------------------------------------------------------===//

namespace {
class StmtGenerator : public SIRGeneratorBase,
                      public StmtVisitor<StmtGenerator> {
  using Visitor = StmtVisitor<StmtGenerator>;

public:
  StmtGenerator(SIRGen &sirGen, mlir::OpBuilder &builder)
      : SIRGeneratorBase(sirGen), builder(builder) {
    func = builder.getInsertionBlock()
               ->getParent()
               ->getParentOfType<mlir::FuncOp>();
    assert(func && "Not inserting in a function?!");
  }

  mlir::OpBuilder &builder;
  mlir::FuncOp func;

  using Visitor::visit;

  void visit(BlockStmtElement node) {
    if (Stmt *stmt = node.dyn_cast<Stmt *>())
      return visit(stmt);
    if (Expr *expr = node.dyn_cast<Expr *>())
      return (void)sirGen.genExpr(builder, expr);
    if (Decl *decl = node.dyn_cast<Decl *>())
      return sirGen.genDecl(builder, decl);
  }

  void visitContinueStmt(ContinueStmt *stmt) {
    llvm_unreachable("Unimplemented - visitContinueStmt");
  }

  void visitBreakStmt(BreakStmt *stmt) {
    llvm_unreachable("Unimplemented - visitBreakStmt");
  }

  void visitReturnStmt(ReturnStmt *stmt) {
    // if we have no result, then it's easy - just generate a return op.
    if (!stmt->hasResult())
      return (void)builder.create<mlir::ReturnOp>(getNodeLoc(stmt));

    mlir::Value result = sirGen.genExpr(builder, stmt->getResult());

    // If the function has no result (= we simplified the return type to void),
    // discard the result.
    if (func.getNumResults() == 0)
      builder.create<mlir::ReturnOp>(getNodeLoc(stmt));
    else
      builder.create<mlir::ReturnOp>(getNodeLoc(stmt), result);
  }

  void visitBlockStmt(BlockStmt *stmt, bool isFree = true);

  void visitIfStmt(IfStmt *stmt) {
    llvm_unreachable("Unimplemented - visitIfStmt");
  }

  void visitWhileStmt(WhileStmt *stmt) {
    llvm_unreachable("Unimplemented - visitWhileStmt");
  }
};
} // namespace

void StmtGenerator::visitBlockStmt(BlockStmt *stmt, bool isFree) {
  // For free blocks, we have to emit a BlockOp whose region shall contain
  // the BlockStmt's contents.
  Optional<mlir::OpBuilder::InsertPoint> insertionPoint;
  if (isFree) {
    // Emit a BlockOp, save the insertion point (which is now after the
    // BlockOp), create a new BB inside the BlockOp's region and set the
    // insertion point to the start of that BB.
    sir::BlockOp blockOp = builder.create<sir::BlockOp>(getNodeLoc(stmt));
    insertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&blockOp.getEntryBlock());
  }

  // Emit the statements
  for (BlockStmtElement elem : stmt->getElements())
    visit(elem);

  // If we modified the insertion point, restore it.
  if (insertionPoint)
    builder.restoreInsertionPoint(*insertionPoint);
}

//===- SIRGen -------------------------------------------------------------===//

void SIRGen::genFunctionBody(mlir::OpBuilder &builder, BlockStmt *stmt) {
  StmtGenerator(*this, builder).visitBlockStmt(stmt, /*isFree*/ false);
}

mlir::Location SIRGen::getNodeLoc(Stmt *stmt) {
  return mlir::OpaqueLoc::get(stmt, getLoc(stmt->getLoc()));
}