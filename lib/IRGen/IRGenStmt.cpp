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
  StmtIRGenerator(IRGen &irGen, mlir::OpBuilder builder)
      : IRGeneratorBase(irGen), builder(builder) {}

  mlir::OpBuilder builder;

  using Visitor::visit;

  void visit(ASTNode node) {
    if (Stmt *stmt = node.dyn_cast<Stmt *>())
      return visit(stmt);
    return irGen.genNode(node, builder);
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
  for (ASTNode elem : stmt->getElements())
    visit(elem);
  // TODO: Emit destructor for variables that have to be destroyed.
  // Currently Sora has no destructors but it's probably a good idea
  // to get some basic infrastructure started to make adding destructors
  // as easy as possible afterwards.
}

//===- IRGen --------------------------------------------------------------===//

void IRGen::genStmt(Stmt *stmt, mlir::OpBuilder builder) {
  builder.createBlock(builder.getBlock()->getParent());
  StmtIRGenerator(*this, builder).visit(stmt);
}

void IRGen::genStmt(BlockStmt *stmt, mlir::OpBuilder builder) {
  return genStmt((Stmt *)stmt, builder);
}