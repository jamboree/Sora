//===--- SemaStmt.cpp -------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Statement Semantic Analysis
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Stmt.hpp"

using namespace sora;

//===- StmtChecker --------------------------------------------------------===//

class Sema::StmtChecker : public StmtVisitor<StmtChecker> {
public:
  Sema &sema;
  DeclContext *dc;

  StmtChecker(Sema &sema, DeclContext *dc) : sema(sema), dc(dc) {}

  void visitDecl(Decl *decl) { sema.typecheckDecl(decl); }

  void visitExpr(Expr *expr) { sema.typecheckExpr(expr, dc); }

  void visitContinueStmt(ContinueStmt *stmt) {
    // TODO
  }

  void visitBreakStmt(BreakStmt *stmt) {
    // TODO
  }

  void visitReturnStmt(ReturnStmt *stmt) {
    // TODO
    if (stmt->hasResult())
      stmt->setResult(sema.typecheckExpr(stmt->getResult(), dc));
  }

  ASTNode checkNode(ASTNode node) {
    if (node.is<Stmt *>())
      sema.typecheckStmt(node.get<Stmt *>(), dc);
    else if (node.is<Expr *>())
      sema.typecheckExpr(node.get<Expr *>(), dc);
    else if (node.is<Decl *>())
      sema.typecheckDecl(node.get<Decl *>());
    else
      llvm_unreachable("unknown ASTNode kind");
    return node;
  }

  void visitBlockStmt(BlockStmt *stmt) {
    for (ASTNode node : stmt->getElements())
      node = checkNode(node);
  }

  void checkCondition(ConditionalStmt *stmt) {
    StmtCondition cond = stmt->getCond();
    if (Expr *expr = cond.getExprOrNull()) {
      cond = sema.typecheckExpr(expr, dc);
      stmt->setCond(cond);
    }
    else if (LetDecl *decl = cond.getLetDecl())
      sema.typecheckDecl(decl);
  }

  void visitIfStmt(IfStmt *stmt) {
    checkCondition(stmt);
    visitBlockStmt(stmt->getThen());
    if (Stmt *elseStmt = stmt->getElse())
      visit(elseStmt);
  }

  void visitWhileStmt(WhileStmt *stmt) {
    checkCondition(stmt);
    visitBlockStmt(stmt->getBody());
  }
};

//===- Sema ---------------------------------------------------------------===//

void Sema::typecheckStmt(Stmt *stmt, DeclContext *dc) {
  assert(stmt && dc);
  StmtChecker(*this, dc).visit(stmt);
}