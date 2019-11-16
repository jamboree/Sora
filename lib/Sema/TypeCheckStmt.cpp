//===--- TypeCheckStmt.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Statement Semantic Analysis
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Stmt.hpp"

using namespace sora;

//===- StmtChecker --------------------------------------------------------===//

namespace {
class StmtChecker : public StmtVisitor<StmtChecker> {
public:
  TypeChecker &tc;
  DeclContext *dc;

  StmtChecker(TypeChecker &tc, DeclContext *dc) : tc(tc), dc(dc) {}

  void visitDecl(Decl *decl) { tc.typecheckDecl(decl); }

  void visitExpr(Expr *expr) { tc.typecheckExpr(expr, dc); }

  void visitContinueStmt(ContinueStmt *stmt) {
    // TODO
  }

  void visitBreakStmt(BreakStmt *stmt) {
    // TODO
  }

  void visitReturnStmt(ReturnStmt *stmt) {
    // TODO
    if (stmt->hasResult())
      stmt->setResult(tc.typecheckExpr(stmt->getResult(), dc));
  }

  ASTNode checkNode(ASTNode node) {
    if (node.is<Stmt *>())
      tc.typecheckStmt(node.get<Stmt *>(), dc);
    else if (node.is<Expr *>())
      tc.typecheckExpr(node.get<Expr *>(), dc);
    else if (node.is<Decl *>())
      tc.typecheckDecl(node.get<Decl *>());
    else
      llvm_unreachable("unknown ASTNode kind");
    return node;
  }

  FuncDecl *getAsFuncDecl(ASTNode node) {
    if (Decl *decl = node.dyn_cast<Decl *>())
      if (FuncDecl *fn = dyn_cast<FuncDecl>(decl))
        return fn;
    return nullptr;
  }

  /// Typechecking a BlockStmt is done in 2 passes.
  /// The first one checks FuncDecls only. The second one checks the rest.
  /// This is needed to support forward-referencing declarations, e.g.
  /// \verbatim
  /// func foo(x: i32) -> i32 {
  ///   return bar(x)
  ///   func bar(x: i32) -> i32 { return x*2 }
  /// }
  /// \endverbatim
  ////
  /// If we were to typecheck everything in one pass, we'd crash because bar
  /// hasn't been checked yet (= doesn't have a type) when we check the return
  /// statement.
  void visitBlockStmt(BlockStmt *stmt) {
    // Do a first pass where we typecheck FuncDecls only.
    // Their body will be checked right away.
    for (ASTNode node : stmt->getElements()) {
      if (FuncDecl *fn = getAsFuncDecl(node)) {
        assert(fn->isLocal() && "Function should be local!");
        tc.typecheckDecl(fn);
      }
    }
    // And do another one where we typecheck the rest
    for (ASTNode &node : stmt->getElements()) {
      if (!getAsFuncDecl(node))
        node = checkNode(node);
    }
  }

  void checkCondition(ConditionalStmt *stmt) {
    StmtCondition cond = stmt->getCond();
    if (Expr *expr = cond.getExprOrNull()) {
      cond = tc.typecheckExpr(expr, dc);
      stmt->setCond(cond);
    }
    else if (LetDecl *decl = cond.getLetDecl())
      tc.typecheckDecl(decl);
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
} // namespace

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::typecheckStmt(Stmt *stmt, DeclContext *dc) {
  assert(stmt && dc);
  StmtChecker(*this, dc).visit(stmt);
}