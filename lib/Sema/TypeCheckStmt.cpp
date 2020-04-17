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
#include "Sora/AST/Types.hpp"

using namespace sora;

//===- StmtChecker --------------------------------------------------------===//

namespace {
class StmtChecker : public ASTCheckerBase, public StmtVisitor<StmtChecker> {
public:
  SmallVector<WhileStmt *, 4> loops;

  DeclContext *dc;

  StmtChecker(TypeChecker &tc, DeclContext *dc) : ASTCheckerBase(tc), dc(dc) {}

  void visitDecl(Decl *decl) { tc.typecheckDecl(decl); }

  void visitExpr(Expr *expr) { tc.typecheckExpr(expr, dc); }

  FuncDecl *getCurrentFunction() {
    // Currently, the DC should always be the function.
    assert(dc->isFuncDecl() && "Not in a function?!");
    return cast<FuncDecl>(dc);
  }

  void visitContinueStmt(ContinueStmt *stmt) {
    if (loops.empty())
      diagnose(stmt->getLoc(), diag::is_only_allowed_inside_loop, "continue");
  }

  void visitBreakStmt(BreakStmt *stmt) {
    if (loops.empty())
      diagnose(stmt->getLoc(), diag::is_only_allowed_inside_loop, "break");
  }

  void visitReturnStmt(ReturnStmt *stmt) {
    FuncDecl *fn = getCurrentFunction();
    Type fnRetTy = fn->getReturnTypeLoc().getType();
    assert(fnRetTy && "fn signature not checked yet?!");

    // If the 'return' has an expression, check that its type is correct
    if (stmt->hasResult()) {
      auto onError = [&](Type a, Type b) {
        // Check that 'b' is indeed the 'ofType' (fnRetTy in this case),
        // else the diag will not be correct
        assert(b.getPtr() == fnRetTy.getPtr());
        diagnose(stmt->getLoc(), diag::cannot_convert_ret_expr, a, b);
      };

      Expr *result = stmt->getResult();
      result = tc.typecheckExpr(result, dc, fnRetTy, onError);
      stmt->setResult(result);
    }
    // If the 'return' has no expression, check that the function returns 'void'
    else {
      if (!fnRetTy->isVoidType())
        diagnose(stmt->getLoc(), diag::non_void_fn_should_return_value);
    }
  }

  void checkBlockStmtElement(BlockStmtElement &elt) {
    if (elt.is<Stmt *>())
      visit(elt.get<Stmt *>());
    else if (elt.is<Expr *>())
      elt = tc.typecheckExpr(elt.get<Expr *>(), dc);
    else if (elt.is<Decl *>())
      tc.typecheckDecl(elt.get<Decl *>());
    else
      llvm_unreachable("unknown BlockStmtElement kind");
  }

  FuncDecl *getAsFuncDecl(BlockStmtElement elt) {
    if (Decl *decl = elt.dyn_cast<Decl *>())
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
    for (BlockStmtElement elt : stmt->getElements()) {
      if (FuncDecl *fn = getAsFuncDecl(elt)) {
        assert(fn->isLocal() && "Function should be local!");
        tc.typecheckDecl(fn);
      }
    }
    // And do another one where we typecheck the rest
    for (BlockStmtElement &elt : stmt->getElements()) {
      if (!getAsFuncDecl(elt))
        checkBlockStmtElement(elt);
    }
  }

  void checkCondition(ConditionalStmt *stmt) {
    StmtCondition cond = stmt->getCond();
    if (Expr *expr = cond.getExprOrNull()) {
      cond = tc.typecheckBooleanCondition(expr, dc);
      stmt->setCond(cond);
    }
    else if (LetDecl *decl = cond.getLetDecl())
      tc.typecheckLetCondition(decl);
  }

  void visitIfStmt(IfStmt *stmt) {
    checkCondition(stmt);
    visitBlockStmt(stmt->getThen());
    if (Stmt *elseStmt = stmt->getElse())
      visit(elseStmt);
  }

  void visitWhileStmt(WhileStmt *stmt) {
    checkCondition(stmt);

    loops.push_back(stmt);
    visitBlockStmt(stmt->getBody());
    loops.pop_back();
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::typecheckStmt(Stmt *stmt, DeclContext *dc) {
  assert(stmt && dc);
  StmtChecker(*this, dc).visit(stmt);
}