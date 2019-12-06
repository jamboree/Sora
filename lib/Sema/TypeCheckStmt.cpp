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
class StmtChecker : public ASTChecker, public StmtVisitor<StmtChecker> {
public:
  DeclContext *dc;

  StmtChecker(TypeChecker &tc, DeclContext *dc) : ASTChecker(tc), dc(dc) {}

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

  void checkNode(ASTNode &node) {
    if (node.is<Stmt *>())
      tc.typecheckStmt(node.get<Stmt *>(), dc);
    else if (node.is<Expr *>())
      node = tc.typecheckExpr(node.get<Expr *>(), dc);
    else if (node.is<Decl *>())
      tc.typecheckDecl(node.get<Decl *>());
    else
      llvm_unreachable("unknown ASTNode kind");
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
        checkNode(node);
    }
  }

  /// \returns a pair: the first element is true if \p expr is a valid "if let"
  /// initializer, false otherwise. The second element is non-null only if the
  /// first element if false and contains the type that should be used to emit
  /// the diagnostic.
  static std::pair<bool, Type> isValidIfLetInitializer(Expr *expr) {
    Type type = expr->getType()->getRValue();
    // If it doesn't have a "Maybe" type it's never okay.
    if (!type->getCanonicalType()->is<MaybeType>())
      return {false, type};
    // If it has a "Maybe" type, but it originates from an
    // "ImplicitMaybeConversionExpr", it's not right either.
    {
      Expr *cur = expr;
      while (auto implicitConv =
                 dyn_cast<ImplicitConversionExpr>(cur->ignoreParens())) {
        if (isa<ImplicitMaybeConversionExpr>(implicitConv)) {
          Type subExprTy = implicitConv->getSubExpr()->getType()->getRValue();
          /*
          assert(!subExprTy->getCanonicalType()->is<MaybeType>() &&
                 "The expr still has a 'maybe' type even without the "
                 "ImplicitMaybeConversionExpr?");
                 */
          return {false, subExprTy};
        }
        cur = implicitConv->getSubExpr();
      }
    }
    // Else it's okay
    return {true, nullptr};
  }

  void checkCondition(ConditionalStmt *stmt) {
    StmtCondition cond = stmt->getCond();
    if (Expr *expr = cond.getExprOrNull()) {
      cond = tc.typecheckCondition(expr, dc);
      stmt->setCond(cond);
    }
    else if (LetDecl *decl = cond.getLetDecl()) {
      // "if let x" implicitly looks inside "maybe" types, so wrap the LetDecl's
      // pattern in an implicit MaybeValuePattern.
      {
        Pattern *letPat = decl->getPattern();
        letPat = new (ctxt) MaybeValuePattern(letPat, /*isImplicit*/ true);
        decl->setPattern(letPat);
      }
      // Type-check the declaration now
      tc.typecheckDecl(decl);
      // To use this construct, the 'let' decl must have an initializer
      Expr *init = decl->getInitializer();
      if (!init) {
        diagnose(decl->getLetLoc(),
                 diag::variable_binding_in_cond_requires_initializer)
            .fixitInsertAfter(decl->getEndLoc(), "= <expression>");
        return;
      }
      // Check the type of the initializer: it must be a "maybe" type.
      bool isValid;
      Type diagTy;
      std::tie(isValid, diagTy) = isValidIfLetInitializer(init);
      if (isValid && canDiagnose(diagTy))
        diagnose(init->getLoc(), diag::cond_binding_must_have_maybe_type,
                 diagTy);
    }
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