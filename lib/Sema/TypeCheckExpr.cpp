//===--- TypeCheckExpr.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Expression Semantic Analysis
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "ConstraintSystem.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Diagnostics/DiagnosticsSema.hpp"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

//===- ExprChecker --------------------------------------------------------===//

namespace {
/// This class handles most of expression type checking
class ExprChecker : public ASTChecker,
                    public ASTWalker,
                    public ExprVisitor<ExprChecker, Expr *> {
public:
  /// The constraint system for this expression
  ConstraintSystem &cs;
  /// The DeclContext in which this expression appears
  DeclContext *dc;
  /// The set of DiscardExpr that are considered valid.
  llvm::SmallPtrSet<DiscardExpr *, 4> validDiscardExprs;

  ExprChecker(TypeChecker &tc, ConstraintSystem &cs, DeclContext *dc)
      : ASTChecker(tc), cs(cs), dc(dc) {}

  SourceFile &getSourceFile() const {
    assert(dc && "no DeclContext?");
    SourceFile *sf = dc->getParentSourceFile();
    assert(sf && "no source file");
    return *sf;
  }

  DeclRefExpr *resolve(UnresolvedDeclRefExpr *udre, ValueDecl *resolved) {
    DeclRefExpr *expr = new (ctxt) DeclRefExpr(udre, resolved);
    expr->setType(resolved->getValueType());
    return expr;
  }

  Expr *handleCast(CastExpr *cast) {
    tc.resolveTypeLoc(cast->getTypeLoc(), getSourceFile());
    return cast;
  }

  /// Given a an assignement expression \p expr, adds every valid DiscardExpr
  /// found in the LHS of the assignement to \c validDiscardExprs
  /// TODO
  void addValidDiscardExprs(BinaryExpr *expr);

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    expr = visit(expr);
    assert(expr && "visit() returned a null expr");
    // FIXME: Remove this & uncomment the line after it once the ExprChecker is
    // complete
    if (expr->getType().isNull())
      expr->setType(ctxt.errorType);
    // assert(expr->getType() && "visit() didn't assign a type to the expr");
    return {true, expr};
  }

  Expr *visitUnresolvedDeclRefExpr(UnresolvedDeclRefExpr *expr);
  Expr *visitUnresolvedMemberRefExpr(UnresolvedMemberRefExpr *expr);
  Expr *visitDeclRefExpr(DeclRefExpr *expr);
  Expr *visitDiscardExpr(DiscardExpr *expr);
  Expr *visitIntegerLiteralExpr(IntegerLiteralExpr *expr);
  Expr *visitFloatLiteralExpr(FloatLiteralExpr *expr);
  Expr *visitBooleanLiteralExpr(BooleanLiteralExpr *expr);
  Expr *visitNullLiteralExpr(NullLiteralExpr *expr);
  Expr *visitErrorExpr(ErrorExpr *expr);
  Expr *visitCastExpr(CastExpr *expr);
  Expr *visitTupleElementExpr(TupleElementExpr *expr);
  Expr *visitTupleExpr(TupleExpr *expr);
  Expr *visitParenExpr(ParenExpr *expr);
  Expr *visitCallExpr(CallExpr *expr);
  Expr *visitConditionalExpr(ConditionalExpr *expr);
  Expr *visitForceUnwrapExpr(ForceUnwrapExpr *expr);
  Expr *visitBinaryExpr(BinaryExpr *expr);
  Expr *visitUnaryExpr(UnaryExpr *expr);
};

Expr *ExprChecker::visitUnresolvedDeclRefExpr(UnresolvedDeclRefExpr *expr) {
  // Perform unqualified value lookup to try to resolve this identifier.
  UnqualifiedValueLookup uvl(getSourceFile());
  uvl.performLookup(expr->getLoc(), expr->getIdentifier());

  // If we have only one result, just resolve the expression.
  if (ValueDecl *decl = uvl.getUniqueResult())
    return resolve(expr, decl);

  // Else, we got an error: emit a diagnostic depending on the situation

  // No result:
  if (uvl.isEmpty()) {
    diagnose(expr->getLoc(), diag::cannot_find_value_in_scope,
             expr->getIdentifier());
  }
  // Multiple results:
  else {
    assert(uvl.results.size() >= 2);
    diagnose(expr->getLoc(), diag::reference_to_value_is_ambiguous,
             expr->getIdentifier());
    for (ValueDecl *candidate : uvl.results)
      diagnose(candidate->getLoc(), diag::potential_candidate_found_here);
  }

  return new (ctxt) ErrorExpr(expr);
}

Expr *ExprChecker::visitUnresolvedMemberRefExpr(UnresolvedMemberRefExpr *expr) {
  return expr;
}

Expr *ExprChecker::visitDeclRefExpr(DeclRefExpr *expr) { return expr; }

Expr *ExprChecker::visitDiscardExpr(DiscardExpr *expr) { return expr; }

Expr *ExprChecker::visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
  return expr;
}

Expr *ExprChecker::visitFloatLiteralExpr(FloatLiteralExpr *expr) {
  return expr;
}

Expr *ExprChecker::visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
  return expr;
}

Expr *ExprChecker::visitNullLiteralExpr(NullLiteralExpr *expr) { return expr; }

Expr *ExprChecker::visitErrorExpr(ErrorExpr *expr) {
  llvm_unreachable("Expr checked twice!");
}

Expr *ExprChecker::visitCastExpr(CastExpr *expr) {
  tc.resolveTypeLoc(expr->getTypeLoc(), getSourceFile());
  // TODO
  return expr;
}

Expr *ExprChecker::visitTupleElementExpr(TupleElementExpr *expr) {
  llvm_unreachable("Expr checked twice!");
}

Expr *ExprChecker::visitTupleExpr(TupleExpr *expr) { return expr; }

Expr *ExprChecker::visitParenExpr(ParenExpr *expr) {
  // The type of this expr is the same as its child.
  expr->setType(expr->getType());
  return expr;
}

Expr *ExprChecker::visitCallExpr(CallExpr *expr) { return expr; }

Expr *ExprChecker::visitConditionalExpr(ConditionalExpr *expr) { return expr; }

Expr *ExprChecker::visitForceUnwrapExpr(ForceUnwrapExpr *expr) { return expr; }

Expr *ExprChecker::visitBinaryExpr(BinaryExpr *expr) { return expr; }

Expr *ExprChecker::visitUnaryExpr(UnaryExpr *expr) { return expr; }
} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *TypeChecker::typecheckExpr(Expr *expr, DeclContext *dc, Type ofType) {
  assert(expr && dc);
  // Create a constraint system for this expression
  ConstraintSystem system(*this);
  // Check the expression
  expr = expr->walk(ExprChecker(*this, system, dc)).second;
  assert(expr && "walk returns null?");
  // TODO: Perform the epilogue (simplify types, diagnose inference errors)
  return expr;
}