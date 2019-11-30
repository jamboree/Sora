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
/// This class handles the bulk of expression type checking.
///
/// Most of the walk is done in post-order (the children are visited first), but
/// some nodes are also visited in pre-order (see walkToExprPre/Post).
///
/// visit() methods return nullptr on error, else they return an expression with
/// a valid type (this can be the same expression, or another one (if the
/// expression must be replaced with something else (e.g. resolving
/// UnresolvedDeclRefExprs)))
class ExprChecker : public ASTChecker,
                    public ASTWalker,
                    public ExprVisitor<ExprChecker, Expr *> {
public:
  /// The constraint system for this expression
  ConstraintSystem &cs;
  /// The DeclContext in which this expression appears
  DeclContext *dc;
  /// The set of DiscardExpr that are valid.
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

  /// If \p expr is a DiscardExpr, adds it to \c validDiscardExprs.
  /// Else, if \p expr is a ParenExpr or TupleExpr, recurses with the
  /// subexpression(s).
  void findValidDiscardExprs(Expr *expr);

  std::pair<Action, Expr *> walkToExprPre(Expr *expr) override {
    // For assignements, we must whitelist DiscardExprs in the LHS.
    if (BinaryExpr *binary = dyn_cast<BinaryExpr>(expr))
      if (binary->isAssignementOp())
        findValidDiscardExprs(binary->getLHS());
    return {Action::Continue, expr};
  }

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    assert(expr && "expr is null");
    Expr *result = visit(expr);
    /// If the visit() method returned something, it must have a type.
    if (result) {
      /// FIXME: Remove this once ExprChecker is complete
      result->hasType() ? void() : result->setType(ctxt.errorType);
      assert(result->hasType() && "Returned an untyped expression");
      expr = result;
    }
    // Else, if it returned nullptr, it's because the expr isn't valid, so just
    // give it an error type.
    else
      expr->setType(ctxt.errorType);

    assert(expr->getType());

    // If 'expr' is an assignement, clear the set of valid discard exprs.
    if (BinaryExpr *binary = dyn_cast<BinaryExpr>(expr))
      if (binary->isAssignementOp())
        validDiscardExprs.clear();
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

  return nullptr;
}

void ExprChecker::findValidDiscardExprs(Expr *expr) {
  // allow _ =
  if (DiscardExpr *discard = dyn_cast<DiscardExpr>(expr))
    validDiscardExprs.insert(discard);
  // allow (_) = ...
  if (ParenExpr *paren = dyn_cast<ParenExpr>(expr))
    findValidDiscardExprs(paren->getSubExpr());
  // allow (_, _) = ..., ((_, _), (_, _)) = ..., etc.
  if (TupleExpr *tuple = dyn_cast<TupleExpr>(expr))
    for (Expr *elem : tuple->getElements())
      findValidDiscardExprs(elem);
}

Expr *ExprChecker::visitUnresolvedMemberRefExpr(UnresolvedMemberRefExpr *expr) {
  return expr;
}

Expr *ExprChecker::visitDeclRefExpr(DeclRefExpr *expr) {
  llvm_unreachable("Expr visited twice!");
}

Expr *ExprChecker::visitDiscardExpr(DiscardExpr *expr) {
  /// This DiscardExpr is not valid, diagnose it.
  if (validDiscardExprs.count(expr) == 0) {
    diagnose(expr->getLoc(), diag::illegal_discard_expr);
    return nullptr;
  }
  /// This DiscardExpr is valid, give it a TypeVariableType with an LValue.
  TypeVariableType *tv = cs.createGeneralTypeVariable();
  expr->setType(LValueType::get(tv));
  return expr;
}

Expr *ExprChecker::visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
  // integer literals have a "float" type variable that'll default to i32 unless
  // unified with another floating point type.
  expr->setType(cs.createIntegerTypeVariable());
  return expr;
}

Expr *ExprChecker::visitFloatLiteralExpr(FloatLiteralExpr *expr) {
  // float literals have a "float" type variable that'll default to f32 unless
  // unified with another floating point type.
  expr->setType(cs.createFloatTypeVariable());
  return expr;
}

Expr *ExprChecker::visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
  // boolean literals always have a "bool" type.
  expr->setType(ctxt.boolType);
  return expr;
}

Expr *ExprChecker::visitNullLiteralExpr(NullLiteralExpr *expr) {
  // The "null" literal has its own type
  expr->setType(ctxt.nullType);
  return expr;
}

Expr *ExprChecker::visitErrorExpr(ErrorExpr *expr) {
  llvm_unreachable("Expr checked twice!");
}

Expr *ExprChecker::visitCastExpr(CastExpr *expr) {
  tc.resolveTypeLoc(expr->getTypeLoc(), getSourceFile());
  // TODO: unify subexpression type w/ cast goal type w/ a special comparator
  // (that checks if the conversion is legit)
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

//===- TypeChecker --------------------------------------------------------===//

class ExprCheckerEpilogue : public ASTChecker, public ASTWalker {
public:
  ConstraintSystem &cs;

  ExprCheckerEpilogue(TypeChecker &tc, ConstraintSystem &cs)
      : ASTChecker(tc), cs(cs) {}

  void simplifyTypeOfExpr(Expr *expr) {
    Type type = expr->getType();
    assert(type && "untyped expr");
    if (!type->hasTypeVariable())
      return;
    bool isAmbiguous = false;
    expr->setType(
        // Called when a TV has no substitution
        cs.simplifyType(expr->getType(), [&](TypeVariableType *type) -> Type {
          TypeVariableInfo &info = TypeVariableInfo::get(type);
          // If it's an integer type variable, default to i32
          if (info.isIntegerTypeVariable())
            return ctxt.i32Type;
          // If it's a float type variable, default to f32
          if (info.isFloatTypeVariable())
            return ctxt.f32Type;
          // Else, we can't do much more, just tag the type as being ambiguous
          isAmbiguous = true;
          return nullptr;
        }));
    // FIXME: Make the diagnostic more precise depending on the circumstances
    if (isAmbiguous)
      diagnose(expr->getLoc(), diag::type_is_ambiguous_without_more_ctxt);
  }

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    simplifyTypeOfExpr(expr);
    return {true, expr};
  }
};

} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *TypeChecker::typecheckExpr(Expr *expr, DeclContext *dc, Type ofType) {
  assert(expr && dc);
  // Create a constraint system for this expression
  ConstraintSystem system(*this);
  // Check the expression
  expr = expr->walk(ExprChecker(*this, system, dc)).second;
  assert(expr && "ExprChecker returns null?");
  // Perform the epilogue (simplify types, diagnose inference errors)
  expr = expr->walk(ExprCheckerEpilogue(*this, system)).second;
  assert(expr && "ExprCheckerEpilogue returns null?");
  return expr;
}