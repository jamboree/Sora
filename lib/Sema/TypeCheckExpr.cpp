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
#include "llvm/Support/raw_ostream.h" // TODO: remove this

using namespace sora;

//===- ExprChecker --------------------------------------------------------===//

namespace {
/// This class handles the bulk of expression type checking.
///
/// Most of the walk is done in post-order (the children are visited first), but
/// some nodes are also visited in pre-order (see walkToExprPre/Post).
///
/// visit() methods return nullptr on error, else they return an expression with
/// a valid type (this can be the same expression, or another one that'll
/// replace it in the AST (e.g. resolving UnresolvedDeclRefExprs))
///
/// TODO-list of things that can be improved:
///     - This is a large class, it'd be a good idea to split it between
///     multiple files if possible.
///     - Handling of DiscardExprs could be improved, perhaps they shouldn't be
///     checked by this class but should be checked by a separate visitor
///     instead?
class ExprChecker : public ASTChecker,
                    public ASTWalker,
                    public ExprVisitor<ExprChecker, Expr *> {
public:
  /// The constraint system for this expression
  ConstraintSystem &cs;
  /// The DeclContext in which this expression appears
  DeclContext *dc;
  /// The set of DiscardExpr that are valid
  llvm::SmallPtrSet<DiscardExpr *, 4> validDiscardExprs;

  ExprChecker(TypeChecker &tc, ConstraintSystem &cs, DeclContext *dc)
      : ASTChecker(tc), cs(cs), dc(dc) {}

  SourceFile &getSourceFile() const {
    assert(dc && "no DeclContext?");
    SourceFile *sf = dc->getParentSourceFile();
    assert(sf && "no source file");
    return *sf;
  }

  DeclRefExpr *resolveDeclRefExpr(UnresolvedDeclRefExpr *udre,
                                  ValueDecl *resolved) {
    DeclRefExpr *expr = new (ctxt) DeclRefExpr(udre, resolved);
    Type type = resolved->getValueType();
    // Everything is immutable by default, except 'mut' VarDecls.
    if (VarDecl *var = dyn_cast<VarDecl>(resolved))
      if (var->isMutable())
        type = LValueType::get(type);
    expr->setType(type);
    return expr;
  }

  TupleElementExpr *resolveTupleElementExpr(UnresolvedMemberRefExpr *umre,
                                            TupleType *tuple, unsigned idx,
                                            bool isMutable) {
    TupleElementExpr *expr = new (ctxt) TupleElementExpr(umre, idx);
    Type type = tuple->getElement(idx);
    if (isMutable)
      type = LValueType::get(type);
    expr->setType(type);
    return expr;
  }

  Expr *handleCast(CastExpr *cast) {
    tc.resolveTypeLoc(cast->getTypeLoc(), getSourceFile());
    return cast;
  }

  /// \returns whether we can emit a diagnostic involving \p type
  bool canDiagnose(Type type) { return !type->hasErrorType(); }

  /// \returns whether we can emit a diagnostic involving \p expr
  bool canDiagnose(Expr *expr) { return canDiagnose(expr->getType()); }

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
      assert(!isa<UnresolvedExpr>(result) && "Returned an UnresolvedExpr");
      assert(result->hasType() && "Returned an untyped expression");
      expr = result;
    }
    // Else, if it returned nullptr, it's because the expr isn't valid, so give
    // it an ErrorType.
    else {
      // If it's an UnresolvedExpr that couldn't be resolved, replace it with an
      // ErrorExpr first. This is needed because UnresolvedExpr are freed after
      // Semantic Analysis completes.
      if (UnresolvedExpr *ue = dyn_cast<UnresolvedExpr>(expr))
        expr = new (ctxt) ErrorExpr(ue);
      expr->setType(ctxt.errorType);
    }
    assert(expr && expr->hasType());
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
    return resolveDeclRefExpr(expr, decl);

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
  // The type on which we'll perform lookup
  Type lookupTy = expr->getBase()->getType();
  // The identifier we'll be looking for
  Identifier memberID = expr->getMemberIdentifier();
  SourceLoc memberIDLoc = expr->getMemberIdentifierLoc();

  // If the type that we want to look into contains an ErrorType, stop here.
  if (lookupTy->hasErrorType())
    return nullptr;

  bool isMutable = false;

  // #1
  //  - Check that the operator used was the correct one
  //  - Find the type in which lookup will be performed
  {
    // If the '->' operator was used, the base must be a reference type
    ReferenceType *ref = lookupTy->getRValue()->getAs<ReferenceType>();
    if (expr->isArrow()) {
      if (!ref) {
        // It's not a reference type, emit a diagnostic: we want to point at the
        // operator & highlight the base expression fully.
        diagnose(expr->getOpLoc(), diag::base_operand_of_arrow_isnt_ref_ty)
            .highlight(expr->getBase()->getSourceRange());
        return nullptr;
      }
      // It's indeed a reference type, and we'll look into the pointee type, and
      // the mutability depends on the mutability of the ref.
      lookupTy = ref->getPointeeType();
      isMutable = ref->isMut();
    }
    // If the '.' operator was used, the base must be a value type.
    else {
      if (ref) {
        // It's a reference type, emit a diagnostic: we want to point at the
        // operator & highlight the base expression fully.
        diagnose(expr->getOpLoc(), diag::base_operand_of_dot_isnt_value_ty)
            .highlight(expr->getBase()->getSourceRange());
        return nullptr;
      }
      // It's a value type, we'll look into the type minus its LValue if
      // present, and the mutability depends on whether the type carries an
      // LValue or not.
      if (lookupTy->isLValue()) {
        lookupTy = lookupTy->getRValue();
        isMutable = true;
      }
    }
  }

  // #2
  //  - Canonicalize the lookup type
  lookupTy = lookupTy->getCanonicalType();

  // #3
  //  - Perform the actual lookup

  // Helper that emits a diagnostic (lookupTy doesn't have member) and returns
  // nullptr.
  auto memberNotFound = [&]() -> Expr * {
    // Emit a diagnostic: We want to highlight the value and member name
    // fully.
    diagnose(memberIDLoc, diag::value_of_type_has_no_member_named, lookupTy,
             memberID)
        .highlight(expr->getBase()->getSourceRange())
        .highlight(expr->getMemberIdentifierLoc());
    return nullptr;
  };

  // A. Looking into a tuple
  if (TupleType *tuple = lookupTy->getAs<TupleType>()) {
    Optional<unsigned> lookupResult = tuple->lookup(memberID);
    if (lookupResult == None)
      return memberNotFound();
    return resolveTupleElementExpr(expr, tuple, *lookupResult, isMutable);
  }
  // (that's it, there are only tuples in sora for now)

  return memberNotFound();
}

Expr *ExprChecker::visitDeclRefExpr(DeclRefExpr *expr) {
  llvm_unreachable("Expr visited twice!");
}

Expr *ExprChecker::visitDiscardExpr(DiscardExpr *expr) {
  /// If this DiscardExpr is not valid, diagnose it.
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

  Type fromType = expr->getSubExpr()->getType();
  Type toType = expr->getTypeLoc().getType();

  assert(toType && "toType is null?");
  // The type of the CastExpr is always the 'to' type, even if the from/toType
  // contains an ErrorType.
  expr->setType(toType);

  // Don't bother checking if the cast is legit if there's an ErrorType
  // somewhere.
  if (fromType->hasErrorType() || toType->hasErrorType())
    return expr;

  // Check if cast is legal
  UnificationOptions options;
  // Use a custom comparator
  options.typeComparator = [&](CanType a, CanType b) -> bool {
    // Allow conversion between integer widths/signedness
    // Allow conversion between float widths as well
    // TODO: int to bool conversion?
    return (a->is<IntegerType>() && b->is<IntegerType>()) ||
           (a->is<FloatType>() && b->is<FloatType>());
  };

  if (!cs.unify(fromType, toType, options)) {
    // Use the simplified "fromType" in the diagnostic
    Type fromTypeSimplified = cs.simplifyType(fromType);
    // Emit the diagnostic
    diagnose(expr->getSubExpr()->getLoc(), diag::cannot_cast_value_of_type,
             fromTypeSimplified, toType)
        .highlight(expr->getAsLoc())
        .highlight(expr->getTypeLoc().getSourceRange());
  }

  return expr;
}

Expr *ExprChecker::visitTupleElementExpr(TupleElementExpr *expr) {
  llvm_unreachable("Expr checked twice!");
}

Expr *ExprChecker::visitTupleExpr(TupleExpr *expr) {
  // If this is an empty tuple, just give it the empty tuple type '()'
  if (expr->isEmpty()) {
    expr->setType(TupleType::getEmpty(ctxt));
    return expr;
  }
  // Create a TupleType of the element's types
  assert(expr->getNumElements() > 1 && "Single Element Tuple Shouldn't Exist!");
  SmallVector<Type, 8> tupleEltsTypes;
  // A tuple's type is an LValue only if every element is also an LValue.
  bool isLValue = true;
  for (Expr *elt : expr->getElements()) {
    Type eltType = elt->getType();
    tupleEltsTypes.push_back(eltType);
    isLValue &= eltType->hasErrorType();
  }
  Type type = TupleType::get(ctxt, tupleEltsTypes);
  if (isLValue)
    type = LValueType::get(type);
  expr->setType(type);
  return expr;
}

Expr *ExprChecker::visitParenExpr(ParenExpr *expr) {
  // The type of this expr is the same as its subexpression, preserving LValues.
  expr->setType(expr->getSubExpr()->getType());
  return expr;
}

Expr *ExprChecker::visitCallExpr(CallExpr *expr) { return nullptr; }

Expr *ExprChecker::visitConditionalExpr(ConditionalExpr *expr) {
  return nullptr;
}

Expr *ExprChecker::visitForceUnwrapExpr(ForceUnwrapExpr *expr) {
  return nullptr;
}

Expr *ExprChecker::visitBinaryExpr(BinaryExpr *expr) { return nullptr; }

Expr *ExprChecker::visitUnaryExpr(UnaryExpr *expr) { return nullptr; }

//===- ExprCheckerEpilogue ------------------------------------------------===//

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
    expr->setType(cs.simplifyType(expr->getType()));
    // FIXME: Make the diagnostic more precise depending on the circumstances
    if (isAmbiguous)
      diagnose(expr->getLoc(), diag::type_is_ambiguous_without_more_ctxt);
  }

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    simplifyTypeOfExpr(expr);
    // Some exprs might require a bit of post processing
    if (CastExpr *cast = dyn_cast<CastExpr>(expr)) {
      // Check whether this cast is useful or not
      Type fromType = cast->getSubExpr()->getType()->getRValue();
      if (fromType->getCanonicalType() == cast->getType()->getCanonicalType())
        cast->setIsUseless();
    }
    return {true, expr};
  }
};

} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *TypeChecker::performExprChecking(ConstraintSystem &cs, Expr *expr,
                                       DeclContext *dc) {
  assert(expr && "Expr* is null");
  assert(dc && "DeclContext* is null");
  expr = expr->walk(ExprChecker(*this, cs, dc)).second;
  assert(expr && "ExprChecker returns a null Expr*?");
  return expr;
}

Expr *TypeChecker::performExprCheckingEpilogue(ConstraintSystem &cs,
                                               Expr *expr) {
  assert(expr && "Expr* is null");
  expr = expr->walk(ExprCheckerEpilogue(*this, cs)).second;
  assert(expr && "ExprChecker returns a null Expr*?");
  return expr;
}

Expr *TypeChecker::typecheckExpr(Expr *expr, DeclContext *dc, Type ofType) {
  // Create a constraint system for this expression
  ConstraintSystem system(*this);

  // Check the expression
  expr = performExprChecking(system, expr, dc);

  // TODO: Unify this expr's type with oftype (will need a handler for when
  // unification fails)

  // Perform the epilogue (simplify types, diagnose inference errors)
  expr = performExprCheckingEpilogue(system, expr);

  return expr;
}