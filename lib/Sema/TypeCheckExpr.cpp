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

using namespace sora;

namespace {

//===- ExprChecker --------------------------------------------------------===//

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
///     multiple files.
///     - Handling of DiscardExprs isn't ideal, perhaps they should be checked
///     by another class?
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
    assert(sf && "no Source File?");
    return *sf;
  }

  DeclRefExpr *resolveDeclRefExpr(UnresolvedDeclRefExpr *udre,
                                  ValueDecl *resolved) {
    DeclRefExpr *expr = new (ctxt) DeclRefExpr(udre, resolved);
    Type type = resolved->getValueType();
    assert(!type.isNull() && "VarDecl type is null!");
    // Everything is immutable by default, except 'mut' VarDecls.
    if (VarDecl *var = dyn_cast<VarDecl>(resolved))
      if (var->isMutable())
        type = LValueType::get(type);
    expr->setType(type);
    return expr;
  }

  TupleElementExpr *resolveTupleElementExpr(UnresolvedMemberRefExpr *umre,
                                            TupleType *tuple, size_t idx,
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

  Expr *visitDiscardExpr(DiscardExpr *expr);
  Expr *visitIntegerLiteralExpr(IntegerLiteralExpr *expr);
  Expr *visitFloatLiteralExpr(FloatLiteralExpr *expr);
  Expr *visitBooleanLiteralExpr(BooleanLiteralExpr *expr);
  Expr *visitNullLiteralExpr(NullLiteralExpr *expr);
  Expr *visitCastExpr(CastExpr *expr);
  Expr *visitTupleElementExpr(TupleElementExpr *expr);
  Expr *visitTupleExpr(TupleExpr *expr);
  Expr *visitParenExpr(ParenExpr *expr);
  Expr *visitCallExpr(CallExpr *expr);
  Expr *visitConditionalExpr(ConditionalExpr *expr);
  Expr *visitForceUnwrapExpr(ForceUnwrapExpr *expr);
  Expr *visitBinaryExpr(BinaryExpr *expr);
  Expr *visitUnaryExpr(UnaryExpr *expr);

  Expr *visitDeclRefExpr(DeclRefExpr *expr) {
    llvm_unreachable("Expr visited twice!");
  }

  Expr *visitImplicitConversionExpr(ImplicitConversionExpr *expr) {
    llvm_unreachable("Expression checked twice!");
  }

  Expr *visitErrorExpr(ErrorExpr *expr) {
    llvm_unreachable("Expr visited twice!");
  }
};

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

Expr *ExprChecker::visitUnresolvedMemberRefExpr(UnresolvedMemberRefExpr *expr) {
  Type baseTy = expr->getBase()->getType();
  Identifier memberID = expr->getMemberIdentifier();
  SourceLoc memberIDLoc = expr->getMemberIdentifierLoc();

  // #0
  //    - Compute the lookup type: it's the canonical type of the base,
  //    simplified.
  Type lookupTy = cs.simplifyType(baseTy->getCanonicalType());

  // If the type that we want to look into contains an ErrorType, stop here.
  if (lookupTy->hasErrorType())
    return nullptr;

  // #1
  //  - Check that the operator used was the correct one
  //  - Find the type in which lookup will be performed
  //  - Find if the source is mutable or not
  bool isMutable = false;
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
      // It's a value type, we'll look into the type (minus LValue if present)
      // and the mutability depends on whether the type carries an LValue or
      // not.
      if (lookupTy->isLValue()) {
        lookupTy = lookupTy->getRValue();
        isMutable = true;
      }
    }
  }

  // #3
  //  - Perform the actual lookup

  // Helper that emits a diagnostic (lookupTy doesn't have member) and returns
  // nullptr.
  auto memberNotFound = [&]() -> Expr * {
    // Emit a diagnostic: We want to highlight the value and member name
    // fully.
    if (canDiagnose(baseTy))
      diagnose(memberIDLoc, diag::value_of_type_has_no_member_named, baseTy,
               memberID)
          .highlight(expr->getBase()->getSourceRange())
          .highlight(expr->getMemberIdentifierLoc());
    return nullptr;
  };

  // A. Looking into a tuple
  if (TupleType *tuple = lookupTy->getAs<TupleType>()) {
    Optional<size_t> lookupResult = tuple->lookup(memberID);
    if (lookupResult == None)
      return memberNotFound();
    return resolveTupleElementExpr(expr, tuple, *lookupResult, isMutable);
  }
  // (that's it, there are only tuples in sora for now)

  return memberNotFound();
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

Expr *ExprChecker::visitCastExpr(CastExpr *expr) {
  // Resolve the type after the 'as'.
  tc.resolveTypeLoc(expr->getTypeLoc(), getSourceFile());

  Type fromType = expr->getSubExpr()->getType();
  Type toType = expr->getTypeLoc().getType();

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
    // Allow "useless" conversions (same type to same type)
    // TODO: int to bool conversion?
    return (a->is<IntegerType>() && b->is<IntegerType>()) ||
           (a->is<FloatType>() && b->is<FloatType>()) || (a == b);
  };
  // Unify the Canonical types
  if (!cs.unify(fromType->getCanonicalType(), toType->getCanonicalType(),
                options)) {
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
  // If the tuple is empty, just give it a () type.
  if (expr->isEmpty()) {
    expr->setType(TupleType::getEmpty(ctxt));
    return expr;
  }
  // Create a TupleType of the element's types
  assert(expr->getNumElements() > 1 && "Single Element Tuple Shouldn't Exist!");
  SmallVector<Type, 8> eltsTypes;
  // A tuple's type is an LValue only if every element is also an LValue.
  bool isLValue = true;
  for (Expr *elt : expr->getElements()) {
    Type eltType = elt->getType();
    eltsTypes.push_back(eltType);
    isLValue &= eltType->isLValue();
  }
  Type type = TupleType::get(ctxt, eltsTypes);
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
  // Fetch the type of the subexpression as an RValue, and simplify it.
  Type subExprType =
      cs.simplifyType(expr->getSubExpr()->getType()->getRValue());

  // Can't check the expression if it has an error type
  if (subExprType->hasErrorType())
    return nullptr;

  // FIXME: Once TypeAliases are implemented, subExprType should be desugared to
  // support things like "type Foo = maybe (), let x: Foo, x!"

  // Check that it's a maybe type
  MaybeType *maybe = subExprType->getAs<MaybeType>();
  if (!maybe) {
    if (canDiagnose(subExprType))
      diagnose(expr->getExclaimLoc(), diag::cannot_force_unwrap_value_of_type,
               subExprType)
          .highlight(expr->getSubExpr()->getSourceRange());
    return nullptr;
  }
  // If it's indeed a maybe type, the type of this expression is the value type
  // of the 'maybe'.
  expr->setType(maybe->getValueType());
  return expr;
}

Expr *ExprChecker::visitBinaryExpr(BinaryExpr *expr) { return nullptr; }

Expr *ExprChecker::visitUnaryExpr(UnaryExpr *expr) { return nullptr; }

//===- ExprCheckerEpilogue ------------------------------------------------===//

class ExprCheckerEpilogue : public ASTChecker, public ASTWalker {
public:
  ConstraintSystem &cs;
  bool canComplain = true;
  Expr *parentWithErrorType = nullptr;

  ExprCheckerEpilogue(TypeChecker &tc, ConstraintSystem &cs)
      : ASTChecker(tc), cs(cs) {}

  /// Simplifies the type of \p expr.
  void simplifyTypeOfExpr(Expr *expr) {
    Type type = expr->getType();
    if (!type->hasTypeVariable())
      return;

    // Whether the type is ambiguous
    bool isAmbiguous = false;

    type = cs.simplifyType(type, &isAmbiguous);
    expr->setType(type);

    if (isAmbiguous && canComplain) {
      // This shouldn't happen with the current iteration of Sora.
      llvm_unreachable("Diagnostic emission for ambiguous expressions is "
                       "currently not supported");
    }
  }

  std::pair<Action, Expr *> walkToExprPre(Expr *expr) override {
    // Mute diagnostics when walking into an Expr with an ErrorType
    if (canComplain && expr->getType()->hasErrorType()) {
      canComplain = false;
      parentWithErrorType = expr;
    }
    return {Action::Continue, expr};
  }

  // Perform simplification in post-order (= children first)
  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    simplifyTypeOfExpr(expr);
    // Some exprs require a bit of post processing
    if (CastExpr *cast = dyn_cast<CastExpr>(expr)) {
      // Check whether this cast is useful or not
      Type fromType = cast->getSubExpr()->getType()->getRValue();
      if (fromType->getCanonicalType() == cast->getType()->getCanonicalType())
        cast->setIsUseless();
    }

    // If this expr is the one that muted diagnostics, unmute diagnostics.
    if (parentWithErrorType == expr) {
      canComplain = true;
      parentWithErrorType = nullptr;
    }

    return {true, expr};
  }
};

} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *
TypeChecker::tryInsertImplicitConversions(ConstraintSystem &cs, Expr *expr,
                                          Type toType,
                                          bool &hasAddedImplicitConversions) {
  assert(toType && expr && expr->hasType());

  // If both types already unify, we don't have anything to do
  if (cs.unify(toType, expr->getType()))
    return expr;

  // TODO: Ignore sugar as well
  toType = toType->getRValue();

  // Check if we can insert an ImplicitMaybeConversionExpr
  if (MaybeType *toMaybe = toType->getAs<MaybeType>()) {
    // If the MaybeType's ValueType is another MaybeType, just recurse first.
    // Perhaps we're facing nested maybe types.
    Type valueType = toMaybe->getValueType();
    if (valueType->getCanonicalType()->is<MaybeType>())
      expr = tryInsertImplicitConversions(cs, expr, valueType,
                                          hasAddedImplicitConversions);
    // Check if the Maybe Type's ValueType unifies w/ the expr's type. If it
    // does, insert the ImplicitMaybeConversionExpr.
    if (cs.unify(toMaybe->getValueType(), expr->getType())) {
      // The Type of the ImplicitMaybeConversionExpr is the subexpression's type
      // (without LValues) wrapped in a MaybeType.
      expr = new (ctxt) ImplicitMaybeConversionExpr(
          expr, MaybeType::get(expr->getType()->getRValue()));
      hasAddedImplicitConversions = true;
    }
  }

#ifndef NDEBUG
  if (hasAddedImplicitConversions)
    assert(cs.unify(toType, expr->getType()) &&
           "Added implicit conversions but still can't unify?");
#endif
  return expr;
}

Expr *TypeChecker::typecheckExpr(
    ConstraintSystem &cs, Expr *expr, DeclContext *dc, Type ofType,
    llvm::function_ref<void(Type, Type)> onUnificationFailure) {
  // Check the expression
  assert(expr && "Expr* is null");
  assert(dc && "DeclContext* is null");
  expr = expr->walk(ExprChecker(*this, cs, dc)).second;
  assert(expr && "ExprChecker returns a null Expr*?");

  // If the expression is expected to be of a certain type, try to make it work.
  if (ofType) {
    bool addedImplicitConversions = false;
    expr = tryInsertImplicitConversions(cs, expr, ofType,
                                        addedImplicitConversions);
    Type exprTy = expr->getType();
    if (!cs.unify(ofType, exprTy)) {
      if (onUnificationFailure)
        onUnificationFailure(cs.simplifyType(exprTy), ofType);
    }
  }

  // Perform the epilogue (simplify types, diagnose inference errors)
  assert(expr && "Expr* is null");
  expr = expr->walk(ExprCheckerEpilogue(*this, cs)).second;
  assert(expr && "ExprChecker returns a null Expr*?");

  return expr;
}

Expr *TypeChecker::typecheckExpr(
    Expr *expr, DeclContext *dc, Type ofType,
    llvm::function_ref<void(Type, Type)> onUnificationFailure) {
  ConstraintSystem cs(*this);
  return typecheckExpr(cs, expr, dc, ofType, onUnificationFailure);
}

//===- ASTChecker ---------------------------------------------------------===//

bool ASTChecker::canDiagnose(Expr *expr) {
  return canDiagnose(expr->getType());
}