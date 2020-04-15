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

//===- Utils --------------------------------------------------------------===//

/// (re-)builds the type of a TupleExpr
static void rebuildTupleExprType(ASTContext &ctxt, TupleExpr *expr) {
  // If the tuple is empty, just give it a () type.
  if (expr->isEmpty()) {
    expr->setType(TupleType::getEmpty(ctxt));
    return;
  }
  // Create a TupleType of the element's types
  assert(expr->getNumElements() > 1 && "Single Element Tuple Shouldn't Exist!");
  SmallVector<Type, 8> eltsTypes;
  eltsTypes.reserve(expr->getNumElements());
  for (Expr *elt : expr->getElements())
    eltsTypes.push_back(elt->getType());
  expr->setType(TupleType::get(ctxt, eltsTypes));
}

/// (re-)builds the type of a ParenExpr
static void rebuildParenExprType(ParenExpr *expr) {
  expr->setType(expr->getSubExpr()->getType());
}

/// Coerces \p expr to an RValue by inserting a LoadExpr if needed.
// \returns the expr that should replace \p expr.
static Expr *coerceToRValue(ASTContext &ctxt, Expr *expr) {
  Type type = expr->getType();

  if (!type->hasLValue())
    return expr;

  // Special-case TupleExpr/ParenExpr - we want to insert the load for their
  // elements, not for the Tuple/Paren.
  if (auto *tuple = dyn_cast<TupleExpr>(expr)) {
    bool changed = false;
    for (size_t k = 0; k < tuple->getNumElements(); ++k) {
      Expr *elt = tuple->getElement(k);
      Expr *result = coerceToRValue(ctxt, elt);

      if (elt->getType().getPtr() != result->getType().getPtr()) {
        changed = true;
        tuple->setElement(k, result);
      }
    }

    if (changed)
      rebuildTupleExprType(ctxt, tuple);
    return tuple;
  }

  if (auto *paren = dyn_cast<ParenExpr>(expr)) {
    paren->setSubExpr(coerceToRValue(ctxt, paren->getSubExpr()));
    rebuildParenExprType(paren);
    return paren;
  }

  // Else, if the expression is just an lvalue, insert a LoadExpr.
  if (type->isLValueType())
    return new (ctxt) LoadExpr(expr, type->getRValueType());

  llvm::dbgs() << "Unhandled expr that has an LValue type (type="
               << type->getString(TypePrintOptions::forDebug())
               << ") in coerceToRValue:\n";
  expr->dump(llvm::dbgs(), ctxt.srcMgr);
  llvm_unreachable("Unhandled expression");
}

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
namespace {
class ExprChecker : public ASTCheckerBase,
                    public ASTWalker,
                    public ExprVisitor<ExprChecker, Expr *> {
public:
  /// The constraint system for this expression
  ConstraintSystem &cs;
  /// The DeclContext in which this expression appears
  DeclContext *dc;
  /// The set of DiscardExpr that are valid/legal.
  llvm::SmallPtrSet<DiscardExpr *, 4> validDiscardExprs;

  ExprChecker(TypeChecker &tc, ConstraintSystem &cs, DeclContext *dc)
      : ASTCheckerBase(tc), cs(cs), dc(dc) {}

  SourceFile &getSourceFile() const {
    assert(dc && "no DeclContext?");
    SourceFile *sf = dc->getParentSourceFile();
    assert(sf && "no Source File?");
    return *sf;
  }

  Expr *coerceToRValue(Expr *expr) { return ::coerceToRValue(ctxt, expr); }

  /// \returns true if \p expr is a mutable reference, false otherwise.
  /// Asserts that \p expr's type is an Reference type.
  bool isMutableReference(Expr *expr) const {
    CanType subExprType = expr->getType()->getCanonicalType();
    assert(subExprType->is<ReferenceType>() && "Not a reference type?!");
    return subExprType->castTo<ReferenceType>()->isMut();
  }

  /// \returns true if \p expr is a mutable LValue, false otherwise.
  /// Asserts that \p expr's type is an LValue type.
  bool isMutableLValue(Expr *expr) const {
    assert(expr->getType()->isLValueType() && "Not an LValue!");
    expr = expr->ignoreParens();
    // DeclRefs = mutable if the DeclRef is.
    if (auto declRef = dyn_cast<DeclRefExpr>(expr)) {
      ValueDecl *decl = declRef->getValueDecl();
      if (auto *var = dyn_cast<VarDecl>(decl))
        return var->isMutable();
      return false;
    }
    // DiscardExprs - always mutable.
    if (isa<DiscardExpr>(expr))
      return true;
    // TupleElement = use the flag
    if (auto tupleElt = dyn_cast<TupleElementExpr>(expr))
      return tupleElt->isMutableLValue();
    // Dereference = mutable if the reference is mutable.
    if (auto unary = dyn_cast<UnaryExpr>(expr)) {
      if (unary->getOpKind() == UnaryOperatorKind::Deref)
        return isMutableReference(unary->getSubExpr());
    }
    return false;
  }

  /// Called when the type-checker is sure that \p udre references \p
  /// resolved. This will check that \p resolved can be referenced from
  /// inside this DeclContext (if it can't, returns nullptr) and, on
  /// success, build a DeclRefExpr. \returns nullptr on failure, a new
  /// DeclRefExpr on success.
  DeclRefExpr *resolveDeclRefExpr(UnresolvedDeclRefExpr *udre,
                                  ValueDecl *resolved) {
    // Check the DeclContext of \p resolved for particular restrictions.
    // These restrictions don't apply on functions because they're values, but
    // not "dynamic" ones (they're always constant)
    if (!isa<FuncDecl>(resolved)) {
      DeclContext *resolvedDC = resolved->getDeclContext();

      if (dc->isFuncDecl() && resolvedDC->isFuncDecl()) {
        // If we're inside a function, and resolvedDC is a function as well,
        // check that the DeclContexts are the same.
        if (dc != resolvedDC) {
          // If they're not, this means that:
          //  1 - We are inside a local function of \p resolvedDC
          assert(resolvedDC->isParentOf(dc) && "Not inside a local function?!");
          //  2 - We are attempting to capture the dynamic environment inside a
          //      local function, which is not allowed (unless resolved is a
          //      func)
          diagnose(udre->getLoc(),
                   diag::cannot_capture_dynamic_env_in_local_func);
          return nullptr;
        }
      }
    }
    assert(resolved->getValueType() && "ValueDecl type is null?");

    DeclRefExpr *expr = new (ctxt) DeclRefExpr(udre, resolved);
    expr->setType(LValueType::get(resolved->getValueType()));
    return expr;
  }

  TupleElementExpr *resolveTupleElementExpr(UnresolvedMemberRefExpr *umre,
                                            TupleType *tuple, size_t idx,
                                            bool isMutableSource,
                                            bool createLValue) {
    TupleElementExpr *expr = new (ctxt) TupleElementExpr(umre, idx);
    Type type = tuple->getElement(idx);

    if (createLValue)
      type = LValueType::get(type);

    expr->setType(type);

    if (isMutableSource) {
      assert(createLValue && "Mutable source is only allowed for LValues!");
      expr->setIsMutableLValue(true);
    }

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
    }
    // Else, if it returned nullptr, it's because the expr isn't valid, so give
    // it an ErrorType.
    else {
      // If it's an UnresolvedExpr that couldn't be resolved, replace it with an
      // ErrorExpr first. This is needed because UnresolvedExpr are freed after
      // Semantic Analysis completes.
      result = expr;
      if (UnresolvedExpr *ue = dyn_cast<UnresolvedExpr>(result))
        result = new (ctxt) ErrorExpr(ue);
      result->setType(ctxt.errorType);
    }
    assert(result && result->hasType());

    return {true, result};
  }

  Expr *tryCoerceExpr(Expr *expr, Type toType) {
    return tc.tryCoerceExpr(cs, expr, toType);
  }

  /// \returns true if \p type is an Integer Type, a Float Type, or a Float/Int
  /// Type Variable;
  bool isNumericTypeOrTypeVariable(Type type) {
    return cs.isFloatTypeOrTypeVariable(type) ||
           cs.isIntegerTypeOrTypeVariable(type);
  }

  /// Diagnoses a bad unary expression using a basic error message.
  /// The simplified type of the subexpression is used the error message (as the
  /// subexpression may contain type variables)
  void diagnoseBadUnaryExpr(UnaryExpr *expr) {
    diagnose(expr->getOpLoc(), diag::cannot_use_unary_oper_on_operand_of_type,
             expr->getOpSpelling(),
             cs.simplifyType(expr->getSubExpr()->getType()))
        .highlight(expr->getSubExpr()->getSourceRange());
  }

  void diagnoseCannotTakeAddressOfExpr(UnaryExpr *expr, Expr *subExpr) {
    TypedDiag<> diag;
    if (isa<AnyLiteralExpr>(subExpr))
      diag = diag::cannot_take_address_of_literal;
    // FIXME: Make this diagnostic a bit more precise.
    else
      diag = diag::cannot_take_address_of_a_temporary;
    diagnose(subExpr->getLoc(), diag).highlight(expr->getOpLoc());
  }

  void diagnoseCannotTakeAddressOfFunc(UnaryExpr *expr, DeclRefExpr *subExpr,
                                       FuncDecl *fn) {
    diagnose(subExpr->getLoc(), diag::cannot_take_address_of_func,
             fn->getIdentifier())
        .highlight(expr->getOpLoc());
  }

  void diagnoseCallWithIncorrectNumberOfArguments(CallExpr *expr,
                                                  FunctionType *calledFn) {
    assert(expr->getNumArgs() != calledFn->getNumArgs());
    diagnose(expr->getFn()->getLoc(),
             diag::fn_called_with_incorrect_number_of_args, calledFn,
             calledFn->getNumArgs(), expr->getNumArgs());
  }

  /// Checks a + or - unary operation
  Expr *checkUnaryPlusOrMinus(UnaryExpr *expr);
  /// Checks a * unary operation
  Expr *checkUnaryDereference(UnaryExpr *expr);
  /// Checks a & unary operation
  Expr *checkUnaryAddressOf(UnaryExpr *expr);
  /// Checks a ! unary operation
  Expr *checkUnaryLNot(UnaryExpr *expr);
  /// Checks a ~ unary operation
  Expr *checkUnaryNot(UnaryExpr *expr);

  /// Checks any assignement
  Expr *checkAssignement(BinaryExpr *expr);
  /// Checks a compound assignement (a binary operator + =, e.g. +=, /=)
  Expr *checkCompoundAssignement(BinaryExpr *expr);
  /// Checks a basic assignement (=)
  Expr *checkBasicAssignement(BinaryExpr *expr);
  /// Checks if \p expr can be assigned to. If it can't, diagnoses it (unless \p
  /// suppressDiagnostics is true)
  /// \returns true if \p expr can be assigned to, false otherwise.
  bool checkExprIsAssignable(Expr *expr, SourceLoc eqLoc,
                             bool suppressDiagnostics = false);
  /// Finishes type-checking of an assignement, giving it a type.
  Expr *finalizeAssignBinaryExpr(BinaryExpr *expr);

  /// Checks a infix binary operation
  Expr *checkBinaryOp(BinaryExpr *expr);
  /// Checks if a binary operation \p op can be applied to types \p lhs and \p
  /// rhs. This will unify the types.
  /// \p rhs is passed by reference so implicit conversions can be inserted
  /// (needed for NullCoalesce).
  /// \returns the type of the operation if it's valid, otherwise returns
  /// nullptr.
  Type checkBinaryOperatorApplication(Expr *lhs, BinaryOperatorKind op,
                                      Expr *&rhs);
  // Checks if ?? can be applied to \p lhs and \p rhs
  /// \p rhs is passed by reference so implicit conversions can be inserted
  /// (needed for NullCoalesce).
  /// \returns the type of the operation if it's valid, otherwise returns
  /// nullptr.
  Type checkNullCoalesceApplication(Expr *lhs, Expr *&rhs);

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

  Expr *visitDestructuredTupleElementExpr(DestructuredTupleElementExpr *expr) {
    llvm_unreachable("Expression checked twice!");
  }

  Expr *visitErrorExpr(ErrorExpr *expr) {
    llvm_unreachable("Expr visited twice!");
  }
};
} // namespace

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

// NOTE:
//    None of the "checkUnary" methods simplify the subexpression's type before
//    checking it. For now it's harmless because there are no situations where a
//    generic type variable can end up as the type of the subexpression of a
//    UnaryExpr, but if inference gets more complex and such situations start to
//    happen, these methods will need to simplify the subexpression's type
//    before checking it.

Expr *ExprChecker::checkUnaryPlusOrMinus(UnaryExpr *expr) {
  // Unary + and - require that the subexpression's type is a numeric type or
  // numeric type variable.
  Type subExprTy = expr->getSubExpr()->getType();
  if (!isNumericTypeOrTypeVariable(subExprTy->getCanonicalType())) {
    diagnoseBadUnaryExpr(expr);
    return nullptr;
  }
  // The type of the expr is the type of its subexpression, minus LValues.
  // Note: We need to keep the type variables intact so things like "let x: i8 =
  // -10" can work (= 10 correctly inferred to i8)
  expr->setType(subExprTy);
  return expr;
}

Expr *ExprChecker::checkUnaryDereference(UnaryExpr *expr) {
  // Dereferencing requires that the subexpression's type is a reference.
  Type subExprTy = expr->getSubExpr()->getType();
  ReferenceType *referenceType =
      subExprTy->getDesugaredType()->getAs<ReferenceType>();
  if (!referenceType) {
    diagnose(expr->getOpLoc(), diag::cannot_deref_value_of_type, subExprTy)
        .highlight(expr->getSubExpr()->getSourceRange());
    diagnose(expr->getLoc(), diag::value_must_have_reference_type);
    return nullptr;
  }
  // The type of the expr is a LValue of the pointee's type.
  Type type = LValueType::get(referenceType->getPointeeType());
  expr->setType(type);
  return expr;
}

Expr *ExprChecker::checkUnaryAddressOf(UnaryExpr *expr) {
  Expr *subExpr = expr->getSubExpr();
  // Check if we can take the address of the value
  if (!subExpr->getType()->isLValueType()) {
    diagnoseCannotTakeAddressOfExpr(expr, subExpr->ignoreParens());
    return nullptr;
  }

  // The type of this expr is a reference type, mutable if the base is.
  // Ignore LValues as well since we require that the operand is an LValue.
  Type subExprType = expr->getSubExpr()->getType()->getRValueType();
  expr->setType(ReferenceType::get(subExprType, isMutableLValue(subExpr)));
  return expr;
}

Expr *ExprChecker::checkUnaryLNot(UnaryExpr *expr) {
  // Unary ! requires that the subexpression's type is a boolean type.
  Type subExprTy = expr->getSubExpr()->getType();
  if (!subExprTy->getCanonicalType()->is<BoolType>()) {
    diagnoseBadUnaryExpr(expr);
    return nullptr;
  }
  // The type of the expr is the type of its subexpression, minus LValues.
  // (It isn't just 'bool' so sugar is preserved).
  expr->setType(subExprTy);
  return expr;
}

Expr *ExprChecker::checkUnaryNot(UnaryExpr *expr) {
  // Unary ~ requires that the subexpression's type is an integer type or
  // integer type variable.
  Type subExprTy = expr->getSubExpr()->getType();
  if (!cs.isIntegerTypeOrTypeVariable(subExprTy->getCanonicalType())) {
    diagnoseBadUnaryExpr(expr);
    return nullptr;
  }
  // The type of the expr is the type of its subexpression, minus LValues.
  // Note: We need to keep the type variables intact so things like "let x: i8 =
  // ~10" can work (= 10 correctly inferred to i8)
  expr->setType(subExprTy);
  return expr;
}

Expr *ExprChecker::checkAssignement(BinaryExpr *expr) {
  assert(expr->isAssignementOp());
  return expr->isCompoundAssignementOp() ? checkCompoundAssignement(expr)
                                         : checkBasicAssignement(expr);
}

Expr *ExprChecker::checkCompoundAssignement(BinaryExpr *expr) {
  assert(expr->isCompoundAssignementOp());
  Expr *lhs = expr->getLHS();
  Expr *rhs = expr->getRHS();

  // 1. Check that the LHS is assignable
  if (!checkExprIsAssignable(lhs, expr->getOpLoc()))
    return nullptr;

  // 2. Check if we can apply that operator to these operands.
  BinaryOperatorKind op = expr->getOpForCompoundAssignementOp();
  Type opType = checkBinaryOperatorApplication(lhs, op, rhs);
  expr->setRHS(rhs);

  if (opType) {
    // Unify the op's type with the LHS. With Sora's type system, this step
    // should always succeed since the type of binary operations is mostly
    // decided by the LHS (note: except for relational operators where the
    // opType is always bool, but there's no "relational compound assignement"
    // obviously since it wouldn't make sense, so it doesn't apply here.)
    //
    // NOTE: This step is ignored for NullCoalesceAssign since it's already
    // performed by checkNullCoalesceApplication. For that operator, just check
    // that the types are right.

    if (expr->getOpKind() != BinaryOperatorKind::NullCoalesceAssign) {
      bool success = cs.unify(expr->getLHS()->getType(), opType);
      assert(success && "This step should never fail for valid binary ops!");
    }
    else {
#ifndef NDEBUG
      Type valueType = lhs->getType()->getMaybeTypeValueType();
      assert(valueType->getRValueType()->getCanonicalType() ==
             opType->getRValueType()->getCanonicalType());
#endif
    }
  }
  else {
    diagnose(expr->getOpLoc(),
             diag::cannot_use_binary_oper_on_operands_of_types,
             expr->getOpSpelling(), cs.simplifyType(lhs->getType()),
             cs.simplifyType(rhs->getType()));
  }

  return finalizeAssignBinaryExpr(expr);
}

Expr *ExprChecker::checkBasicAssignement(BinaryExpr *expr) {
  assert(expr->getOpKind() == BinaryOperatorKind::Assign);
  Expr *lhs = expr->getLHS();
  Expr *rhs = expr->getRHS();

  // 1. Check that the LHS is assignable
  if (!checkExprIsAssignable(lhs, expr->getOpLoc()))
    return nullptr;

  // 2. Insert implicit conversions in the RHS to the LHS's type.
  //    We ignore the LValue here because we know the LHS has to be an lvalue of
  //    some sort.
  rhs = tryCoerceExpr(rhs, lhs->getType()->getRValueType());
  expr->setRHS(rhs);

  // 3. Check if LHS and RHS unify.
  if (!cs.unify(lhs->getType(), rhs->getType())) {
    diagnose(rhs->getLoc(), diag::cannot_assign_value_of_type_to_type,
             cs.simplifyType(rhs->getType()), cs.simplifyType(lhs->getType()))
        .highlight(lhs->getSourceRange())
        .highlight(rhs->getSourceRange())
        .highlight(expr->getOpLoc());
    return nullptr;
  }

  return finalizeAssignBinaryExpr(expr);
}

bool ExprChecker::checkExprIsAssignable(Expr *expr, SourceLoc eqLoc,
                                        bool suppressDiagnostics) {
  expr = expr->ignoreParens();

  // If the expr has an LValue type, it's assignable only if it's mutable.
  Type type = expr->getType();
  if (type->isLValueType() && isMutableLValue(expr))
    return true;
  else if (auto *tuple = dyn_cast<TupleExpr>(expr)) {
    // If it's a tuple, dive into it and check if all of its elements are
    // assignable.
    if (tuple->getNumElements() != 0) {
      bool assignable = true;
      for (Expr *elt : tuple->getElements())
        assignable &=
            checkExprIsAssignable(elt, eqLoc, /*suppressDiagnostics*/ true);
      if (assignable)
        return true;
    }
  }

  // Expr is not assignable - emit a diagnostic if needed.
  if (suppressDiagnostics || type->hasErrorType())
    return false;

  SourceLoc loc = expr->getLoc();
  InFlightDiagnostic diag;
  if (isa<AnyLiteralExpr>(expr))
    diag = diagnose(loc, diag::cannot_assign_to_literal);
  else if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(expr))
    diag = diagnose(loc, diag::cannot_assign_to_immutable_named_value,
                    dre->getValueDecl()->getIdentifier());
  else {
    Type simplified = cs.simplifyType(type);
    if (simplified->hasErrorType())
      return false;

    diag = diagnose(loc, diag::cannot_assign_to_immutable_expr_of_type,
                    cs.simplifyType(type));
  }
  diag.highlight(eqLoc).highlight(expr->getSourceRange());
  return false;
}

Expr *ExprChecker::finalizeAssignBinaryExpr(BinaryExpr *expr) {
  // The assignement's type is its RHS'
  expr->setType(expr->getRHS()->getType());
  return expr;
}

Expr *ExprChecker::checkBinaryOp(BinaryExpr *expr) {
  assert(!expr->isAssignementOp());
  Expr *lhs = expr->getLHS();
  Expr *rhs = expr->getRHS();

  Type lhsTy = lhs->getType();
  Type rhsTy = rhs->getType();
  assert(!lhsTy->hasErrorType() && !rhsTy->hasErrorType());

  Type opType = checkBinaryOperatorApplication(lhs, expr->getOpKind(), rhs);
  assert((opType.isNull() || !opType->hasLValue()) &&
         "Operator return type cannot have an LValue");
  expr->setRHS(rhs);

  if (!opType) {
    diagnose(
        expr->getOpLoc(), diag::cannot_use_binary_oper_on_operands_of_types,
        expr->getOpSpelling(), cs.simplifyType(lhsTy), cs.simplifyType(rhsTy));
    return nullptr;
  }

  expr->setType(opType);
  return expr;
}

// Helper for the function below.
// Checks if operator == or != can be used on operands of type \p type.
static bool isEqualityComparable(Type type) {
  // == and != support integer, floats, boolean, references and tuples thereof.
  // FIXME: Should they support "maybe" types as well?
  type = type->getRValueType()->getCanonicalType();
  switch (type->getKind()) {
  default:
    return false;
  case TypeKind::Integer:
  case TypeKind::Float:
  case TypeKind::Bool:
  case TypeKind::Reference:
    return true;
  case TypeKind::Tuple: {
    TupleType *tt = type->castTo<TupleType>();
    for (Type elt : tt->getElements())
      if (!isEqualityComparable(elt))
        return false;
    return true;
  }
  }
}

Type ExprChecker::checkBinaryOperatorApplication(Expr *lhs,
                                                 BinaryOperatorKind op,
                                                 Expr *&rhs) {
  // Fetch the type of the LHS and RHS and strip LValues, as they're irrelevant
  // here.
  Type lhsType = lhs->getType()->getRValueType();
  Type rhsType = rhs->getType()->getRValueType();
  assert(!lhsType->hasErrorType() && !rhsType->hasErrorType());
  using Op = BinaryOperatorKind;
  // NullCoalesce is a special case
  if (op == Op::NullCoalesce)
    return checkNullCoalesceApplication(lhs, rhs);

  // Unifies the LHS and RHS with a fresh, general type variable.
  // Returns the type variable on success, nullptr if it failed to unify with
  // the lhs/rhs.
  auto inferReturnTypeOfOperator = [&]() -> Type {
    Type tyVar = cs.createGeneralTypeVariable();
    if (cs.unify(lhsType, tyVar) && cs.unify(rhsType, tyVar))
      return tyVar;
    return nullptr;
  };

  switch (op) {
  case Op::Eq:
  case Op::NEq: {
    // == and != requires that the LHS and RHS are exactly equal, but we allow
    // differences in reference mutability (e.g. compare &T with &mut T)
    // FIXME: Wouldn't it be more correct to insert a MutToImmut conversion
    // here? It's not an easy task but would result in a more complete AST.
    UnificationOptions options;
    options.ignoreReferenceMutability = true;
    if (!cs.unify(lhsType, rhsType, options))
      return nullptr;
    // Check if == and != can be used here
    if (!isEqualityComparable(cs.simplifyType(lhsType)))
      return nullptr;
    // Equality operators always return bool
    return ctxt.boolType;
  }
    // Operators that work on any numeric types.
  case Op::Add:
  case Op::Sub:
  case Op::Mul:
  case Op::Div: {
    Type type = inferReturnTypeOfOperator();
    if (type.isNull() || !isNumericTypeOrTypeVariable(type))
      return nullptr;
    return type;
  }
  case Op::LT:
  case Op::LE:
  case Op::GT:
  case Op::GE: {
    Type type = inferReturnTypeOfOperator();
    if (type.isNull() || !isNumericTypeOrTypeVariable(type))
      return nullptr;
    return ctxt.boolType;
  }
    // Operator that only work on integer types
  case Op::Shl:
  case Op::Shr: {
    // Both << and >> have type "(T, U) -> T" where T and U are integer types
    if (!cs.isIntegerTypeOrTypeVariable(lhsType))
      return nullptr;
    if (!cs.isIntegerTypeOrTypeVariable(rhsType))
      return nullptr;
    return lhsType;
  }
  case Op::Rem:
  case Op::And:
  case Op::Or:
  case Op::XOr: {
    Type type = inferReturnTypeOfOperator();
    if (type.isNull() || !cs.isIntegerTypeOrTypeVariable(type))
      return nullptr;
    return type;
  }
    // Logical AND and OR
  case Op::LAnd:
  case Op::LOr:
    // The type of those operators is '(bool, bool) -> bool'.
    // FIXME: Normally, there should be no general type variable as LHS/RHS of
    // this, so checking using ->isBoolType is fine.
    if (lhsType->isBoolType() && rhsType->isBoolType())
      return ctxt.boolType;
    return nullptr;
  default:
    llvm_unreachable("Unknown Operator?");
  }
}

Type ExprChecker::checkNullCoalesceApplication(Expr *lhs, Expr *&rhs) {
  // LHS must be "maybe T" and RHS must be "T".
  Type lhsType = lhs->getType()->getRValueType()->getDesugaredType();
  MaybeType *maybeType = lhsType->getAs<MaybeType>();
  if (!maybeType)
    return nullptr;
  rhs = tryCoerceExpr(rhs, maybeType->getValueType());
  if (!cs.unify(maybeType->getValueType(), rhs->getType()))
    return nullptr;
  return maybeType->getValueType();
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
  Expr *base = expr->getBase();
  Type baseTy = base->getType();
  Identifier memberID = expr->getMemberIdentifier();
  SourceLoc memberIDLoc = expr->getMemberIdentifierLoc();
  bool createLValue = false, isMutableSource = false;

  // #0
  //    - Compute the lookup type
  CanType lookupTy = cs.simplifyType(baseTy)->getCanonicalType();

  // If the type that we want to look into contains an ErrorType, stop here.
  if (lookupTy->hasErrorType())
    return nullptr;

  if (lookupTy->isLValueType()) {
    createLValue = true;
    isMutableSource = isMutableLValue(base);
    lookupTy = CanType(lookupTy->getRValueType());
  }

  // #1
  //  - Check that the operator used was the correct one
  //  - Find the type in which lookup will be performed
  //  - Find if the source is mutable or not
  {
    // If the '->' operator was used, the base must be a reference type
    ReferenceType *ref = lookupTy->getRValueType()->getAs<ReferenceType>();
    if (expr->isArrow()) {
      if (!ref) {
        // It's not a reference type, emit a diagnostic: we want to point at
        // the operator & highlight the base expression fully.
        diagnose(expr->getOpLoc(), diag::base_operand_of_arrow_isnt_ref_ty)
            .highlight(expr->getBase()->getSourceRange());
        return nullptr;
      }
      // It's indeed a reference type, and we'll look into the pointee type,
      // and the mutability depends on the mutability of the ref.
      lookupTy = CanType(ref->getPointeeType());

      // We're always going to generate LValues when accessing a memory
      // location, of course.
      createLValue = true;
      isMutableSource = ref->isMut();
    }
    // Else, if the '.' operator was used, the base must be a value type.
    else if (ref) {
      // It's a reference type, emit a diagnostic: we want to point at the
      // operator & highlight the base expression fully.
      diagnose(expr->getOpLoc(), diag::base_operand_of_dot_isnt_value_ty)
          .highlight(expr->getBase()->getSourceRange());
      return nullptr;
    }
  }
  assert(isMutableSource ? createLValue
                         : true && "If the result is mutable, the result must "
                                   "be an lvalue!");

  // #3
  //  - Perform the actual lookup

  // Helper that emits a diagnostic (lookupTy doesn't have member) and
  // returns nullptr.
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
    return resolveTupleElementExpr(expr, tuple, *lookupResult, isMutableSource,
                                   createLValue);
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

  TypeVariableType *tv = cs.createGeneralTypeVariable();
  expr->setType(LValueType::get(tv));
  return expr;
}

Expr *ExprChecker::visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
  // integer literals have a "float" type variable that'll default to i32
  // unless unified with another floating point type.
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
  // Coerce the subexpression to an RValue.
  expr->setSubExpr(coerceToRValue(expr->getSubExpr()));

  // Resolve the type after the 'as'.
  tc.resolveTypeLoc(expr->getTypeLoc(), getSourceFile());
  Type toType = expr->getTypeLoc().getType();

  // The type of the CastExpr is always the 'to' type, even if the from/toType
  // contains an ErrorType.
  expr->setType(toType);

  // Don't bother checking if the cast is legit if there's an ErrorType
  // somewhere.
  Type subExprTy = expr->getSubExpr()->getType();
  if (subExprTy->hasErrorType() || toType->hasErrorType())
    return expr;

  // Insert potential implicit conversions
  expr->setSubExpr(tryCoerceExpr(expr->getSubExpr(), toType));

  // Then check if the conversion can happen
  Type subExprType = expr->getSubExpr()->getType();
  if (!tc.canExplicitlyCast(cs, subExprType, toType)) {
    // For the diagnostic, use the simplified type of the subexpression w/o
    // implicit conversions
    Type fromType = cs.simplifyType(
        expr->getSubExpr()->ignoreImplicitConversions()->getType());
    diagnose(expr->getSubExpr()->getLoc(), diag::cannot_cast_value_of_type,
             fromType, toType)
        .highlight(expr->getAsLoc())
        .highlight(expr->getTypeLoc().getSourceRange());
  }

  // Finally, unify the types if the subexpression's type contains a type
  // variable.
  // Note that we do not check whether unification was successful because, in
  // some cases, it may not be successful (e.g. int TV to bool), in that case we
  // just want to leave the TV alone so it can be bound to its default type
  // during the epilogue.
  if (subExprType->hasTypeVariable())
    cs.unify(subExprType, toType);

  return expr;
}

Expr *ExprChecker::visitTupleElementExpr(TupleElementExpr *expr) {
  llvm_unreachable("Expr checked twice!");
}

Expr *ExprChecker::visitTupleExpr(TupleExpr *expr) {
  rebuildTupleExprType(ctxt, expr);
  return expr;
}

Expr *ExprChecker::visitParenExpr(ParenExpr *expr) {
  rebuildParenExprType(expr);
  return expr;
}

Expr *ExprChecker::visitCallExpr(CallExpr *expr) {
  // Coerce the callee and all of its arguments to RValues
  expr->setFn(coerceToRValue(expr->getFn()));
  for (Expr *&arg : expr->getArgs())
    arg = coerceToRValue(arg);

  // First, check if the callee is a FunctionType
  Type calleeType = expr->getFn()->getType();

  // Don't bother checking if we have an ErrorType
  if (calleeType->hasErrorType())
    return nullptr;

  auto *fnTy = calleeType->getDesugaredType()->getAs<FunctionType>();
  if (!fnTy) {
    // Use cs.simplifyType so things like 0() are correctly diagnosed as i32 and
    // not diagnosed as '_'
    diagnose(expr->getFn()->getLoc(),
             diag::value_of_non_function_type_isnt_callable,
             cs.simplifyType(calleeType))
        .highlight({expr->getLParenLoc(), expr->getRParenLoc()});
    return nullptr;
  }

  // The type of the CallExpr is the return type of the function, even on error.
  // (= after this point, return expr on error and not nullptr!);
  expr->setType(fnTy->getReturnType());

  // Now, check if we're passing the right amount of parameters to the function
  if (expr->getNumArgs() != fnTy->getNumArgs()) {
    diagnoseCallWithIncorrectNumberOfArguments(expr, fnTy);
    return expr;
  }

  // Finally, check if the call is correct
  for (size_t k = 0; k < expr->getNumArgs(); ++k) {
    Type expectedType = fnTy->getArg(k);
    Expr *arg = expr->getArg(k);

    // Don't bother checking an ill-formed arg.
    if (arg->getType()->hasErrorType())
      continue;

    // Try to apply implicit conversions
    arg = tryCoerceExpr(arg, expectedType);
    expr->setArg(k, arg);

    // Unify
    if (!cs.unify(arg->getType(), expectedType)) {
      // Don't forget to use the type without implicit conversions
      Type argType = arg->ignoreImplicitConversions()->getType();
      diagnose(arg->getLoc(),
               diag::cannot_convert_value_of_ty_to_expected_arg_ty, argType,
               expectedType);
    }
  }

  return expr;
}

Expr *ExprChecker::visitConditionalExpr(ConditionalExpr *expr) {
  // Coerce all operands to an RValue.
  expr->setCond(coerceToRValue(expr->getCond()));
  expr->setThen(coerceToRValue(expr->getThen()));
  expr->setElse(coerceToRValue(expr->getElse()));

  // Check that the condition has a boolean type
  bool isValid = true;
  {
    Type condTy = expr->getCond()->getType();
    if (!condTy->getDesugaredType()->is<BoolType>()) {
      diagnose(expr->getCond()->getLoc(), diag::value_cannot_be_used_as_cond,
               cs.simplifyType(condTy), ctxt.boolType)
          .highlight(expr->getQuestionLoc());
      isValid = false;
    }
  }
  // Create a TypeVariable for the type of the expr
  Type exprTV = cs.createGeneralTypeVariable();
  expr->setType(exprTV);

  Type thenTy = expr->getThen()->getType();
  Type elseTy = expr->getElse()->getType();
  // The type of both operands must unify w/ the TV.

  // Start with the 'then' expr - the one between '?' and ':'
  if (!cs.unify(thenTy, exprTV))
    // Since the TV is general, unification should never fail here since the
    // TV is unbound at this point
    llvm_unreachable("First unification failed?");

  // And now the else - the expr after ':'
  if (!cs.unify(elseTy, exprTV)) {
    // Simplify both types just in case
    diagnose(expr->getColonLoc(),
             diag::result_values_in_ternary_have_different_types,
             cs.simplifyType(thenTy), cs.simplifyType(elseTy))
        .highlight(expr->getColonLoc())
        .highlight(expr->getThen()->getSourceRange())
        .highlight(expr->getElse()->getSourceRange());
    isValid = false;
  }

  return isValid ? expr : nullptr;
}

Expr *ExprChecker::visitForceUnwrapExpr(ForceUnwrapExpr *expr) {
  // Coerce the subexpression to an RValue.
  expr->setSubExpr(coerceToRValue(expr->getSubExpr()));

  // Fetch the type of the subexpression and simplify it.
  Type subExprType = cs.simplifyType(expr->getSubExpr()->getType());

  // Can't check the expression if it has an error type
  if (subExprType->hasErrorType())
    return nullptr;

  subExprType = subExprType->getDesugaredType();

  // Check that it's a maybe type
  MaybeType *maybe = subExprType->getAs<MaybeType>();
  if (!maybe) {
    if (canDiagnose(subExprType))
      diagnose(expr->getExclaimLoc(), diag::cannot_force_unwrap_value_of_type,
               subExprType)
          .highlight(expr->getSubExpr()->getSourceRange());
    return nullptr;
  }
  // If it's indeed a maybe type, the type of this expression is the value
  // type of the 'maybe'.
  expr->setType(maybe->getValueType());
  return expr;
}

Expr *ExprChecker::visitBinaryExpr(BinaryExpr *expr) {
  // If one operand has an error type, don't bother checking the expr.
  if (expr->getLHS()->getType()->hasErrorType() ||
      expr->getRHS()->getType()->hasErrorType())
    return nullptr;

  // Except for assignements, coerce the RHS to an RValue.
  if (!expr->isAssignementOp())
    expr->setLHS(coerceToRValue(expr->getLHS()));
  // Coerce the RHS to an RValue.
  expr->setRHS(coerceToRValue(expr->getRHS()));

  if (expr->isAssignementOp())
    return checkAssignement(expr);
  return checkBinaryOp(expr);
}

Expr *ExprChecker::visitUnaryExpr(UnaryExpr *expr) {
  using UOp = UnaryOperatorKind;
  // Don't bother checking if the subexpression has an error type
  if (expr->getSubExpr()->getType()->hasErrorType())
    return nullptr;

  // Unless this is an AdressOf, we require the subexpression to be an RValue.
  if (expr->getOpKind() != UOp::AddressOf)
    expr->setSubExpr(coerceToRValue(expr->getSubExpr()));

  switch (expr->getOpKind()) {
  case UOp::Plus:
  case UOp::Minus:
    return checkUnaryPlusOrMinus(expr);
  case UOp::AddressOf:
    return checkUnaryAddressOf(expr);
  case UOp::Deref:
    return checkUnaryDereference(expr);
  case UOp::Not:
    return checkUnaryNot(expr);
  case UOp::LNot:
    return checkUnaryLNot(expr);
  }
  llvm_unreachable("Unknown UnaryOperatorKind");
}

//===- ExprCheckerEpilogue ------------------------------------------------===//

namespace {
class ExprCheckerEpilogue : public ASTCheckerBase, public ASTWalker {
public:
  ConstraintSystem &cs;
  bool canComplain = true;
  Expr *parentWithErrorType = nullptr;

  ExprCheckerEpilogue(TypeChecker &tc, ConstraintSystem &cs)
      : ASTCheckerBase(tc), cs(cs) {}

  /// Simplifies the type of \p expr.
  void simplifyTypeOfExpr(Expr *expr) {
    Type type = expr->getType();
    if (!type->hasTypeVariable())
      return;

    // Whether the type is ambiguous
    bool isAmbiguous = false;

    type = cs.simplifyType(type, &isAmbiguous);
    expr->setType(type);

    if (isAmbiguous && canComplain && canDiagnose(expr)) {
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
      Type fromType = cast->getSubExpr()->getType()->getRValueType();
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

//===- ImplicitConversionBuilder ------------------------------------------===//

/// The ImplicitConversionBuilder attempts to insert implicit conversions
/// inside an expression in order to change its type to a desired type.
///
/// Please note that this does not insert LoadExprs. In fact, this completely
/// ignores LValues in the destination type. Why? This is often called to
/// implicitly convert an expression to another expr's type.
namespace {
class ImplicitConversionBuilder
    : private ExprVisitor<ImplicitConversionBuilder, Expr *, Type> {

  using Inherited = ExprVisitor<ImplicitConversionBuilder, Expr *, Type>;
  friend Inherited;

  /// Destructures a tuple value \p expr, creating a DestructuredTupleExpr in
  /// which implicit conversions can be inserted, and visits the
  /// DestructuredTupleExpr's result expression.
  DestructuredTupleExpr *destructureAndVisitTupleValue(Expr *expr,
                                                       Type destType) {
    assert(!isa<TupleExpr>(expr) && !isa<ParenExpr>(expr));
    // Fetch the TupleType
    TupleType *tt = expr->getType()
                        ->getRValueType()
                        ->getDesugaredType()
                        ->getAs<TupleType>();
    assert(tt && "Not a value of type TupleType!");

    // Create the DTE
    DestructuredTupleExpr *dte = new (ctxt) DestructuredTupleExpr(expr);

    // Create the DestructuredTupleElementExprs
    SmallVector<Expr *, 8> destructuredElts;
    destructuredElts.reserve(tt->getNumElements());

    {
      SourceRange exprRange = expr->getSourceRange();
      size_t counter = 0;
      for (Type elt : tt->getElements())
        destructuredElts.push_back(
            new (ctxt) DestructuredTupleElementExpr(exprRange, counter++, elt));
    }

    // Create the result expression : an implicit TupleExpr
    TupleExpr *tuple = TupleExpr::createImplicit(ctxt, destructuredElts);
    rebuildTupleExprType(ctxt, tuple);

    Expr *result = visit(tuple, destType);

    dte->setResultExpr(result);
    dte->setType(result->getType());

    return dte;
  }

public:
  ImplicitConversionBuilder(TypeChecker &tc, ConstraintSystem &cs)
      : tc(tc), cs(cs), ctxt(cs.ctxt) {}

  TypeChecker &tc;
  ConstraintSystem &cs;
  ASTContext &ctxt;

  Expr *doIt(Type destType, Expr *expr) {
    // If both types already unify, we don't have anything to do
    if (cs.unify(destType, expr->getType()))
      return expr;

    if (!tc.canImplicitlyCast(cs, expr->getType(), destType))
      return expr;

    // Visit the Expr using the desugared destination type.
    expr = Inherited::visit(expr, destType->getDesugaredType());
    assert(expr && "visit returned nullptr?");
    return expr;
  }

private:
  Expr *visit(Expr *expr, Type destType) {
    // FIXME: This is an ugly fix, ideally !destType->isLValueType() should be
    // added to the assert below.
    destType = destType->getRValueType();
    assert(!expr->getType()->isLValueType() &&
           "ImplicitConversionBuilder shouldn't have to deal with LValues!");
    Expr *result = Inherited::visit(expr, destType);
    assert(result && "visit() returned null!");
    return result;
  }

  Expr *visitParenExpr(ParenExpr *expr, Type destType) {
    expr->setSubExpr(visit(expr->getSubExpr(), destType));
    rebuildParenExprType(expr);
    return expr;
  }

  Expr *visitDestructuredTupleExpr(DestructuredTupleExpr *expr, Type destType) {
    Expr *result = visit(expr->getResultExpr(), destType);
    expr->setResultExpr(result);
    expr->setType(result->getType());
    return expr;
  }

  Expr *visitTupleExpr(TupleExpr *expr, Type destType) {
    // destType must be a TupleType with the same number of elements as this
    // expr, else, we can't do anything.

    /// Empty tuples are covered by visitExpr.
    if (expr->isEmpty())
      return visitExpr(expr, destType);

    TupleType *destTupleType = destType->getAs<TupleType>();
    if (!destTupleType)
      return expr;

    size_t numElem = destTupleType->getNumElements();

    if (numElem != expr->getNumElements())
      return expr;

    MutableArrayRef<Expr *> tupleElts = expr->getElements();
    for (size_t k = 0; k < numElem; ++k)
      tupleElts[k] = visit(tupleElts[k], destTupleType->getElement(k));

    rebuildTupleExprType(ctxt, expr);
    return expr;
  }

  // For everything else, just try to insert an implicit conversion to type \p
  // destType.
  Expr *visitExpr(Expr *expr, Type destType) {
    // Check if we're trying to convert from a tuple type to another tuple
    // type - if we are, destructure the tuple and visit the result
    // expression.
    if (destType->isTupleType() && expr->getType()->isTupleType()) {
      // If the expr isn't a TupleExpr, destructure it.
      if (!isa<TupleExpr>(expr))
        return destructureAndVisitTupleValue(expr, destType);
    }

    // If we want to convert to a "maybe" type, try to insert a
    // ImplicitMaybeConversionExpr
    if (MaybeType *toMaybe = destType->getAs<MaybeType>()) {
      Type valueType = toMaybe->getValueType();
      // First, recurse on the valuetype, so we can insert other implicit
      // conversions if needed.
      expr = visitExpr(expr, valueType);

      Type exprTy = expr->getType();
      // If the Maybe Type's ValueType unifies w/ the expr's type, or if the
      // expr's type is "null", insert the ImplicitMaybeConversionExpr.
      if (exprTy->isNullType() || cs.unify(valueType, exprTy)) {
        // The Type of the ImplicitMaybeConversionExpr is the destination
        // type. for instance, in "let : maybe Foo = 0" where "Foo" is i32, we
        // want the the implicit conversion to have a "maybe Foo" type.
        expr = new (ctxt) ImplicitMaybeConversionExpr(expr, toMaybe);
      }
      return expr;
    }

    // If we want to convert to a reference type, check if we can insert a
    // MutToImmutReferenceExpr.
    if (ReferenceType *toRef = destType->getAs<ReferenceType>()) {
      Type exprTy = expr->getType();
      // Check if the Expression's type is also a ReferenceType
      auto *exprRefTy = exprTy->getDesugaredType()->getAs<ReferenceType>();
      if (!exprRefTy)
        return expr;

      // Check if we're trying to go from '&mut' to '&'
      //  - The type of the expr must be '&mut'
      //  - The type that we wish to convert to must be '&'
      if (!exprRefTy->isMut() || toRef->isMut())
        return expr;

      // Check if the pointee types are the same
      if (!cs.unify(toRef->getPointeeType(), exprRefTy->getPointeeType()))
        return expr;

      // If everything's okay, insert the MutToImmutReferenceExpr.
      return new (ctxt) MutToImmutReferenceExpr(expr, toRef->withoutMut());
    }

    return expr;
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *TypeChecker::tryCoerceExpr(ConstraintSystem &cs, Expr *expr,
                                 Type toType) {
  return ImplicitConversionBuilder(*this, cs).doIt(toType, expr);
}

Expr *TypeChecker::typecheckBooleanCondition(Expr *expr, DeclContext *dc) {
  assert(expr && "Expr* is null");
  assert(dc && "DeclContext* is null");
  return typecheckExpr(
      expr, dc, ctxt.boolType, [&](Type exprType, Type boolType) {
        assert(exprType);
        assert(boolType.getPtr() == ctxt.boolType.getPtr());
        if (!exprType->hasErrorType())
          diagnose(expr->getLoc(), diag::value_of_non_bool_type_used_as_cond,
                   exprType);
      });
}

/// Main Entry Point for Expression Type-Checking.
Expr *TypeChecker::typecheckExpr(
    ConstraintSystem &cs, Expr *expr, DeclContext *dc, Type ofType,
    llvm::function_ref<void(Type, Type)> onUnificationFailure) {
  // Check the expression
  assert(expr && "Expr* is null");
  assert(dc && "DeclContext* is null");
  expr = expr->walk(ExprChecker(*this, cs, dc)).second;
  assert(expr && "ExprChecker returns a null Expr*?");

  // Insert a LoadExpr if needed.
  expr = coerceToRValue(ctxt, expr);

  // If the expression is expected to be of a certain type, try to make it work.
  if (ofType) {
    if (!expr->getType()->hasErrorType() && !ofType->hasErrorType())
      expr = tryCoerceExpr(cs, expr, ofType);
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

//===- ASTCheckerBase -----------------------------------------------------===//

bool TypeChecker::canDiagnose(Expr *expr) {
  return canDiagnose(expr->getType());
}