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

//===- Utils --------------------------------------------------------------===//

/// (re-)builds the type of a TupleExpr
void rebuildTupleExprType(ASTContext &ctxt, TupleExpr *expr) {
  // If the tuple is empty, just give it a () type.
  if (expr->isEmpty()) {
    expr->setType(TupleType::getEmpty(ctxt));
    return;
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
}

/// (re-)builds the type of a ParenExpr
void rebuildParenExprType(ParenExpr *expr) {
  expr->setType(expr->getSubExpr()->getType());
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

  /// Called when the type-checker is sure that \p udre references \p resolved.
  /// This will check that \p resolved can be referenced from inside this
  /// DeclContext (if it can't, returns nullptr) and, on success, build a
  /// DeclRefExpr.
  /// \returns nullptr on failure, a new DeclRefExpr on success.
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
          // Add additional note? Ask on discord!
          return nullptr;
        }
      }
    }
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

  Expr *tryInsertImplicitConversions(Expr *expr, Type toType) {
    return tc.tryInsertImplicitConversions(cs, expr, toType);
  }

  bool isNumericTypeOrNumericTypeVariable(Type type) {
    return isIntegerTypeOrIntegerTypeVariable(type) ||
           isFloatTypeOrFloatTypeVariable(type);
  }

  bool isIntegerTypeOrIntegerTypeVariable(Type type) {
    assert(type);
    if (type->is<IntegerType>())
      return true;
    if (TypeVariableType *tv = type->getAs<TypeVariableType>()) {
      if (cs.isIntegerTypeVariable(tv))
        return true;
      if (cs.hasSubstitution(tv))
        return isIntegerTypeOrIntegerTypeVariable(cs.simplifyType(tv));
    }
    return false;
  }

  bool isFloatTypeOrFloatTypeVariable(Type type) {
    assert(type);
    if (type->is<FloatType>())
      return true;
    if (TypeVariableType *tv = type->getAs<TypeVariableType>()) {
      if (cs.isFloatTypeVariable(tv))
        return true;
      if (cs.hasSubstitution(tv))
        return isFloatTypeOrFloatTypeVariable(cs.simplifyType(type));
    }
    return false;
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
    else if (isa<TupleExpr>(subExpr))
      diag = diag::cannot_take_address_of_temp_tuple;
    // FIXME: Pretty much all other expressions are guaranteed to be
    // temporaries, right?
    else
      diag = diag::cannot_take_address_of_a_temp_value;
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

  /// \returns true if we can take the address of \p subExpr.
  /// If we can't, this function will emit the appropriate diagnostics.
  bool checkCanTakeAddressOf(UnaryExpr *expr, Expr *subExpr);

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

  /// Checks an assignement
  Expr *checkAssignement(BinaryExpr *expr);
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

bool ExprChecker::checkCanTakeAddressOf(UnaryExpr *expr, Expr *subExpr) {
  subExpr = subExpr->ignoreParens();

  // We can only take the address of:
  //    - Variables/Parameters (DeclRefExprs)
  //    - TupleElementExprs if we can also take the address of their base
  if (DeclRefExpr *dre = dyn_cast<DeclRefExpr>(subExpr)) {
    ValueDecl *valueDecl = dre->getValueDecl();
    if (FuncDecl *fn = dyn_cast<FuncDecl>(valueDecl)) {
      diagnoseCannotTakeAddressOfFunc(expr, dre, fn);
      return false;
    }
    assert((isa<ParamDecl>(valueDecl) || isa<VarDecl>(valueDecl)) &&
           "Unknown ValueDecl kind");
    return true;
  }
  if (TupleElementExpr *tee = dyn_cast<TupleElementExpr>(subExpr))
    return checkCanTakeAddressOf(expr, tee->getBase()->ignoreParens());
  diagnoseCannotTakeAddressOfExpr(expr, subExpr);
  return false;
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
  if (!isNumericTypeOrNumericTypeVariable(
          subExprTy->getRValue()->getCanonicalType())) {
    diagnoseBadUnaryExpr(expr);
    return nullptr;
  }
  // The type of the expr is the type of its subexpression, minus LValues.
  // Note: We need to keep the type variables intact so things like "let x: i8 =
  // -10" can work (= 10 correctly inferred to i8)
  expr->setType(subExprTy->getRValue());
  return expr;
}

Expr *ExprChecker::checkUnaryDereference(UnaryExpr *expr) {
  // Dereferencing requires that the subexpression's type is a reference.
  Type subExprTy = expr->getSubExpr()->getType();
  ReferenceType *referenceType =
      subExprTy->getRValue()->getDesugaredType()->getAs<ReferenceType>();
  if (!referenceType) {
    diagnose(expr->getOpLoc(), diag::cannot_deref_value_of_type, subExprTy)
        .highlight(expr->getSubExpr()->getSourceRange());
    diagnose(expr->getLoc(), diag::value_must_have_reference_type);
    return nullptr;
  }
  // The type of the expr is the pointee type of the reference type, plus an
  // lvalue for mutable references.
  Type type = referenceType->getPointeeType();
  if (referenceType->isMut())
    type = LValueType::get(type);
  expr->setType(type);
  return expr;
}

Expr *ExprChecker::checkUnaryAddressOf(UnaryExpr *expr) {
  // Check if we can take the address of the value
  if (!checkCanTakeAddressOf(expr, expr->getSubExpr()))
    return nullptr;

  // The type of this expr is a reference type, with 'mut' if the subExpr is an
  // LValue.
  Type subExprType = expr->getSubExpr()->getType();
  bool isMut = subExprType->isLValue();
  expr->setType(ReferenceType::get(subExprType->getRValue(), isMut));
  return expr;
}

Expr *ExprChecker::checkUnaryLNot(UnaryExpr *expr) {
  // Unary ! requires that the subexpression's type is a boolean type.
  Type subExprTy = expr->getSubExpr()->getType();
  if (!subExprTy->getRValue()->getCanonicalType()->is<BoolType>()) {
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
  if (!isIntegerTypeOrIntegerTypeVariable(
          subExprTy->getRValue()->getCanonicalType())) {
    diagnoseBadUnaryExpr(expr);
    return nullptr;
  }
  // The type of the expr is the type of its subexpression, minus LValues.
  // Note: We need to keep the type variables intact so things like "let x: i8 =
  // ~10" can work (= 10 correctly inferred to i8)
  expr->setType(subExprTy->getRValue());
  return expr;
}

Expr *ExprChecker::checkAssignement(BinaryExpr *expr) {
  assert(expr->isAssignementOp());
  return nullptr;
}

Expr *ExprChecker::checkBinaryOp(BinaryExpr *expr) {
  assert(!expr->isAssignementOp());
  Expr *lhs = expr->getLHS();
  Expr *rhs = expr->getRHS();

  Type lhsTy = lhs->getType();
  Type rhsTy = rhs->getType();
  assert(!lhsTy->hasErrorType() && !rhsTy->hasErrorType());

  Type opType = checkBinaryOperatorApplication(
      lhs, expr->getOpKind(), rhs);
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
  type = type->getRValue()->getCanonicalType();
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
  Type lhsType = lhs->getType()->getRValue();
  Type rhsType = rhs->getType()->getRValue();
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
    if (type.isNull() || !isNumericTypeOrNumericTypeVariable(type))
      return nullptr;
    return type;
  }
  case Op::LT:
  case Op::LE:
  case Op::GT:
  case Op::GE: {
    Type type = inferReturnTypeOfOperator();
    if (type.isNull() || !isNumericTypeOrNumericTypeVariable(type))
      return nullptr;
    return ctxt.boolType;
  }
    // Operator that only work on integer types
  case Op::Shl:
  case Op::Shr: {
    // Both << and >> have type "(T, U) -> T" where T and U are integer types
    if (!isIntegerTypeOrIntegerTypeVariable(lhsType))
      return nullptr;
    if (!isIntegerTypeOrIntegerTypeVariable(rhsType))
      return nullptr;
    return lhsType;
  }
  case Op::Rem:
  case Op::And:
  case Op::Or:
  case Op::XOr: {
    Type type = inferReturnTypeOfOperator();
    if (type.isNull() || !isIntegerTypeOrIntegerTypeVariable(type))
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
  Type lhsType = lhs->getType()->getRValue()->getDesugaredType();
  MaybeType *maybeType = lhsType->getAs<MaybeType>();
  if (!maybeType)
    return nullptr;
  rhs = tryInsertImplicitConversions(rhs, maybeType->getValueType());
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
        // It's not a reference type, emit a diagnostic: we want to point at
        // the operator & highlight the base expression fully.
        diagnose(expr->getOpLoc(), diag::base_operand_of_arrow_isnt_ref_ty)
            .highlight(expr->getBase()->getSourceRange());
        return nullptr;
      }
      // It's indeed a reference type, and we'll look into the pointee type,
      // and the mutability depends on the mutability of the ref.
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
  // Resolve the type after the 'as'.
  tc.resolveTypeLoc(expr->getTypeLoc(), getSourceFile());
  Type toType = expr->getTypeLoc().getType();

  // The type of the CastExpr is always the 'to' type, even if the from/toType
  // contains an ErrorType.
  expr->setType(toType);

  // Don't bother checking if the cast is legit if there's an ErrorType
  // somewhere.
  if (expr->getSubExpr()->getType()->hasErrorType() || toType->hasErrorType())
    return expr;

  // Insert potential implicit conversions
  expr->setSubExpr(tryInsertImplicitConversions(expr->getSubExpr(), toType));

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
  // variable
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
  // First, check if the callee is a FunctionType
  Type calleeType = expr->getFn()->getType()->getRValue();

  // Don't bother checking if we have an ErrorType
  if (calleeType->hasErrorType())
    return nullptr;

  FunctionType *calledFn =
      calleeType->getDesugaredType()->getAs<FunctionType>();
  if (!calledFn) {
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
  expr->setType(calledFn->getReturnType());

  // Now, check if we're passing the right amount of parameters to the function
  if (expr->getNumArgs() != calledFn->getNumArgs()) {
    diagnoseCallWithIncorrectNumberOfArguments(expr, calledFn);
    return expr;
  }

  // Finally, check if the call is correct
  for (size_t k = 0; k < expr->getNumArgs(); ++k) {
    Type expectedType = calledFn->getArg(k);
    Expr *arg = expr->getArg(k);

    // Try to apply implicit conversions
    arg = tryInsertImplicitConversions(arg, expectedType);
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
  // Check that the condition has a boolean type
  bool isValid = true;
  {
    Type condTy = expr->getCond()->getType();
    condTy = condTy->getRValue();
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
  // Fetch the type of the subexpression and simplify it.
  Type subExprType = cs.simplifyType(expr->getSubExpr()->getType());

  // Can't check the expression if it has an error type
  if (subExprType->hasErrorType())
    return nullptr;

  subExprType = subExprType->getRValue()->getDesugaredType();

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

  if (expr->isAssignementOp())
    return checkAssignement(expr);
  return checkBinaryOp(expr);
}

Expr *ExprChecker::visitUnaryExpr(UnaryExpr *expr) {
  using UOp = UnaryOperatorKind;
  // Don't bother checking if the subexpression has an error type
  if (expr->getSubExpr()->getType()->hasErrorType())
    return nullptr;

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

//===- ImplicitConversionBuilder ------------------------------------------===//

/// The ImplicitConversionBuilder attempts to insert implicit conversions
/// inside an expression in order to change its type to a desired type.
class ImplicitConversionBuilder
    : private ExprVisitor<ImplicitConversionBuilder, Expr *, Type> {

  using Inherited = ExprVisitor<ImplicitConversionBuilder, Expr *, Type>;
  friend Inherited;

  /// Destructures a tuple value \p expr, creating a DestructuredTupleExpr in
  /// which implicit conversions can be inserted.
  DestructuredTupleExpr *destructureTupleValue(Expr *expr) {
    assert(!isa<TupleExpr>(expr) && !isa<ParenExpr>(expr) &&
           !isa<ImplicitConversionExpr>(expr));
    // Fetch the TupleType
    TupleType *tt =
        expr->getType()->getRValue()->getDesugaredType()->getAs<TupleType>();
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
    TupleExpr *result = TupleExpr::createImplicit(ctxt, destructuredElts);
    rebuildTupleExprType(ctxt, result);

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

    // Check if we're trying to convert from a tuple type to another tuple
    // type
    if (destType->isTupleType() && expr->getType()->isTupleType()) {
      bool needsDestructuring = false;
      // If the expr isn't a TupleExpr, we may need to destructure the tuple in
      // order to apply implicit conversions.
      if (!isa<TupleExpr>(expr->ignoreParens())) {
        // Only destructure if the expr isn't a ParenExpr - we want to
        // destructure the expr inside ParenExprs, not the ParenExpr itself.
        needsDestructuring =
            !isa<ParenExpr>(expr) && !isa<ImplicitConversionExpr>(expr);
      }
      // Destructure the tuple if needed
      if (needsDestructuring)
        expr = destructureTupleValue(expr);
    }

    // Visit the Expr using the desugared destination type.
    expr = Inherited::visit(expr, destType->getRValue()->getDesugaredType());
    assert(expr && "visit returned nullptr?");
    return expr;
  }

private:
#ifndef NDEBUG
  Expr *visit(Expr *expr, Type destType) {
    Expr *result = Inherited::visit(expr, destType);
    assert(result && "visit() returned null!");
    return result;
  }
#endif

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
      ReferenceType *exprRefTy =
          exprTy->getRValue()->getDesugaredType()->getAs<ReferenceType>();
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

Expr *TypeChecker::tryInsertImplicitConversions(ConstraintSystem &cs,
                                                Expr *expr, Type toType) {
  return ImplicitConversionBuilder(*this, cs).doIt(toType, expr);
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
    if (!expr->getType()->hasErrorType() && !ofType->hasErrorType())
      expr = tryInsertImplicitConversions(cs, expr, ofType);
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

bool TypeChecker::canDiagnose(Expr *expr) {
  return canDiagnose(expr->getType());
}