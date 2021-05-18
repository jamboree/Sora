//===--- SIRGenExpr.cpp - Expressions SIR Generation ------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Types.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace sora;

//===- Helpers -----------------------------------------------------------===//

/// \returns true if \p expr is an implicit conversion that needs codegen.
bool needsCodeGen(ImplicitConversionExpr *expr) {
  // Mut to Immut doesn't need codegen as there is no distinction between & and
  // &mut in the IR.
  return !isa<MutToImmutReferenceExpr>(expr);
}

/// \returns \p expr without parens and/or implicit conversions that don't need
/// codegen.
Expr *getCodeGenExpr(Expr *expr) {
  // NOTE: The ASTVerifier enforces that implicit conversions never have
  // ParenExpr as subexpression.

  // First ignore parens.
  expr = expr->ignoreParens();

  // Then look at the implicit conversions inside.
  while (auto *implConv = dyn_cast<ImplicitConversionExpr>(expr)) {
    if (needsCodeGen(implConv))
      break;
    expr = implConv->getSubExpr();
  }

  return expr;
}

/// \returns true if \p expr has at least one meaningful implicit conversions.
/// DTEs that don't have any can just be generated like tuples.
bool hasMeaningfulImplicitConversions(DestructuredTupleExpr *expr) {
  TupleExpr *tuple = cast<TupleExpr>(expr->getResultExpr());
  for (Expr *elt : tuple->getElements())
    if (!isa<DestructuredTupleExpr>(getCodeGenExpr(elt)))
      return true;
  return false;
}

/// This looks at \p expr and return each element of the expression.
///   - For TupleExprs, stores each tuple element in \p elts.
///   - For DestructureTupleExpr, stores each tuple element in \p elts iff no
///   implicit conversion needs codegen (see \c needsCodeGen)
///   - For everything else, just stores \p expr in \p elts.
///
/// Note that this doesn't use \c getCodeGenExpr on \p expr.
void getExprElements(Expr *expr, SmallVectorImpl<Expr *> &elts) {
  auto handleTuple = [&](TupleExpr *tuple) {
    ArrayRef<Expr *> exprs = tuple->getElements();
    elts.reserve(exprs.size());
    elts.append(exprs.begin(), exprs.end());
  };

  // Tuples: Append elements to elts.
  if (auto *tuple = dyn_cast<TupleExpr>(expr))
    return handleTuple(tuple);

  // DestructureTupleExpr: Act like tuples iff there is no meaningful implicit
  // conversion.
  if (auto *dte = dyn_cast<DestructuredTupleExpr>(expr))
    if (!hasMeaningfulImplicitConversions(dte))
      return handleTuple(cast<TupleExpr>(dte->getSubExpr()));

  // Else, just store \p expr in \p elts.
  elts.push_back(expr);
}

//===- ExprGenerator -----------------------------------------------------===//

namespace {
/// Class responsible for generating IR for expression with an RValue type.
///
/// This is where most of the logic is. This class owns the LValueGenerator as
/// well, which is used to generate LValues inside expressions.
class ExprGenerator : public SIRGeneratorBase,
                      public ExprVisitor<ExprGenerator, mlir::Value> {
public:
  using Base = ExprVisitor<ExprGenerator, mlir::Value>;

  ExprGenerator(SIRGen &sirGen, mlir::OpBuilder &builder)
      : SIRGeneratorBase(sirGen), builder(builder) {}

  mlir::OpBuilder &builder;

  mlir::Value visitUnresolvedExpr(UnresolvedExpr *) {
    llvm_unreachable("UnresolvedExpr past Sema!");
  }

  mlir::Value visitErrorExpr(ErrorExpr *) {
    llvm_unreachable("ErrorExpr past Sema!");
  }

  mlir::Value visitDiscardExpr(DiscardExpr *expr) {
    llvm_unreachable("DiscardExpr shouldn't be visited directly!");
  }

  mlir::Value visit(Expr *expr) {
    mlir::Value result = Base::visit(expr);
    assert(result.getType() == getType(expr) && "Unexpected Operation Type!");
    return result;
  }

  /// Generates an address-of (&) operation on \p value.
  mlir::Value genAddressOf(mlir::Location opLoc, mlir::Value value);
  /// Generates a dereference (*) operation on \p value.
  mlir::Value genDeref(mlir::Location opLoc, mlir::Value value);
  /// Generates a NOT operation on \p value (invert all of its bits).
  /// \p value must be a signless int.
  /// This is for both bitwise and logical nots, which both produce a
  /// sir.bitnot.
  mlir::Value genNot(mlir::Location opLoc, mlir::Value value);
  /// Generates an unary minus (-) operation on \p value.
  mlir::Value genUnaryMinus(mlir::Location opLoc, mlir::Value value);

  /// Assigns a value to \p dest, using \p assignLoc as the loc of the store/set
  /// operation, and returns the Value of the source.
  /// The source is lazily fetched from \p getSrc before generating the store.
  mlir::Value genBasicAssign(Expr *dest, mlir::Location assignLoc,
                             llvm::function_ref<mlir::Value()> getSrc);

  // Visit Methods
  mlir::Value visitDeclRefExpr(DeclRefExpr *expr);
  mlir::Value visitIntegerLiteralExpr(IntegerLiteralExpr *expr);
  mlir::Value visitFloatLiteralExpr(FloatLiteralExpr *expr);
  mlir::Value visitBooleanLiteralExpr(BooleanLiteralExpr *expr);
  mlir::Value visitNullLiteralExpr(NullLiteralExpr *expr);
  mlir::Value
  visitImplicitMaybeConversionExpr(ImplicitMaybeConversionExpr *expr);
  mlir::Value visitMutToImmutReferenceExpr(MutToImmutReferenceExpr *expr);
  mlir::Value visitDestructuredTupleExpr(DestructuredTupleExpr *expr);
  mlir::Value
  visitDestructuredTupleElementExpr(DestructuredTupleElementExpr *expr);
  mlir::Value visitLoadExpr(LoadExpr *expr);
  mlir::Value visitCastExpr(CastExpr *expr);
  mlir::Value visitTupleElementExpr(TupleElementExpr *expr);
  mlir::Value visitTupleExpr(TupleExpr *expr);
  mlir::Value visitParenExpr(ParenExpr *expr);
  mlir::Value visitCallExpr(CallExpr *expr);
  mlir::Value visitConditionalExpr(ConditionalExpr *expr);
  mlir::Value visitForceUnwrapExpr(ForceUnwrapExpr *expr);
  mlir::Value visitBinaryExpr(BinaryExpr *expr);
  mlir::Value visitUnaryExpr(UnaryExpr *expr);
};
} // namespace

mlir::Value ExprGenerator::visitDeclRefExpr(DeclRefExpr *expr) {
  // TODO: Move this logic in SIRGenDecl once I know how to generate FuncDecls.
  // That way I can get rid of the Decl.hpp include.
  ValueDecl *vd = expr->getValueDecl();

  /// This is an LValue (basically a pointer), so just return the value that
  /// contains the address of the function.
  if (VarDecl *var = dyn_cast<VarDecl>(vd))
    return sirGen.getVarDeclAddress(var);

  if (ParamDecl *param = dyn_cast<ParamDecl>(vd))
    llvm_unreachable("visitDeclRefExpr - ParamDecl handling");

  // If we get to this point, we want to generate an indirect call, so emit a
  // constant.
  if (FuncDecl *func = dyn_cast<FuncDecl>(vd)) {
    mlir::FuncOp funcOp = sirGen.getFuncOp(func);
    return builder.create<mlir::ConstantOp>(
        getNodeLoc(expr), funcOp.getType(),
        mlir::SymbolRefAttr::get(&mlirCtxt, funcOp.getName()));
  }

  llvm_unreachable("Unknown ValueDecl kind!");
}

mlir::Value ExprGenerator::visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
  assert(expr->getType()->isAnyIntegerType() && "Not an Integer Type?!");
  mlir::Type type = getType(expr->getType());
  assert(type.isa<mlir::IntegerType>() && "Not an IntegerType?!");

  // TODO: Create a "BuildIntConstant" method as I'll probably need to create
  // int constants a lot.
  auto valueAttr = mlir::IntegerAttr::get(type, expr->getValue());
  return builder.create<mlir::ConstantOp>(getNodeLoc(expr), valueAttr);
}

mlir::Value ExprGenerator::visitFloatLiteralExpr(FloatLiteralExpr *expr) {
  assert(expr->getType()->isAnyFloatType() && "Not a Float Type?!");
  mlir::Type type = getType(expr->getType());
  assert(type.isa<mlir::FloatType>() && "Not a FloatType?!");

  auto valueAttr = mlir::FloatAttr::get(type, expr->getValue());
  return builder.create<mlir::ConstantOp>(getNodeLoc(expr), valueAttr);
}

mlir::Value ExprGenerator::visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
  assert(expr->getType()->isBoolType() && "Not a Bool Type?!");
  mlir::Type type = getType(expr->getType());
  assert(type.isInteger(1) && "Expected a i1!");
  APInt value(1, expr->getValue() ? 1 : 0);

  auto valueAttr = mlir::IntegerAttr::get(type, value);
  return builder.create<mlir::ConstantOp>(getNodeLoc(expr), valueAttr);
}

mlir::Value ExprGenerator::visitNullLiteralExpr(NullLiteralExpr *expr) {
  // "null" is a contextual thing. It's only emitted through its parent,
  // usually a MaybeConversionExpr.
  llvm_unreachable(
      "NullLiteralExprs should only be emitted through their parents!");
}

mlir::Value ExprGenerator::visitImplicitMaybeConversionExpr(
    ImplicitMaybeConversionExpr *expr) {
  // TODO: Handle "NullLiteralExpr" as child expr as a special case.
  llvm_unreachable("Unimplemented - visitImplicitMaybeConversionExpr");
}

mlir::Value
ExprGenerator::visitMutToImmutReferenceExpr(MutToImmutReferenceExpr *expr) {
  // As there is no distinction between mutable and immutable references in the
  // IR, we don't have to do anything.
  return visit(expr->getSubExpr());
}

mlir::Value
ExprGenerator::visitDestructuredTupleExpr(DestructuredTupleExpr *expr) {
  llvm_unreachable("Unimplemented - visitDestructuredTupleExpr");
}

mlir::Value ExprGenerator::visitDestructuredTupleElementExpr(
    DestructuredTupleElementExpr *expr) {
  llvm_unreachable("Unimplemented - visitDestructuredTupleElementExpr");
}

mlir::Value ExprGenerator::visitLoadExpr(LoadExpr *expr) {
  return builder.create<sir::LoadOp>(getNodeLoc(expr),
                                     visit(expr->getSubExpr()));
}

mlir::Value ExprGenerator::visitCastExpr(CastExpr *expr) {
  Expr *subExpr = expr->getSubExpr();

  // Generate IR for the subexpression, and if the cast is "useless", stop here.
  mlir::Value subExprValue = visit(subExpr);
  if (expr->isUseless())
    return subExprValue;

  Type type = expr->getType();

  // Convert the result type and the loc to their MLIR equivalent.
  mlir::Type mlirType = getType(type);
  assert(mlirType != subExprValue.getType() && "Cast is useless!");

  mlir::Location loc = getNodeLoc(expr);

  // Currently, all sora casts are static casts, so just emit a static_cast op.
  // We do not need to handle things like creating maybe types - implicit casts
  // handle those.
  return builder.create<sir::StaticCastOp>(loc, subExprValue, mlirType);
}

mlir::Value ExprGenerator::visitTupleElementExpr(TupleElementExpr *expr) {
  // This will need to be handled differently depending on whether we're working
  // on an LValue or not.
  // LValue tuple element should be a gep-like operation, others should be
  // extract_element-like.
  // TODO: Once this is working, also add assignement tests in
  // /test/SIRGen/expr/binary/simple-assigns.sora.
  llvm_unreachable("Unimplemented - visitTupleElementExpr");
}

mlir::Value ExprGenerator::visitTupleExpr(TupleExpr *expr) {
  mlir::Location loc = getNodeLoc(expr);

  // Empty tuples are just void constants.
  if (expr->isEmpty())
    return builder.create<sir::VoidConstantOp>(loc);

  SmallVector<mlir::Value, 8> tupleElts;
  tupleElts.reserve(expr->getNumElements());

  for (Expr *tupleElt : expr->getElements())
    tupleElts.push_back(visit(tupleElt));

  return builder.create<sir::CreateTupleOp>(loc, tupleElts);
}

mlir::Value ExprGenerator::visitParenExpr(ParenExpr *expr) {
  return visit(expr->getSubExpr());
}

mlir::Value ExprGenerator::visitCallExpr(CallExpr *expr) {
  llvm_unreachable("Unimplemented - visitCallExpr");
}

mlir::Value ExprGenerator::visitConditionalExpr(ConditionalExpr *expr) {
  llvm_unreachable("Unimplemented - visitConditionalExpr");
}

mlir::Value ExprGenerator::visitForceUnwrapExpr(ForceUnwrapExpr *expr) {
  llvm_unreachable("Unimplemented - visitForceUnwrapExpr");
}

mlir::Value
ExprGenerator::genBasicAssign(Expr *destExpr, mlir::Location assignLoc,
                              llvm::function_ref<mlir::Value()> getSrc) {
  destExpr = getCodeGenExpr(destExpr);

  // If the LHS is a DiscardExpr, just gen the RHS and return.
  if (isa<DiscardExpr>(destExpr))
    return getSrc();

  // If the LHS is any expression of LValue type, emit a simple store.
  if (destExpr->getType()->is<LValueType>()) {
    // Sanity check: TupleExprs never have an LValue type.
    assert(!isa<TupleExpr>(destExpr) &&
           "TupleExpr can never have an LValue type!");

    mlir::Value dest = visit(destExpr);
    mlir::Value src = getSrc();

    builder.create<sir::StoreOp>(assignLoc, src, dest);

    return src;
  }

  // We can also have TupleExprs on the LHS, in this case, decompose.
  //
  // NOTE: We don't special-case things like (a, b) = (0, 1) here. Instead, we
  // let constant folding fold the create_tuple + destructure_tuple.
  if (auto *destTupleExpr = dyn_cast<TupleExpr>(destExpr)) {
    ArrayRef<Expr *> destTupleElts = destTupleExpr->getElements();

    mlir::Value src = getSrc();

    // FIXME: Is the Location right for this op?
    auto destructureTupleOp =
        builder.create<sir::DestructureTupleOp>(assignLoc, src);

    mlir::ResultRange destructuredTupleValues = destructureTupleOp.getResults();

    assert(destructuredTupleValues.size() == destTupleElts.size() &&
           "Number of elements don't match on the lhs/rhs of the assignement!");

    for (size_t k = 0; k < destTupleElts.size(); ++k)
      genBasicAssign(destTupleElts[k], assignLoc,
                     [&] { return destructuredTupleValues[k]; });

    return src;
  }

  llvm_unreachable("Unhandled assignement kind!");
}

mlir::Value ExprGenerator::visitBinaryExpr(BinaryExpr *expr) {
  Expr *lhs = expr->getLHS();
  Expr *rhs = expr->getRHS();
  mlir::Location loc = getNodeLoc(expr);

  if (expr->isAssignementOp()) {
    if (expr->isCompoundAssignementOp())
      llvm_unreachable(
          "Unimplemented - visitBinaryExpr - isCompoundAssignementOp");
    return genBasicAssign(lhs, loc, [&] { return visit(getCodeGenExpr(rhs)); });
  }
  llvm_unreachable("Unimplemented - visitBinaryExpr");
}

mlir::Value ExprGenerator::genAddressOf(mlir::Location opLoc,
                                        mlir::Value value) {
  // AddressOf just converts a sir.pointer into a sir.reference (lvalue to &).
  assert(value.getType().isa<sir::PointerType>() &&
         "This operation should only be possible on pointers (lvalues)!");
  auto ptrType = value.getType().cast<sir::PointerType>();
  auto refType = sir::ReferenceType::get(ptrType.getPointeeType());
  return builder.create<sir::StaticCastOp>(opLoc, value, refType);
}

mlir::Value ExprGenerator::genDeref(mlir::Location opLoc, mlir::Value value) {
  // AddressOf just converts a sir.reference into a sir.pointer (& to lvalue).
  assert(value.getType().isa<sir::ReferenceType>() &&
         "This operation should only be possible on reference types!");
  auto refType = value.getType().cast<sir::ReferenceType>();
  auto ptrType = sir::PointerType::get(refType.getPointeeType());
  return builder.create<sir::StaticCastOp>(opLoc, value, ptrType);
}

mlir::Value ExprGenerator::genNot(mlir::Location opLoc, mlir::Value value) {
  assert(value.getType().isSignlessInteger() &&
         "Operand must be a signless int!");
  return builder.create<sir::BitNotOp>(opLoc, value);
}

mlir::Value ExprGenerator::genUnaryMinus(mlir::Location opLoc,
                                         mlir::Value value) {
  // Generate a 0 - value in both cases.
  if (auto intTy = value.getType().dyn_cast<mlir::IntegerType>()) {
    mlir::Value zero = builder.create<mlir::ConstantIntOp>(opLoc, 0, intTy);
    return builder.create<mlir::SubIOp>(opLoc, zero, value);
  }
  else if (auto fltTy = value.getType().dyn_cast<mlir::FloatType>()) {
    APFloat zero(fltTy.getFloatSemantics(), 0);
    mlir::Value lhs = builder.create<mlir::ConstantFloatOp>(opLoc, zero, fltTy);
    return builder.create<mlir::SubFOp>(opLoc, lhs, value);
  }
  else
    llvm_unreachable("Unsupported type for Unary Minus!");
}

mlir::Value ExprGenerator::visitUnaryExpr(UnaryExpr *expr) {
  mlir::Value value = visit(expr->getSubExpr());
  mlir::Location loc = getLoc(expr->getOpLoc());

  switch (expr->getOpKind()) {
  case UnaryOperatorKind::AddressOf:
    return genAddressOf(loc, value);
  case UnaryOperatorKind::Deref:
    return genDeref(loc, value);
  case UnaryOperatorKind::Not:
  case UnaryOperatorKind::LNot:
    return genNot(loc, value);
  case UnaryOperatorKind::Minus:
    return genUnaryMinus(loc, value);
  case UnaryOperatorKind::Plus:
    // It's a no-op.
    return value;
  default:
    llvm_unreachable("Unknown Unary Operator Kind");
  }
}

//===- SIRGen -------------------------------------------------------------===//

mlir::Value SIRGen::genExpr(mlir::OpBuilder &builder, Expr *expr) {
  // This entry point should only be used to generate:
  //  -> The initializer of LetDecls
  //  -> Expressions used as statements
  // All of those should have RValue types.
  assert(!expr->getType()->is<LValueType>() && "Expected an RValue!");
  return ExprGenerator(*this, builder).visit(expr);
}

mlir::Type SIRGen::getType(Expr *expr) { return getType(expr->getType()); }

mlir::Location SIRGen::getNodeLoc(Expr *expr) {
  return mlir::OpaqueLoc::get(expr, getLoc(expr->getLoc()));
}