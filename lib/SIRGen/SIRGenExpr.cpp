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

  // UnaryExpr Helpers
  mlir::Value genUnaryAddressOf(UnaryExpr *expr);
  mlir::Value genUnaryNot(UnaryExpr *expr);
  mlir::Value genUnaryLNot(UnaryExpr *expr);
  mlir::Value genUnaryMinus(UnaryExpr *expr);
  mlir::Value genUnaryPlus(UnaryExpr *expr);

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

  if (VarDecl *var = dyn_cast<VarDecl>(vd))
    return sirGen.getVarDeclAddress(var);

  if (ParamDecl *param = dyn_cast<ParamDecl>(vd))
    llvm_unreachable("visitDeclRefExpr - ParamDecl handling");

  if (FuncDecl *func = dyn_cast<FuncDecl>(vd))
    llvm_unreachable("visitDeclRefExpr - FuncDecl handling");

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

mlir::Value ExprGenerator::visitBinaryExpr(BinaryExpr *expr) {
  llvm_unreachable("Unimplemented - visitBinaryExpr");
}

mlir::Value ExprGenerator::genUnaryAddressOf(UnaryExpr *expr) {
  llvm_unreachable("Unimplemented - genUnaryAddressOf");
}

mlir::Value ExprGenerator::genUnaryNot(UnaryExpr *expr) {
  llvm_unreachable("Unimplemented - genUnaryNot");
}

mlir::Value ExprGenerator::genUnaryLNot(UnaryExpr *expr) {
  mlir::Location loc = getNodeLoc(expr);
  mlir::Value value = visit(expr->getSubExpr());
  assert(value.getType().isInteger(1) && "Unexpected operand type for LNot!");

  // If the value is an ConstantIntOp, just swap its value instead of generating
  // a XOR.
  if (auto constant = dyn_cast<mlir::ConstantIntOp>(value.getDefiningOp())) {
    constant.setAttr("value", mlir::IntegerAttr::get(constant.getType(),
                                                     !constant.getValue()));
    return value;
  }

  mlir::IntegerType intTy = value.getType().dyn_cast<mlir::IntegerType>();
  assert(intTy && "Not an integer type for Unary LNOT?!");
  mlir::Value one = builder.create<mlir::ConstantIntOp>(loc, 1, intTy);

  return builder.create<mlir::XOrOp>(loc, value, one);
}

static mlir::Value genIntUnaryMinus(mlir::OpBuilder &builder,
                                    mlir::Location loc, mlir::Value value,
                                    mlir::IntegerType intTy) {
  // If the value is a ConstantIntOp, modify it in-place.
  if (auto constant = dyn_cast<mlir::ConstantIntOp>(value.getDefiningOp())) {
    constant.setAttr("value", mlir::IntegerAttr::get(constant.getType(),
                                                     -constant.getValue()));
    return value;
  }

  // Else generate a 0-value.
  mlir::Value zero = builder.create<mlir::ConstantIntOp>(loc, 0, intTy);
  return builder.create<mlir::SubIOp>(loc, zero, value);
}

static mlir::Value genFloatUnaryMinus(mlir::OpBuilder &builder,
                                      mlir::Location loc, mlir::Value value,
                                      mlir::FloatType fltTy) {
  // If the value is a ConstantFloatOp, modify it in-place.
  if (auto constant = dyn_cast<mlir::ConstantFloatOp>(value.getDefiningOp())) {
    constant.setAttr("value", mlir::FloatAttr::get(constant.getType(),
                                                   -constant.getValue()));
    return value;
  }

  // Else generate a 0-value.
  APFloat zero(fltTy.getFloatSemantics(), 0);
  mlir::Value lhs = builder.create<mlir::ConstantFloatOp>(loc, zero, fltTy);
  return builder.create<mlir::SubFOp>(loc, lhs, value);
}

mlir::Value ExprGenerator::genUnaryMinus(UnaryExpr *expr) {
  mlir::Location loc = getNodeLoc(expr);
  mlir::Value value = visit(expr->getSubExpr());

  if (auto intTy = value.getType().dyn_cast<mlir::IntegerType>())
    return genIntUnaryMinus(builder, loc, value, intTy);
  else if (auto fltTy = value.getType().dyn_cast<mlir::FloatType>())
    return genFloatUnaryMinus(builder, loc, value, fltTy);
  else
    llvm_unreachable("Unsupported type for Unary Minus!");
}

mlir::Value ExprGenerator::genUnaryPlus(UnaryExpr *expr) {
  // Unary plus is syntactic sugar - it doesn't do anything.
  return visit(expr->getSubExpr());
}

mlir::Value ExprGenerator::visitUnaryExpr(UnaryExpr *expr) {
  switch (expr->getOpKind()) {
  case UnaryOperatorKind::AddressOf:
    return genUnaryAddressOf(expr);
  case UnaryOperatorKind::Deref:
    llvm_unreachable("This is an LValue and should have been handled by the "
                     "LValueGenerator!");
  case UnaryOperatorKind::Not:
    return genUnaryNot(expr);
  case UnaryOperatorKind::LNot:
    return genUnaryLNot(expr);
  case UnaryOperatorKind::Minus:
    return genUnaryMinus(expr);
  case UnaryOperatorKind::Plus:
    return genUnaryPlus(expr);
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
  return mlir::OpaqueLoc::get(expr, getFileLineColLoc(expr->getLoc()));
}