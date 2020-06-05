//===--- SIRGenExpr.cpp - Expressions SIR Generation ------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Types.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace sora;

//===- Common Helpers -----------------------------------------------------===//

/// This is a separate function, as both the LValue and RValue can generate
/// those, but they always generate in the same way.
static mlir::Value genTupleElementExpr(TupleElementExpr *expr,
                                       mlir::OpBuilder &builder, SIRGen &gen) {
  // I think this will be the same for lvalue/non-lvalues.
  // Just have one op, e.g. "get_tuple_element" that can do both, and is lowered
  // to a gep/extractvalue depending on the types.
  llvm_unreachable("Unimplemented - genTupleElementExpr");
}

//===- LValueGenerator ----------------------------------------------------===//

namespace {
class RValueGenerator;

/// The LValue IR Generator generates IR for expressions with an LValue Type.
class LValueGenerator : public SIRGeneratorBase,
                        public ExprVisitor<LValueGenerator, mlir::Value> {
public:
  using Base = ExprVisitor<LValueGenerator, mlir::Value>;

  LValueGenerator(RValueGenerator &rvalueGenerator);

  mlir::OpBuilder &builder;
  RValueGenerator &rvalueGenerator;

  mlir::Value visit(Expr *expr) {
    assert(expr->getType()->is<LValueType>() &&
           "Use genRValue to emit RValue!");
    return Base::visit(expr);
  }

  mlir::Value genRValue(Expr *expr);

  mlir::Value visitExpr(Expr *expr) {
    expr->dump();
    llvm_unreachable("Unhandled LValue expression kind!");
  }

  mlir::Value visitUnresolvedExpr(UnresolvedExpr *) {
    llvm_unreachable("UnresolvedExpr past Sema!");
  }

  mlir::Value visitErrorExpr(ErrorExpr *) {
    llvm_unreachable("ErrorExpr past Sema!");
  }

  mlir::Value visitDeclRefExpr(DeclRefExpr *expr);
  mlir::Value visitDiscardExpr(DiscardExpr *expr);
  mlir::Value visitTupleElementExpr(TupleElementExpr *expr) {
    return genTupleElementExpr(expr, builder, sirGen);
  }
  mlir::Value visitUnaryExpr(UnaryExpr *expr);
};
} // namespace

//===- RValueGenerator ----------------------------------------------------===//

namespace {
/// Class responsible for generating IR for expression with an RValue type.
///
/// This is where most of the logic is. This class owns the LValueGenerator as
/// well, which is used to generate LValues inside expressions.
class RValueGenerator : public SIRGeneratorBase,
                        public ExprVisitor<RValueGenerator, mlir::Value> {
public:
  using Base = ExprVisitor<RValueGenerator, mlir::Value>;

  RValueGenerator(SIRGen &sirGen, mlir::OpBuilder &builder)
      : SIRGeneratorBase(sirGen), builder(builder), lvalueGenerator(*this) {}

  mlir::OpBuilder &builder;
  LValueGenerator lvalueGenerator;

  mlir::Value visitUnresolvedExpr(UnresolvedExpr *) {
    llvm_unreachable("UnresolvedExpr past Sema!");
  }

  mlir::Value visitErrorExpr(ErrorExpr *) {
    llvm_unreachable("ErrorExpr past Sema!");
  }

  mlir::Value visitDeclRefExpr(DeclRefExpr *expr) {
    llvm_unreachable("DeclRefExpr is handled by the LValueGenerator!");
  }

  mlir::Value visitDiscardExpr(DiscardExpr *expr) {
    llvm_unreachable("DiscardExpr is handled by the LValueGenerator!");
  }

  mlir::Value visit(Expr *expr) {
    assert(!expr->getType()->is<LValueType>() &&
           "Use genLValue to emit LValues!");
    mlir::Value result = Base::visit(expr);
    assert(result.getType() == getType(expr) && "Unexpected Operation Type!");
    return result;
  }

  /// Generates an LValue, returning a Value with an LValue type.
  mlir::Value genLValue(Expr *expr) { return lvalueGenerator.visit(expr); }

  // UnaryExpr Helpers
  mlir::Value genUnaryAddressOf(UnaryExpr *expr);
  mlir::Value genUnaryNot(UnaryExpr *expr);
  mlir::Value genUnaryLNot(UnaryExpr *expr);
  mlir::Value genUnaryMinus(UnaryExpr *expr);
  mlir::Value genUnaryPlus(UnaryExpr *expr);

  // Visit Methods
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
  mlir::Value visitTupleElementExpr(TupleElementExpr *expr) {
    return genTupleElementExpr(expr, builder, sirGen);
  }
  mlir::Value visitTupleExpr(TupleExpr *expr);
  mlir::Value visitParenExpr(ParenExpr *expr);
  mlir::Value visitCallExpr(CallExpr *expr);
  mlir::Value visitConditionalExpr(ConditionalExpr *expr);
  mlir::Value visitForceUnwrapExpr(ForceUnwrapExpr *expr);
  mlir::Value visitBinaryExpr(BinaryExpr *expr);
  mlir::Value visitUnaryExpr(UnaryExpr *expr);
};
} // namespace

//===- LValueGenerator Impl ---------------------------------------------===//

LValueGenerator::LValueGenerator(RValueGenerator &rvalueGenerator)
    : SIRGeneratorBase(rvalueGenerator.sirGen),
      builder(rvalueGenerator.builder), rvalueGenerator(rvalueGenerator) {}

mlir::Value LValueGenerator::genRValue(Expr *expr) {
  return rvalueGenerator.visit(expr);
}

mlir::Value LValueGenerator::visitDeclRefExpr(DeclRefExpr *expr) {
  // This is awaiting a fix in the - currently broken - variable handling logic.
  llvm_unreachable("Unimplemented - visitDeclRefExpr");
}

mlir::Value LValueGenerator::visitDiscardExpr(DiscardExpr *expr) {
  // This will be added later as I currently don't have any idea of the best way
  // to approach this.
  llvm_unreachable("Unimplemented - visitDiscardExpr");
}

mlir::Value LValueGenerator::visitUnaryExpr(UnaryExpr *expr) {
  assert(expr->getOpKind() == UnaryOperatorKind::Deref &&
         "Only deref can be an LValue!");
  // This should only work with dereference.
  llvm_unreachable("Unimplemented - visitUnaryExpr");
}

//===- RValueGenerator Impl ---------------------------------------------===//

mlir::Value RValueGenerator::visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
  assert(expr->getType()->isAnyIntegerType() && "Not an Integer Type?!");
  mlir::Type type = getType(expr->getType());
  assert(type.isa<mlir::IntegerType>() && "Not an IntegerType?!");

  // TODO: Create a "BuildIntConstant" method as I'll probably need to create
  // int constants a lot.
  auto valueAttr = mlir::IntegerAttr::get(type, expr->getValue());
  return builder.create<mlir::ConstantOp>(getNodeLoc(expr), valueAttr);
}

mlir::Value RValueGenerator::visitFloatLiteralExpr(FloatLiteralExpr *expr) {
  assert(expr->getType()->isAnyFloatType() && "Not a Float Type?!");
  mlir::Type type = getType(expr->getType());
  assert(type.isa<mlir::FloatType>() && "Not a FloatType?!");

  auto valueAttr = mlir::FloatAttr::get(type, expr->getValue());
  return builder.create<mlir::ConstantOp>(getNodeLoc(expr), valueAttr);
}

mlir::Value RValueGenerator::visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
  assert(expr->getType()->isBoolType() && "Not a Bool Type?!");
  mlir::Type type = getType(expr->getType());
  assert(type.isInteger(1) && "Expected a i1!");
  APInt value(1, expr->getValue() ? 1 : 0);

  auto valueAttr = mlir::IntegerAttr::get(type, value);
  return builder.create<mlir::ConstantOp>(getNodeLoc(expr), valueAttr);
}

mlir::Value RValueGenerator::visitNullLiteralExpr(NullLiteralExpr *expr) {
  // "null" is a contextual thing. It's only emitted through its parent,
  // usually a MaybeConversionExpr.
  llvm_unreachable(
      "NullLiteralExprs should only be emitted through their parents!");
}

mlir::Value RValueGenerator::visitImplicitMaybeConversionExpr(
    ImplicitMaybeConversionExpr *expr) {
  // TODO: Handle "NullLiteralExpr" as child expr as a special case.
  llvm_unreachable("Unimplemented - visitImplicitMaybeConversionExpr");
}

mlir::Value
RValueGenerator::visitMutToImmutReferenceExpr(MutToImmutReferenceExpr *expr) {
  // As there is no distinction between mutable and immutable references in the
  // IR, we don't have to do anything.
  return visit(expr->getSubExpr());
}

mlir::Value
RValueGenerator::visitDestructuredTupleExpr(DestructuredTupleExpr *expr) {
  llvm_unreachable("Unimplemented - visitDestructuredTupleExpr");
}

mlir::Value RValueGenerator::visitDestructuredTupleElementExpr(
    DestructuredTupleElementExpr *expr) {
  llvm_unreachable("Unimplemented - visitDestructuredTupleElementExpr");
}

mlir::Value RValueGenerator::visitLoadExpr(LoadExpr *expr) {
  return builder.create<sir::LoadOp>(getNodeLoc(expr),
                                     genLValue(expr->getSubExpr()));
}

mlir::Value RValueGenerator::visitCastExpr(CastExpr *expr) {
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

mlir::Value RValueGenerator::visitTupleExpr(TupleExpr *expr) {
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

mlir::Value RValueGenerator::visitParenExpr(ParenExpr *expr) {
  return visit(expr->getSubExpr());
}

mlir::Value RValueGenerator::visitCallExpr(CallExpr *expr) {
  llvm_unreachable("Unimplemented - visitCallExpr");
}

mlir::Value RValueGenerator::visitConditionalExpr(ConditionalExpr *expr) {
  llvm_unreachable("Unimplemented - visitConditionalExpr");
}

mlir::Value RValueGenerator::visitForceUnwrapExpr(ForceUnwrapExpr *expr) {
  llvm_unreachable("Unimplemented - visitForceUnwrapExpr");
}

mlir::Value RValueGenerator::visitBinaryExpr(BinaryExpr *expr) {
  llvm_unreachable("Unimplemented - visitBinaryExpr");
}

mlir::Value RValueGenerator::genUnaryAddressOf(UnaryExpr *expr) {
  llvm_unreachable("Unimplemented - genUnaryAddressOf");
}

mlir::Value RValueGenerator::genUnaryNot(UnaryExpr *expr) {
  llvm_unreachable("Unimplemented - genUnaryNot");
}

mlir::Value RValueGenerator::genUnaryLNot(UnaryExpr *expr) {
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

mlir::Value RValueGenerator::genUnaryMinus(UnaryExpr *expr) {
  mlir::Location loc = getNodeLoc(expr);
  mlir::Value value = visit(expr->getSubExpr());

  if (auto intTy = value.getType().dyn_cast<mlir::IntegerType>())
    return genIntUnaryMinus(builder, loc, value, intTy);
  else if (auto fltTy = value.getType().dyn_cast<mlir::FloatType>())
    return genFloatUnaryMinus(builder, loc, value, fltTy);
  else
    llvm_unreachable("Unsupported type for Unary Minus!");
}

mlir::Value RValueGenerator::genUnaryPlus(UnaryExpr *expr) {
  // Unary plus is syntactic sugar - it doesn't do anything.
  return visit(expr->getSubExpr());
}

mlir::Value RValueGenerator::visitUnaryExpr(UnaryExpr *expr) {
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
  assert(!expr->getType()->is<LValueType>() &&
         "This entry point is only for RValue expressions");
  return RValueGenerator(*this, builder).visit(expr);
}

mlir::Type SIRGen::getType(Expr *expr) { return getType(expr->getType()); }

mlir::Location SIRGen::getNodeLoc(Expr *expr) {
  return mlir::OpaqueLoc::get(expr, getFileLineColLoc(expr->getLoc()));
}