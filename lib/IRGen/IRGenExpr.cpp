//===--- IRGenExpr.cpp - Expressions IR Generation --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Types.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace sora;

//===- ExprIRGenerator ----------------------------------------------------===//

namespace {
class ExprIRGenerator : public IRGeneratorBase,
                        public ExprVisitor<ExprIRGenerator, mlir::Value> {
public:
  ExprIRGenerator(IRGen &irGen, mlir::OpBuilder builder)
      : IRGeneratorBase(irGen), builder(builder) {}

  mlir::OpBuilder builder;

  /// \returns true of if \p type is a bool or integer type
  bool isIntegerOrIntegerLike(Type type) {
    return type->isBoolType() || type->isAnyIntegerType();
  }

  mlir::Value visitUnresolvedExpr(UnresolvedExpr *) {
    llvm_unreachable("UnresolvedExpr past Sema!");
  }

  mlir::Value visitErrorExpr(ErrorExpr *) {
    llvm_unreachable("ErrorExpr past Sema!");
  }

  mlir::Value visitDeclRefExpr(DeclRefExpr *expr) {
    llvm_unreachable("Unimplemented - visitDeclRefExpr");
  }

  mlir::Value visitDiscardExpr(DiscardExpr *expr) {
    llvm_unreachable("Unimplemented - visitDiscardExpr");
  }

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

mlir::Value ExprIRGenerator::visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
  assert(expr->getType()->isAnyIntegerType() && "Not an Integer Type?!");
  return builder.create<ir::IntegerConstantOp>(
      irGen.getMLIRLoc(expr), expr->getValue(),
      irGen.getMLIRType(expr->getType()));
}

mlir::Value ExprIRGenerator::visitFloatLiteralExpr(FloatLiteralExpr *expr) {
  assert(expr->getType()->isAnyFloatType() && "Not a Float Type?!");
  mlir::Type type = irGen.getMLIRType(expr->getType());
  assert(type.isa<mlir::FloatType>() && "Not a FloatType?!");
  return builder.create<mlir::ConstantFloatOp>(
      irGen.getMLIRLoc(expr), expr->getValue(), type.cast<mlir::FloatType>());
}

mlir::Value ExprIRGenerator::visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
  assert(expr->getType()->isBoolType() && "Not a Bool Type?!");
  return builder.create<ir::BoolConstantOp>(irGen.getMLIRLoc(expr),
                                            expr->getValue());
}

mlir::Value ExprIRGenerator::visitNullLiteralExpr(NullLiteralExpr *expr) {
  // "null" is a contextual thing. It's only emitted through its parent,
  // usually a MaybeConversionExpr.
  llvm_unreachable(
      "NullLiteralExprs should only be emitted through their parents!");
}

mlir::Value ExprIRGenerator::visitImplicitMaybeConversionExpr(
    ImplicitMaybeConversionExpr *expr) {
  llvm_unreachable("Unimplemented - visitImplicitMaybeConversionExpr");
}

mlir::Value
ExprIRGenerator::visitMutToImmutReferenceExpr(MutToImmutReferenceExpr *expr) {
  llvm_unreachable("Unimplemented - visitMutToImmutReferenceExpr");
}

mlir::Value
ExprIRGenerator::visitDestructuredTupleExpr(DestructuredTupleExpr *expr) {
  llvm_unreachable("Unimplemented - visitDestructuredTupleExpr");
}
mlir::Value ExprIRGenerator::visitDestructuredTupleElementExpr(
    DestructuredTupleElementExpr *expr) {
  llvm_unreachable("Unimplemented - visitDestructuredTupleElementExpr");
}

mlir::Value ExprIRGenerator::visitCastExpr(CastExpr *expr) {
  Expr *subExpr = expr->getSubExpr();

  // Generate IR for the subexpression, and if the cast is "useless", stop here.
  mlir::Value subExprValue = visit(subExpr);
  if (expr->isUseless())
    return subExprValue;

  Type subExprType = subExpr->getType();
  Type type = expr->getType();

  // Convert the result type and the loc to their MLIR equivalent.
  mlir::Type mlirType = irGen.getMLIRType(type);
  mlir::Location loc = irGen.getMLIRLoc(expr->getLoc());

  // Integer conversions
  if (isIntegerOrIntegerLike(subExprType) && isIntegerOrIntegerLike(type))
    return builder.create<ir::IntegerCastOp>(loc, mlirType, subExprValue);

  llvm_unreachable("Unimplemented Cast Kind");
}

mlir::Value ExprIRGenerator::visitTupleElementExpr(TupleElementExpr *expr) {
  llvm_unreachable("Unimplemented - visitTupleElementExpr");
}

mlir::Value ExprIRGenerator::visitTupleExpr(TupleExpr *expr) {
  llvm_unreachable("Unimplemented - visitTupleExpr");
}

mlir::Value ExprIRGenerator::visitParenExpr(ParenExpr *expr) {
  return visit(expr->getSubExpr());
}

mlir::Value ExprIRGenerator::visitCallExpr(CallExpr *expr) {
  llvm_unreachable("Unimplemented - visitCallExpr");
}

mlir::Value ExprIRGenerator::visitConditionalExpr(ConditionalExpr *expr) {
  llvm_unreachable("Unimplemented - visitConditionalExpr");
}

mlir::Value ExprIRGenerator::visitForceUnwrapExpr(ForceUnwrapExpr *expr) {
  llvm_unreachable("Unimplemented - visitForceUnwrapExpr");
}

mlir::Value ExprIRGenerator::visitBinaryExpr(BinaryExpr *expr) {
  llvm_unreachable("Unimplemented - visitBinaryExpr");
}

mlir::Value ExprIRGenerator::visitUnaryExpr(UnaryExpr *expr) {
  llvm_unreachable("Unimplemented - visitUnaryExpr");
}

//===- IRGen --------------------------------------------------------------===//

mlir::Value IRGen::genExpr(Expr *expr, mlir::OpBuilder builder) {
  return ExprIRGenerator(*this, builder).visit(expr);
}

mlir::Type IRGen::getMLIRType(Expr *expr) {
  return getMLIRType(expr->getType());
}