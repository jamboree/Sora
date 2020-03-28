//===--- IRGenExpr.cpp - Expressions IR Generation --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Expr.hpp"
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

  mlir::Value visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
    llvm_unreachable("Unimplemented - visitIntegerLiteralExpr");
  }

  mlir::Value visitFloatLiteralExpr(FloatLiteralExpr *expr) {
    llvm_unreachable("Unimplemented - visitFloatLiteralExpr");
  }

  mlir::Value visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
    return builder.create<mlir::ConstantIntOp>(
        irGen.getMLIRLoc(expr), expr->getValue() ? 1 : 0,
        irGen.getMLIRType(expr->getType()));
  }

  mlir::Value visitNullLiteralExpr(NullLiteralExpr *expr) {
    // "null" is a contextual thing. It's only emitted through its parent,
    // usually a MaybeConversionExpr.
    llvm_unreachable(
        "NullLiteralExprs should only be emitted through their parents!");
  }

  mlir::Value
  visitImplicitMaybeConversionExpr(ImplicitMaybeConversionExpr *expr) {
    llvm_unreachable("Unimplemented - visitImplicitMaybeConversionExpr");
  }

  mlir::Value visitMutToImmutReferenceExpr(MutToImmutReferenceExpr *expr) {
    llvm_unreachable("Unimplemented - visitMutToImmutReferenceExpr");
  }

  mlir::Value visitDestructuredTupleExpr(DestructuredTupleExpr *expr) {
    llvm_unreachable("Unimplemented - visitDestructuredTupleExpr");
  }

  mlir::Value
  visitDestructuredTupleElementExpr(DestructuredTupleElementExpr *expr) {
    llvm_unreachable("Unimplemented - visitDestructuredTupleElementExpr");
  }

  mlir::Value visitCastExpr(CastExpr *expr) {
    llvm_unreachable("Unimplemented - visitCastExpr");
  }

  mlir::Value visitTupleElementExpr(TupleElementExpr *expr) {
    llvm_unreachable("Unimplemented - visitTupleElementExpr");
  }

  mlir::Value visitTupleExpr(TupleExpr *expr) {
    llvm_unreachable("Unimplemented - visitTupleExpr");
  }

  mlir::Value visitParenExpr(ParenExpr *expr) {
    return visit(expr->getSubExpr());
  }

  mlir::Value visitCallExpr(CallExpr *expr) {
    llvm_unreachable("Unimplemented - visitCallExpr");
  }

  mlir::Value visitConditionalExpr(ConditionalExpr *expr) {
    llvm_unreachable("Unimplemented - visitConditionalExpr");
  }

  mlir::Value visitForceUnwrapExpr(ForceUnwrapExpr *expr) {
    llvm_unreachable("Unimplemented - visitForceUnwrapExpr");
  }

  mlir::Value visitBinaryExpr(BinaryExpr *expr) {
    llvm_unreachable("Unimplemented - visitBinaryExpr");
  }

  mlir::Value visitUnaryExpr(UnaryExpr *expr) {
    llvm_unreachable("Unimplemented - visitUnaryExpr");
  }
};
} // namespace

//===- IRGen --------------------------------------------------------------===//

mlir::Value IRGen::genExpr(Expr *expr, mlir::OpBuilder builder) {
  return ExprIRGenerator(*this, builder).visit(expr);
}

mlir::Type IRGen::getMLIRType(Expr *expr) {
  return getMLIRType(expr->getType());
}