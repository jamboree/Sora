//===--- ConstantFolding.cpp ------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//
/// Implementation of the "fold" methods of the SIR Dialect Operations.
//
//===----------------------------------------------------------------------===//

#include "Sora/SIR/Dialect.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpDefinition.h"

using namespace sora;
using namespace sora::sir;

//===- BitNotOp -----------------------------------------------------------===//

mlir::OpFoldResult BitNotOp::fold(ArrayRef<Attribute> operands) {
  mlir::Operation *op = getOperand().getDefiningOp();

  // If the operand is a constant, we can just invert the constant's bits.
  mlir::ConstantOp constantSrc = dyn_cast_or_null<mlir::ConstantOp>(op);
  if (!constantSrc)
    return {};

  // The attribute must be an integer attribute for us to do something.
  // If it isn't an IntegerAttr something is likely broken, but let's not emit
  // errors here and just let the verifier do its thing.
  auto intAttr = constantSrc.getValue().dyn_cast<mlir::IntegerAttr>();
  if (!intAttr)
    return {};

  llvm::APInt value = intAttr.getValue();
  return mlir::IntegerAttr::get(intAttr.getType(), ~value);
}

//===- DestructureTupleExpr -----------------------------------------------===//

mlir::LogicalResult
DestructureTupleOp::fold(ArrayRef<mlir::Attribute> operands,
                         llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
  mlir::Operation *op = getOperand().getDefiningOp();

  // If the operation is a sir.create_tuple, we can fold.
  auto tupleSrc = dyn_cast_or_null<CreateTupleOp>(op);
  if (!tupleSrc)
    return mlir::failure();

  for (mlir::Value tupleElt : tupleSrc.getOperands())
    results.emplace_back(tupleElt);
  return mlir::success();
}