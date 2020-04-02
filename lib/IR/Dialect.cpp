//===--- Dialect.cpp --------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/IR/Dialect.hpp"
#include "Sora/IR/Types.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LogicalResult.h"

using namespace sora;
using namespace sora::ir;

SoraDialect::SoraDialect(mlir::MLIRContext *mlirCtxt)
    : mlir::Dialect("sora", mlirCtxt) {
  addOperations<
#define GET_OP_LIST
#include "Sora/IR/Ops.cpp.inc"
      >();

  addTypes<MaybeType>();
}

//===- Operation Verification ---------------------------------------------===//

namespace {

/// The IntegerConstant's attribute's type must match its return type.
mlir::LogicalResult verify(IntegerConstantOp &op) {
  if (op.getType() != op.valueAttr().getType())
    return mlir::failure();
  return mlir::success();
}

/// The FloatConstant's attribute's type must match its return type.
mlir::LogicalResult verify(FloatConstantOp &op) {
  if (op.getType() != op.valueAttr().getType())
    return mlir::failure();
  return mlir::success();
}

} // namespace

//===----------------------------------------------------------------------===//
// TableGen'd Method Definitions
//===----------------------------------------------------------------------===//

using namespace ::mlir;
#define GET_OP_CLASSES
#include "Sora/IR/Ops.cpp.inc"