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
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace sora;
using namespace sora::ir;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

SoraDialect::SoraDialect(mlir::MLIRContext *mlirCtxt)
    : mlir::Dialect("sora", mlirCtxt) {
  addOperations<
#define GET_OP_LIST
#include "Sora/IR/Ops.cpp.inc"
      >();

  addTypes<MaybeType>();
  addTypes<ReferenceType>();
  addTypes<LValueType>();
}

//===----------------------------------------------------------------------===//
// LoadLValueOp
//===----------------------------------------------------------------------===//

static void build(mlir::OpBuilder &builder, mlir::OperationState &result,
                  mlir::Value &value) {
  LValueType lvalue = value.getType().dyn_cast<LValueType>();
  assert(lvalue && "Value is not an LValue type!");
  result.addTypes(lvalue.getObjectType());
  result.addOperands(value);
}

static mlir::LogicalResult verify(LoadLValueOp op) {
  mlir::Type resultType = op.getType();
  LValueType operandType = op.getOperand().getType().cast<LValueType>();
  return (resultType == operandType.getObjectType()) ? mlir::success()
                                                     : mlir::failure();
}

//===----------------------------------------------------------------------===//
// TableGen'd Method Definitions
//===----------------------------------------------------------------------===//

using namespace ::mlir;
#define GET_OP_CLASSES
#include "Sora/IR/Ops.cpp.inc"