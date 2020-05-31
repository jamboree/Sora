//===--- Dialect.cpp --------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/IR/Dialect.hpp"

#include "Sora/Common/LLVM.hpp"
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
  addTypes<PointerType>();
  addTypes<VoidType>();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

static void buildLoadOp(mlir::OpBuilder &builder, mlir::OperationState &result,
                        mlir::Value &value) {
  PointerType pointer = value.getType().dyn_cast<PointerType>();
  assert(pointer && "Value is not a Pointer type!");
  result.addTypes(pointer.getObjectType());
  result.addOperands(value);
}

static mlir::LogicalResult verifyLoadOp(LoadOp op) {
  mlir::Type resultType = op.getType();
  PointerType operandType = op.getOperand().getType().cast<PointerType>();
  return (resultType == operandType.getObjectType()) ? mlir::success()
                                                     : mlir::failure();
}

//===----------------------------------------------------------------------===//
// CreateTupleOp
//===----------------------------------------------------------------------===//

static void buildCreateTupleOp(mlir::OpBuilder &builder,
                               mlir::OperationState &result,
                               ArrayRef<mlir::Value> elts) {
  assert(elts.size() > 0 && "Cannot create empty tuples!");
  assert(elts.size() > 1 && "Cannot create single-element tuples");

  SmallVector<mlir::Type, 8> types;
  types.reserve(elts.size());

  for (const mlir::Value &elt : elts)
    types.push_back(elt.getType());

  mlir::TupleType tupleType = mlir::TupleType::get(types, builder.getContext());

  result.addOperands(elts);
  result.addTypes(tupleType);
}

static mlir::LogicalResult verifyCreateTupleOp(CreateTupleOp op) {
  if (op.getNumOperands() <= 1)
    return mlir::failure();

  SmallVector<mlir::Type, 8> types;
  types.reserve(op.getNumOperands());

  for (const mlir::Value &elt : op.getOperands())
    types.push_back(elt.getType());

  if (op.getType() != mlir::TupleType::get(types, op.getContext()))
    return mlir::failure();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// DestructureTupleOp
//===----------------------------------------------------------------------===//

static void buildDestructureTupleOp(mlir::OpBuilder &builder,
                                    mlir::OperationState &result,
                                    mlir::Value tuple) {
  mlir::TupleType tupleType = tuple.getType().dyn_cast<mlir::TupleType>();
  assert(tupleType && "The value must have a TupleType!");
  assert(tupleType.getTypes().size() > 1 && "Illegal tuple!");

  result.addOperands(tuple);
  result.addTypes(tupleType.getTypes());
}

static mlir::LogicalResult verifyDestructureTupleOp(DestructureTupleOp op) {
  mlir::TupleType tupleType =
      op.getOperand().getType().dyn_cast<mlir::TupleType>();
  if (!tupleType)
    return mlir::failure();

  ArrayRef<mlir::Type> results = op.getResultTypes();
  ArrayRef<mlir::Type> tupleElts = tupleType.getTypes();

  if (results.size() != tupleElts.size())
    return mlir::failure();

  if (results.size() <= 1)
    return mlir::failure();

  for (size_t k = 0; k < results.size(); ++k)
    if (results[k] != tupleElts[k])
      return mlir::failure();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AnyBlock
//===----------------------------------------------------------------------===//

static void printAnyBlock(mlir::OpAsmPrinter &p, mlir::Region &region,
                          const char *name) {
  p << "sora." << name << " ";
  p.printRegion(region);
}

static mlir::ParseResult parseAnyBlock(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::Region *region = result.addRegion();
  if (parser.parseRegion(*region, llvm::None, llvm::None))
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd Method Definitions
//===----------------------------------------------------------------------===//

using namespace ::mlir;
#define GET_OP_CLASSES
#include "Sora/IR/Ops.cpp.inc"