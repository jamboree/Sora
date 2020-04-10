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
}

//===----------------------------------------------------------------------===//
// TableGen'd Method Definitions
//===----------------------------------------------------------------------===//

using namespace ::mlir;
#define GET_OP_CLASSES
#include "Sora/IR/Ops.cpp.inc"