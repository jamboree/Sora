//===--- Dialect.cpp --------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/IR/Dialect.hpp"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Builders.h"

using namespace ::mlir;
using namespace ::sora;
using namespace ::sora::ir;

SoraDialect::SoraDialect(mlir::MLIRContext *mlirCtxt)
    : mlir::Dialect("sora", mlirCtxt) {
  addOperations<
#define GET_OP_LIST
#include "Sora/IR/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd Method Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Sora/IR/Ops.cpp.inc"