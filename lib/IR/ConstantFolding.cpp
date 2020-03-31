//===--- ConstantFolding.cpp ------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//
/// Implementation of the "fold" methods of the Sora Dialect Operations.
//
//===----------------------------------------------------------------------===//

#include "Sora/IR/Dialect.hpp"
#include "mlir/IR/OpDefinition.h"

using namespace sora;
using namespace sora::ir;


mlir::OpFoldResult
IntegerConstantOp::fold(llvm::ArrayRef<mlir::Attribute> attrs) {
  return valueAttr();
}

mlir::OpFoldResult
FloatConstantOp::fold(llvm::ArrayRef<mlir::Attribute> attrs) {
  return valueAttr();
}

mlir::OpFoldResult BoolConstantOp::fold(llvm::ArrayRef<mlir::Attribute> attrs) {
  return valueAttr();
}

