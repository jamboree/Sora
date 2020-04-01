//===--- Dialect.hpp - Sora IR MLIR Dialect Declaration ---------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffects.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace sora {

class SoraDialect : public mlir::Dialect {
public:
  explicit SoraDialect(mlir::MLIRContext *mlirCtxt);

  void printType(Type type, DialectAsmPrinter &os) const override;

  static llvm::StringRef getDialectNamespace() { return "sora"; }
};

// Include the TableGen'd file containing the declarations of the Sora IR
// Operations.
#define GET_OP_CLASSES
#include "Sora/IR/Ops.h.inc"
} // namespace sora
} // namespace mlir

namespace sora {
namespace ir {
/// This is a dirty hack to have the Sora IR reside in the sora::ir namespace.
/// Why? The MLIR TableGen backend doesn't fully qualify names, so the dialect
/// *has* to be in the mlir namespace, but typing "mlir::sora" will get
/// verbose quickly when manipulating IR, so this is a way of shortening the
/// namespace.
///
/// TL;DR: Always use sora::ir to access IR stuff.
using namespace ::mlir::sora;
} // namespace ir
} // namespace sora
