//===--- Dialect.hpp - Sora IR MLIR Dialect Declaration ---------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace sir {

class SIRDialect : public mlir::Dialect {
public:
  explicit SIRDialect(mlir::MLIRContext *mlirCtxt);

  void printType(Type type, DialectAsmPrinter &os) const override;
  Type parseType(DialectAsmParser &parser) const override;

  static llvm::StringRef getDialectNamespace() { return "sir"; }
};

// Include the TableGen'd file containing the declarations of the Sora IR
// Operations.
#define GET_OP_CLASSES
#include "Sora/SIR/Ops.h.inc"
} // namespace sir
} // namespace mlir

namespace sora {
namespace sir {
/// This is a small hack to have the Sora IR reside in the sora::sir namespace.
///
/// Why? The MLIR TableGen backend doesn't fully qualify names, so the dialect
/// *has* to be in the mlir::sir namespace, but we want it to be in the
/// sora::sir namespace.
///
/// TL;DR: Always use sora::sir to access SIR stuff.
using namespace ::mlir::sir;
} // namespace sir
} // namespace sora
