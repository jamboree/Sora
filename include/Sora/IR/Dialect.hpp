//===--- Dialect.hpp - Sora IR MLIR Dialect Declaration ---------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Dialect.h"

namespace sora {
namespace ir {
class SoraDialect : public mlir::Dialect {
public:
  explicit SoraDialect(mlir::MLIRContext *mlirCtxt);

  static llvm::StringRef getDialectNamespace() { return "sora"; }
};
} // namespace ir
} // namespace sora
