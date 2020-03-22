//===--- IRGen.hpp - IR Generator Interface ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "llvm/Support/Error.h"
#include "mlir/IR/Function.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
class Type;
} // namespace mlir

namespace sora {
class ASTContext;
class DiagnosticEngine;
class FuncDecl;
class SourceFile;
class Type;

class IRGen {
public:
  IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt);

  /// Generates IR for \p sf, returning the MLIR Module.
  mlir::ModuleOp genSourceFile(SourceFile &sf);

  /// Generates IR for \p func, returning the MLIR function.
  mlir::FuncOp genFunction(FuncDecl *func);

  /// \returns the MLIR Type equivalent of \p type
  mlir::Type getMLIRType(Type type);

  ASTContext &astCtxt;
  mlir::MLIRContext &mlirCtxt;
};

/// A small common base between IR Generators for different AST hierarchies
/// (Expr, Stmt, etc.)
class IRGeneratorBase {
public:
  IRGeneratorBase(ASTContext &&ctxt) : ctxt(ctxt) {}

  ASTContext &ctxt;
};
} // namespace sora