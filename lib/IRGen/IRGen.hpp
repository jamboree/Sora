//===--- IRGen.hpp - IR Generator Interface ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/IR/Dialect.hpp"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace sora {
class ASTContext;
class DiagnosticEngine;
class FuncDecl;
class SourceFile;
class SourceManager;
class Type;

class IRGen {
  bool debugInfoEnabled;

  llvm::DenseMap<TypeBase *, mlir::Type> typeCache;

public:
  IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt, bool enableDebugInfo);

  /// Generates IR for \p sf, returning the MLIR Module.
  void genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule);

  /// Generates IR for \p func, returning the MLIR function.
  mlir::FuncOp genFunction(FuncDecl *func);

  /// Generates IR for \p func, returning the MLIR function.
  mlir::Location getFuncDeclLoc(FuncDecl *func);

  /// \returns the MLIR Type equivalent of \p type
  mlir::Type getMLIRType(Type type);

  /// \returns the MLIR Identifier for \p str
  mlir::Identifier getMLIRIdentifier(StringRef str);

  /// \returns the MLIR Identifier for \p str
  mlir::Identifier getMLIRIdentifier(const char *str) {
    return getMLIRIdentifier(StringRef(str));
  }

  /// \returns the MLIR Identifier for \p ident
  mlir::Identifier getMLIRIdentifier(Identifier ident) {
    return getMLIRIdentifier(ident.c_str());
  }

  ASTContext &astCtxt;
  const SourceManager &srcMgr;
  mlir::MLIRContext &mlirCtxt;
  mlir::LLVM::LLVMDialect &llvmDialect;
  ir::SoraDialect &soraDialect;
};

/// A small common base between IR Generators for different AST hierarchies
/// (Expr, Stmt, etc.)
class IRGeneratorBase {
public:
  IRGeneratorBase(IRGen &irGen)
      : astCtxt(irGen.astCtxt), mlirCtxt(irGen.mlirCtxt) {}

  ASTContext &astCtxt;
  mlir::MLIRContext &mlirCtxt;
};
} // namespace sora