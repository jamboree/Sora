//===--- IRGen.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/SourceFile.hpp"
#include "Sora/EntryPoints.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"

using namespace sora;

IRGen::IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt,
             bool enableDebugInfo)
    : debugInfoEnabled(enableDebugInfo), astCtxt(astCtxt), mlirCtxt(mlirCtxt) {}

void IRGen::genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule) {
  llvm_unreachable("genSourceFile - unimplemented");
}

//===- Entry Points -------------------------------------------------------===//

mlir::ModuleOp sora::createMLIRModule(mlir::MLIRContext &mlirCtxt,
                                      SourceFile &sf) {
  // FIXME: Use a correct loc
  return mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirCtxt),
                                sf.getBufferIdentifier());
}

void sora::performIRGen(mlir::MLIRContext &mlirCtxt, mlir::ModuleOp &mlirModule,
                        SourceFile &sf, bool enableDebugInfo) {
  IRGen irGen(sf.astContext, mlirCtxt, enableDebugInfo);
  irGen.genSourceFile(sf, mlirModule);
}