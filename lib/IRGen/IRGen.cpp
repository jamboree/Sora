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

IRGen::IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt)
    : astCtxt(astCtxt), mlirCtxt(mlirCtxt) {}

mlir::ModuleOp IRGen::genSourceFile(SourceFile &sf) {
  llvm_unreachable("genSourceFile - unimplemented");
}

//===- performIRGen -------------------------------------------------------===//

void sora::performIRGen(mlir::MLIRContext &mlirCtxt, SourceFile &sf) {
  IRGen irGen(sf.astContext, mlirCtxt);
}