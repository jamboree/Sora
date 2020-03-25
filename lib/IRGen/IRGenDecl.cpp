//===--- IRGenDecl.cpp - Declarations IR Generation -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Types.hpp"
#include "mlir/IR/Types.h"

using namespace sora;

//===- IRGen --------------------------------------------------------------===//

mlir::Location IRGen::getFuncDeclLoc(FuncDecl *func) {
  if (!debugInfoEnabled)
    return mlir::UnknownLoc::get(&mlirCtxt);
  StringRef filename = func->getSourceFile().getBufferIdentifier();
  std::pair<unsigned, unsigned> loc = srcMgr.getLineAndColumn(func->getLoc());
  return mlir::FileLineColLoc::get(filename, loc.first, loc.second, &mlirCtxt);
}

mlir::FuncOp IRGen::genFunction(FuncDecl *func) {
  auto name = getMLIRIdentifier(func->getIdentifier());
  mlir::Location loc = getFuncDeclLoc(func);

  Type fnTy = func->getValueType()->getAs<FunctionType>();
  assert(fnTy->is<FunctionType>() && "Function's type is not a FunctionType?!");
  mlir::FunctionType mlirFuncTy = getMLIRType(fnTy).cast<mlir::FunctionType>();
  return mlir::FuncOp::create(loc, name, mlirFuncTy);
}