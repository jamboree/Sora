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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Types.h"

using namespace sora;

//===- IRGen --------------------------------------------------------------===//

void IRGen::genDecl(Decl *decl, mlir::OpBuilder builder) {
  llvm_unreachable("Unimplemented - Decl IRGen");
}

mlir::FuncOp IRGen::getFuncOp(FuncDecl *func) {
  auto it = funcCache.find(func);
  if (it != funcCache.end())
    return it->second;

  auto name = getIRIdentifier(func->getIdentifier());
  mlir::Location loc = getNodeLoc(func);

  Type fnTy = func->getValueType()->getAs<FunctionType>();
  assert(fnTy->is<FunctionType>() && "Function's type is not a FunctionType?!");
  auto mlirFuncTy = getIRType(fnTy).cast<mlir::FunctionType>();

  auto funcOp = mlir::FuncOp::create(loc, name, mlirFuncTy);
  funcCache.insert({func, funcOp});
  return funcOp;
}

mlir::FuncOp IRGen::genFunctionBody(FuncDecl *func) {
  mlir::FuncOp funcOp = getFuncOp(func);
  // FIXME: Is this the proper way to check if the body is empty?
  if (!funcOp.getBody().empty())
    return funcOp;

  mlir::Block *entryBlock = funcOp.addEntryBlock();

  mlir::OpBuilder builder(&mlirCtxt);
  builder.setInsertionPointToStart(entryBlock);

  genStmt(func->getBody(), builder);

  return funcOp;
}

mlir::Location IRGen::getNodeLoc(Decl *decl) {
  return mlir::OpaqueLoc::get(decl, getFileLineColLoc(decl->getLoc()));
}