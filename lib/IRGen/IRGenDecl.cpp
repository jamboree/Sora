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

void IRGen::genVarDecl(mlir::OpBuilder &builder, VarDecl *decl,
                       mlir::Value value) {
  assert(vars.count(decl) == 0 && "Variable has already been declared");

  // If we don't have an initial value, generate a default one.
  if (!value)
    value = builder.create<ir::CreateDefaultValueOp>(
        getNodeLoc(decl), getType(decl->getValueType()));

  vars.insert(decl, value);
}

void IRGen::genDecl(mlir::OpBuilder &builder, Decl *decl) {
  assert(!isa<VarDecl>(decl) &&
         "VarDecls can only be generated through genVarDecl!");

  // This is temporary code for testing purposes.
  if (LetDecl *let = dyn_cast<LetDecl>(decl)) {
    mlir::Value initialValue;
    if (Expr *init = let->getInitializer())
      initialValue = genExpr(builder, init);
    genPattern(builder, let->getPattern(), initialValue);
    return;
  }

  llvm_unreachable("Unimplemented - Decl IRGen");
}

void IRGen::setVarValue(VarDecl *decl, mlir::Value value) {
  assert(vars.count(decl) != 0 &&
         "Variable has not been declared yet, use genVarDecl first.");
  assert(value && "value cannot be null!");
  vars.insert(decl, value);
}

mlir::Value sora::IRGen::getVarValue(VarDecl *decl) {
  assert(vars.count(decl) != 0 && "Variable has not been declared yet!");
  return vars.lookup(decl);
}

mlir::FuncOp IRGen::getFuncOp(FuncDecl *func) {
  auto it = funcCache.find(func);
  if (it != funcCache.end())
    return it->second;

  auto name = getIRIdentifier(func->getIdentifier());
  mlir::Location loc = getNodeLoc(func);

  Type fnTy = func->getValueType()->getAs<FunctionType>();
  assert(fnTy->is<FunctionType>() && "Function's type is not a FunctionType?!");
  auto mlirFuncTy = getType(fnTy).cast<mlir::FunctionType>();

  auto funcOp = mlir::FuncOp::create(loc, name, mlirFuncTy);
  funcCache.insert({func, funcOp});
  return funcOp;
}

mlir::FuncOp IRGen::genFunction(FuncDecl *func) {
  mlir::FuncOp funcOp = getFuncOp(func);
  // FIXME: Is this the proper way to check if the body is empty?
  if (!funcOp.getBody().empty())
    return funcOp;

  mlir::Block *entryBlock = funcOp.addEntryBlock();

  mlir::OpBuilder builder(&mlirCtxt);
  builder.setInsertionPointToStart(entryBlock);

  {
    BlockScope blockScope(*this);
    genFunctionBody(builder, func->getBody());
  }

  return funcOp;
}

mlir::Location IRGen::getNodeLoc(Decl *decl) {
  return mlir::OpaqueLoc::get(decl, getFileLineColLoc(decl->getLoc()));
}