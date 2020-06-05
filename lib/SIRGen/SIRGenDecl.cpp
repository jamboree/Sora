//===--- SIRGenDecl.cpp - Declarations SIR Generation -----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Types.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Types.h"

using namespace sora;

//===- SIRGen -------------------------------------------------------------===//

mlir::Value SIRGen::genVarDeclAlloc(mlir::OpBuilder &builder, VarDecl *decl) {
  assert(varAddresses.count(decl) == 0 && "Variable has already been declared");

  sir::PointerType type = sir::PointerType::get(getType(decl->getValueType()));

  mlir::Value address =
      builder.create<sir::AllocStackOp>(getNodeLoc(decl), type);
  varAddresses.insert(decl, address);
  return address;
}

void SIRGen::genDecl(mlir::OpBuilder &builder, Decl *decl) {
  assert(!isa<VarDecl>(decl) &&
         "VarDecls can only be generated through genVarDecl!");

  // This is temporary code for testing purposes.
  if (LetDecl *let = dyn_cast<LetDecl>(decl)) {
    Pattern *pattern = let->getPattern();
    if (Expr *init = let->getInitializer()) {
      mlir::Value value = genExpr(builder, init);
      genPattern(builder, pattern, value);
    }
    else
      genPattern(builder, pattern);
    return;
  }

  llvm_unreachable("Unimplemented - Decl SIRGen");
}

mlir::Value sora::SIRGen::getVarDeclAddress(VarDecl *decl) {
  assert(varAddresses.count(decl) != 0 && "Unknown variable");
  return varAddresses.lookup(decl);
}

mlir::FuncOp SIRGen::getFuncOp(FuncDecl *func) {
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

mlir::FuncOp SIRGen::genFunction(FuncDecl *func) {
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

mlir::Location SIRGen::getNodeLoc(Decl *decl) {
  return mlir::OpaqueLoc::get(decl, getFileLineColLoc(decl->getLoc()));
}