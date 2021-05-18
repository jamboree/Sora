//===--- SIRGen.cpp ---------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/EntryPoints.hpp"
#include "mlir/IR/MLIRContext.h"

using namespace sora;

//===- SIRGen -------------------------------------------------------------===//

SIRGen::SIRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt,
               bool enableDebugInfo)
    : debugInfoEnabled(enableDebugInfo), astCtxt(astCtxt),
      srcMgr(astCtxt.srcMgr), mlirCtxt(mlirCtxt) {}

void SIRGen::genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule) {
  for (ValueDecl *decl : sf.getMembers())
    mlirModule.push_back(genFunction(cast<FuncDecl>(decl)));
}

mlir::Location SIRGen::getLoc(SourceLoc loc) {
  if (!debugInfoEnabled)
    return mlir::UnknownLoc::get(&mlirCtxt);

  BufferID buffer = srcMgr.findBufferContainingLoc(loc);
  StringRef filename = srcMgr.getBufferName(buffer);
  std::pair<unsigned, unsigned> lineAndCol = srcMgr.getLineAndColumn(loc);
  return mlir::FileLineColLoc::get(&mlirCtxt, filename, lineAndCol.first,
                                   lineAndCol.second);
}

mlir::Identifier SIRGen::getIRIdentifier(StringRef str) {
  return mlir::Identifier::get(str, &mlirCtxt);
}

//===- Entry Points -------------------------------------------------------===//

mlir::ModuleOp sora::createMLIRModule(mlir::MLIRContext &mlirCtxt,
                                      SourceFile &sf) {
  auto bufferID = mlir::Identifier::get(sf.getBufferName(), &mlirCtxt);
  auto nameLoc = mlir::NameLoc::get(bufferID);
  return mlir::ModuleOp::create(nameLoc, sf.getBufferName());
}

void sora::performSIRGen(mlir::MLIRContext &mlirCtxt,
                         mlir::ModuleOp &mlirModule, SourceFile &sf,
                         bool enableDebugInfo) {
  SIRGen(sf.astContext, mlirCtxt, enableDebugInfo)
      .genSourceFile(sf, mlirModule);
}
