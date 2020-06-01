//===--- IRGen.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/EntryPoints.hpp"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"

using namespace sora;

//===- IRGen --------------------------------------------------------------===//

static ir::SoraDialect &getSoraDialect(mlir::MLIRContext &mlirCtxt) {
  auto *dialect = mlirCtxt.getRegisteredDialect<ir::SoraDialect>();
  assert(dialect && "SoraDialect not registered! Did you call "
                    "'sora::registerMLIRDialects()'?");
  return *dialect;
}

IRGen::IRGen(ASTContext &astCtxt, mlir::MLIRContext &mlirCtxt,
             bool enableDebugInfo)
    : debugInfoEnabled(enableDebugInfo), astCtxt(astCtxt),
      srcMgr(astCtxt.srcMgr), mlirCtxt(mlirCtxt),
      soraDialect(getSoraDialect(mlirCtxt)) {}

void IRGen::genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule) {
  for (ValueDecl *decl : sf.getMembers())
    mlirModule.push_back(genFunction(cast<FuncDecl>(decl)));
}

mlir::Location IRGen::getFileLineColLoc(SourceLoc loc) {
  if (!debugInfoEnabled)
    return mlir::UnknownLoc::get(&mlirCtxt);

  BufferID buffer = srcMgr.findBufferContainingLoc(loc);
  StringRef filename = srcMgr.getBufferName(buffer);
  std::pair<unsigned, unsigned> lineAndCol = srcMgr.getLineAndColumn(loc);
  return mlir::FileLineColLoc::get(filename, lineAndCol.first,
                                   lineAndCol.second, &mlirCtxt);
}

mlir::Identifier IRGen::getIRIdentifier(StringRef str) {
  return mlir::Identifier::get(str, &mlirCtxt);
}

//===- Entry Points -------------------------------------------------------===//

#ifndef NDEBUG
static bool alreadyRegisteredMLIRDialects = false;
#endif

void sora::registerMLIRDialects() {
  // Registering dialects multiple times shouldn't be an issue, but don't
  // encourage it.
  assert(!alreadyRegisteredMLIRDialects && "Registering dialects again!");

  mlir::registerDialect<ir::SoraDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();

#ifndef NDEBUG
  alreadyRegisteredMLIRDialects = true;
#endif
}

mlir::ModuleOp sora::createMLIRModule(mlir::MLIRContext &mlirCtxt,
                                      SourceFile &sf) {
  auto bufferID = mlir::Identifier::get(sf.getBufferName(), &mlirCtxt);
  auto nameLoc = mlir::NameLoc::get(bufferID, &mlirCtxt);
  return mlir::ModuleOp::create(nameLoc, sf.getBufferName());
}

void sora::performIRGen(mlir::MLIRContext &mlirCtxt, mlir::ModuleOp &mlirModule,
                        SourceFile &sf, bool enableDebugInfo) {
  IRGen irGen(sf.astContext, mlirCtxt, enableDebugInfo);
  irGen.genSourceFile(sf, mlirModule);
}
