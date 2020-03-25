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

static mlir::LLVM::LLVMDialect &getLLVMDialect(mlir::MLIRContext &mlirCtxt) {
  auto *dialect = mlirCtxt.getRegisteredDialect<mlir::LLVM::LLVMDialect>();
  assert(dialect && "LLVMDialect not registered! Did you call "
                    "'sora::registerMLIRDialects()'?");
  return *dialect;
}

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
      llvmDialect(getLLVMDialect(mlirCtxt)), soraDialect(soraDialect) {}

void IRGen::genSourceFile(SourceFile &sf, mlir::ModuleOp &mlirModule) {
  for (ValueDecl *decl : sf.getMembers())
    mlirModule.push_back(genFunction(cast<FuncDecl>(decl)));
}

mlir::Identifier IRGen::getMLIRIdentifier(StringRef str) {
  return mlir::Identifier::get(str, &mlirCtxt);
}

//===- Entry Points -------------------------------------------------------===//

static bool areMLIRDialectsRegistered = false;

void sora::registerMLIRDialects() {
  if (!areMLIRDialectsRegistered) {
    mlir::registerDialect<mlir::LLVM::LLVMDialect>();
    mlir::registerDialect<ir::SoraDialect>();
    areMLIRDialectsRegistered = true;
  }
}

mlir::ModuleOp sora::createMLIRModule(mlir::MLIRContext &mlirCtxt,
                                      SourceFile &sf) {
  // FIXME: The Module's name and NameLoc should really be different.
  //    - The name should be shorter, probably something like the file name's
  //    minus the project's root.
  //    - The NameLoc should still be the path.
  //  The question is how to achieve this. This may require some changes to the
  //  SourceManager so it can remember the path of files (in a separate map or
  //  something, then the buffer identifier can stay like this, and it'll be the
  //  responsability of the Driver to assign a clear name to the file)
  auto bufferID = mlir::Identifier::get(sf.getBufferIdentifier(), &mlirCtxt);
  auto nameLoc = mlir::NameLoc::get(bufferID, &mlirCtxt);
  return mlir::ModuleOp::create(nameLoc, sf.getBufferIdentifier());
}

void sora::performIRGen(mlir::MLIRContext &mlirCtxt, mlir::ModuleOp &mlirModule,
                        SourceFile &sf, bool enableDebugInfo) {
  IRGen irGen(sf.astContext, mlirCtxt, enableDebugInfo);
  irGen.genSourceFile(sf, mlirModule);
}
