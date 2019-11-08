//===--- SemaDecl.cpp -------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Declaration Semantic Analysis
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

//===- DeclChecker --------------------------------------------------------===//

/// Performs semantic analysis of declarations in a file.
class Sema::DeclChecker : public DeclVisitor<DeclChecker> {
public:
  Sema &sema;

  DeclChecker(Sema &sema) : sema(sema) {}

  void visitVarDecl(VarDecl *decl) {
    // Nothing to do here.
    // VarDecls are checked through their VarPattern.
  }

  void visitParamDecl(ParamDecl *decl) {
    // Resolve the type of the ParamDecl
    sema.resolveTypeLoc(decl->getTypeLoc());
  }

  void visitFuncDecl(FuncDecl *decl) {
    Type returnType;
    // Resolve the return type if present
    if (decl->hasReturnType()) {
      TypeLoc &tyLoc = decl->getReturnTypeLoc();
      sema.resolveTypeLoc(tyLoc);
      assert(tyLoc.hasType());
      returnType = tyLoc.getType();
    }
    // If the function doesn't have a return type, it returns void.
    else
      returnType = sema.ctxt.voidType;

    ParamList *params = decl->getParamList();
    SmallVector<Type, 8> paramTypes;
    for (ParamDecl *param : *params) {
      visit(param);
      Type paramType = param->getTypeLoc().getType();
      assert(paramType && "Function parameter doesn't have a type");
      paramTypes.push_back(paramType);
    }

    decl->setValueType(FunctionType::get(paramTypes, returnType));

    if (decl->hasBody())
      sema.definedFunctions.push_back(decl);
  }

  void visitLetDecl(LetDecl *decl) {
    sema.typecheckPattern(decl->getPattern());

    if (decl->hasInitializer()) {
      assert(sema.ignoredDecls.empty() &&
             "ignoredDecls vector should be empty!");
      decl->forEachVarDecl(
          [&](VarDecl *var) { sema.ignoredDecls.push_back(var); });
      decl->setInitializer(
          sema.typecheckExpr(decl->getInitializer(), decl->getDeclContext()));
      sema.ignoredDecls.clear();
    }
  }
};

//===- Sema ---------------------------------------------------------------===//

void Sema::typecheckDecl(Decl *decl) {
  assert(decl);
  DeclChecker(*this).visit(decl);
}

void Sema::typecheckFunctionBodies() {
  for (FuncDecl *func : definedFunctions)
    typecheckStmt(func->getBody(), func);

  definedFunctions.clear();
}