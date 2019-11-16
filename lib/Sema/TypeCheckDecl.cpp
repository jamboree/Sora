//===--- TypeCheckDecl.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Declaration Semantic Analysis
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

//===- TypeChecker::DeclChecker -------------------------------------------===//

namespace {
/// Performs semantic analysis of declarations in a file.
class DeclChecker : public DeclVisitor<DeclChecker> {
public:
  TypeChecker &tc;
  SourceFile &file;

  DeclChecker(TypeChecker &tc, SourceFile &file) : tc(tc), file(file) {}

  void visitVarDecl(VarDecl *decl) {
    // Nothing to do here.
    // VarDecls are checked through their VarPattern.
  }

  void visitParamDecl(ParamDecl *decl) {
    assert(decl->getValueType().isNull() && "Decl checked twice!");
    // Resolve the type of the ParamDecl
    tc.resolveTypeLoc(decl->getTypeLoc(), file);
  }

  void visitFuncDecl(FuncDecl *decl) {
    assert(decl->getValueType().isNull() && "Decl checked twice!");
    Type returnType;
    // Resolve the return type if present
    if (decl->hasReturnType()) {
      TypeLoc &tyLoc = decl->getReturnTypeLoc();
      tc.resolveTypeLoc(tyLoc, file);
      assert(tyLoc.hasType());
      returnType = tyLoc.getType();
    }
    // If the function doesn't have a return type, it returns void.
    else
      returnType = tc.ctxt.voidType;

    ParamList *params = decl->getParamList();
    SmallVector<Type, 8> paramTypes;
    for (ParamDecl *param : *params) {
      visit(param);
      Type paramType = param->getTypeLoc().getType();
      assert(paramType && "Function parameter doesn't have a type");
      paramTypes.push_back(paramType);
    }

    decl->setValueType(FunctionType::get(paramTypes, returnType));

    // Check the body directly for local functions, else delay it.
    if (decl->isLocal())
      tc.typecheckFunctionBody(decl);
    else
      tc.definedFunctions.push_back(decl);
  }

  void visitLetDecl(LetDecl *decl) {
    tc.typecheckPattern(decl->getPattern());

    if (decl->hasInitializer()) {
      assert(tc.ignoredDecls.empty() && "ignoredDecls vector should be empty!");
      decl->forEachVarDecl(
          [&](VarDecl *var) { tc.ignoredDecls.push_back(var); });
      decl->setInitializer(
          tc.typecheckExpr(decl->getInitializer(), decl->getDeclContext()));
      tc.ignoredDecls.clear();
    }
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::typecheckDecl(Decl *decl) {
  assert(decl);
  if (decl->isChecked())
    return;
  DeclChecker(*this, decl->getSourceFile()).visit(decl);
  decl->setChecked();
}

void TypeChecker::typecheckFunctionBody(FuncDecl *func) {
  if (func->isBodyChecked())
    return;
  typecheckStmt(func->getBody(), func);
  func->setBodyChecked();
}

void TypeChecker::typecheckDefinedFunctions() {
#ifndef NDEBUG
  unsigned numDefinedFunc = definedFunctions.size();
#endif
  for (FuncDecl *func : definedFunctions)
    typecheckFunctionBody(func);
  assert((numDefinedFunc == definedFunctions.size()) &&
         "Extra functions were found while checking the bodies "
         "of defined non-local functions");
  definedFunctions.clear();
}