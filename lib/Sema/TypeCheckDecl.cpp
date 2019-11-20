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
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

//===- TypeChecker::DeclChecker -------------------------------------------===//

namespace {
/// Performs semantic analysis of a declaration.
/// Note that this doesn't check if a declaration is an illegal redeclaration -
/// that's handled by checkForRedeclaration which is usually called after this
/// class.
class DeclChecker : DeclVisitor<DeclChecker> {
  friend DeclVisitor<DeclChecker>;

public:
  TypeChecker &tc;
  SourceFile &file;

  DeclChecker(TypeChecker &tc, SourceFile &file) : tc(tc), file(file) {}

  void check(Decl *decl) { visit(decl); }

private:
  void visitVarDecl(VarDecl *decl) { tc.checkForRedeclaration(decl); }

  void visitParamDecl(ParamDecl *decl) {
    assert(decl->getValueType().isNull() && "Decl checked twice!");

    tc.checkForRedeclaration(decl);

    // Resolve the type of the ParamDecl
    tc.resolveTypeLoc(decl->getTypeLoc(), file);
  }

  void visitFuncDecl(FuncDecl *decl) {
    assert(decl->getValueType().isNull() && "Decl checked twice!");

    tc.checkForRedeclaration(decl);

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
  // Check the semantics of the declaration
  DeclChecker(*this, decl->getSourceFile()).check(decl);
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

void TypeChecker::checkForRedeclaration(ValueDecl *decl) {
  assert(decl && "decl is null!");
  UnqualifiedValueLookup uvl(decl->getSourceFile());

  // Limit lookup to the current block & file so shadowing rules are respected.
  uvl.options.onlyLookInCurrentBlock = true;
  uvl.options.onlyLookInCurrentFile = true;

  // This decl shouldn't part of the result set
  uvl.ignore(decl);

  // Perform the lookup
  uvl.performLookup(decl->getIdentifierLoc(), decl->getIdentifier());
  // Remove every result that come *after* us
  uvl.filterResults([&](ValueDecl *result) {
    return (result->getBegLoc() > decl->getBegLoc());
  });

  // If there are no results, this is a valid declaration
  if (uvl.isEmpty())
    return decl->setIsIllegalRedeclaration(false);
  decl->setIsIllegalRedeclaration(true);

  // Else, find the earliest result.
  ValueDecl *earliest = nullptr;
  for (ValueDecl *result : uvl.results)
    if (!earliest || (earliest->getBegLoc() > result->getBegLoc()))
      earliest = result;
  assert(earliest && "no earliest result found?");
  assert(earliest != decl);

  // Emit the diagnostic, pointing at the earliest result.
  diagnose(decl->getIdentifierLoc(), diag::value_already_declared_in_scope,
           decl->getIdentifier());
  diagnose(earliest->getIdentifierLoc(), diag::previously_declared_here,
           earliest->getIdentifier());
}