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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

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
    Pattern *pattern = decl->getPattern();
    assert(pattern && "LetDecl doesn't have a pattern");

    // Check that this pattern doesn't bind the variables inside it more than
    // once.
    checkPatternBindings(pattern);

    // Type-check the pattern
    tc.typecheckPattern(pattern);

    if (decl->hasInitializer()) {
      assert(tc.ignoredDecls.empty() && "ignoredDecls vector should be empty!");
      decl->forEachVarDecl(
          [&](VarDecl *var) { tc.ignoredDecls.push_back(var); });
      decl->setInitializer(
          tc.typecheckExpr(decl->getInitializer(), decl->getDeclContext()));
      tc.ignoredDecls.clear();
    }
  }

  /// Checks that the same identifier isn't bound more than once in \p decls.
  /// FIXME: This function isn't really efficient, but since it's only ran once
  /// per LetDecl and it usually works with 1 to 5 vars, it should be ok.
  void checkPatternBindings(Pattern *pat) {
    // The map of identifiers -> first VarDecl* that binds it
    llvm::DenseMap<Identifier, VarDecl *> boundIdentifiers;
    // The map of identifiers -> bad (duplicate) var decls
    // FIXME: Is a DenseMap of SmallPtrSet efficient?
    llvm::DenseMap<Identifier, llvm::SmallPtrSet<VarDecl *, 2>> badVarDecls;

    // Adds a VarDecl to \c badVarDecls, asserting that it's wasn't added to the
    // set before.
    auto addBadVarDecl = [&](VarDecl *var) {
      auto result = badVarDecls[var->getIdentifier()].insert(var);
      assert(result.second && "Var already known bad!");
    };

    // Removes a VarDecl from \c badVarDecls.
    auto removeBadVarDecl = [&](VarDecl *var) {
      badVarDecls[var->getIdentifier()].erase(var);
    };

    // Iterate over the the VarDecls in the Pattern
    pat->forEachVarDecl([&](VarDecl *var) {
      VarDecl *&firstBound = boundIdentifiers[var->getIdentifier()];
      // If it's the first time this identifier is bound in this pattern, just
      // store it.
      if (!firstBound) {
        firstBound = var;
        return;
      }
      // Else, store it in the erroneous VarDecls set.
      // If var comes after the VarDecl that first binds the identifier,
      // just put it in the set.
      // This is the most likely scenario as forEachVarDecl should be iterating
      // the VarDecls in order of appearance in a well-formed AST.
      if (var->getBegLoc() > firstBound->getBegLoc())
        addBadVarDecl(var);
      // Else, var must become the firstBound var, and add firstBound to the
      // set of bad VarDecls instead.
      else {
        addBadVarDecl(firstBound);
        removeBadVarDecl(var);
        firstBound = var;
      }
    });

    // Now, if we must emit any diagnostics, do so.
    if (badVarDecls.empty())
      return;

    for (auto entry : badVarDecls) {
      Identifier ident = entry.first;
      // Diagnose each variable
      for (VarDecl *badVar : entry.second) {
        tc.diagnose(badVar->getIdentifierLoc(),
                    diag::identifier_bound_multiple_times_in_pat, ident);
      }
      // And finally note the first variable declared.
      VarDecl *first = boundIdentifiers[ident];
      tc.diagnose(first->getIdentifierLoc(), diag::identifier_first_bound_here,
                  ident);
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

  // If it's an illegal redeclaration, don't re-check.
  if (decl->isIllegalRedeclaration())
    return;

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
  decl->setIsIllegalRedeclaration();

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