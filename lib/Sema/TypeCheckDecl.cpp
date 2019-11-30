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
#include "llvm/ADT/SmallVector.h"

using namespace sora;

//===- TypeChecker::DeclChecker -------------------------------------------===//

namespace {
/// Performs semantic analysis of a declaration.
/// Note that this doesn't check if a declaration is an illegal redeclaration -
/// that's handled by checkForRedeclaration which is usually called after this
/// class.
class DeclChecker : public ASTChecker, DeclVisitor<DeclChecker> {
  friend DeclVisitor<DeclChecker>;

public:
  SourceFile &file;

  DeclChecker(TypeChecker &tc, SourceFile &file) : ASTChecker(tc), file(file) {}

  void check(Decl *decl) { visit(decl); }

private:
  // An RAII object that marks a declaratin as being checked on destruction.
  class RAIIDeclChecking {
    Decl *const decl = nullptr;

  public:
    RAIIDeclChecking(Decl *decl) : decl(decl) {}
    ~RAIIDeclChecking() { decl->setChecked(); }
  };

  void visitVarDecl(VarDecl *decl) {
    RAIIDeclChecking declChecking(decl);
    tc.checkIsIllegalRedeclaration(decl);
  }

  // Called by visitFuncDecl
  //
  // Only resolves the ParamDecl's type.
  void visitParamDecl(ParamDecl *decl) {
    RAIIDeclChecking declChecking(decl);
    // Only resolve the type.
    // We don't need to call tc.checkIsIllegalRedeclaration since
    // ParamDecls can shadow anything. Duplicate parameter names in the same
    // parameter list are handled by visitFuncDecl.
    tc.resolveTypeLoc(decl->getTypeLoc(), file);
  }

  void visitFuncDecl(FuncDecl *decl) {
    RAIIDeclChecking declChecking(decl);

    // Check if this function isn't an illegal redeclaration
    tc.checkIsIllegalRedeclaration(decl);

    // Collect the parameters in an array of ValueDecl*s, as
    // tc.checkForDuplicateBindingsInList wants an ArrayRef<ValueDecl*>
    SmallVector<ValueDecl *, 4> paramBindings;
    ParamList *params = decl->getParamList();
    for (ParamDecl *param : *params)
      paramBindings.push_back(param);

    // Check that all parameter names are unique.
    tc.checkForDuplicateBindingsInList(
        paramBindings,
        // diagnoseDuplicateBinding
        [&](ValueDecl *decl) {
          diagnose(decl->getIdentifierLoc(),
                   diag::identifier_bound_multiple_times_in_same_paramlist,
                   decl->getIdentifier());
        },
        // noteFirstBinding
        [&](ValueDecl *decl) {
          diagnose(decl->getIdentifierLoc(), diag::identifier_bound_first_here,
                   decl->getIdentifier());
        });

    // Resolve the return type of the function
    Type returnType;
    if (decl->hasReturnType()) {
      // If there's an explicit return type, resolve it.
      TypeLoc &tyLoc = decl->getReturnTypeLoc();
      tc.resolveTypeLoc(tyLoc, file);
      assert(tyLoc.hasType());
      returnType = tyLoc.getType();
    }
    else
      // If there's no explicit return type, the return type is void.
      returnType = tc.ctxt.voidType;

    // Visit the parameters and collect their type.
    SmallVector<Type, 8> paramTypes;
    for (ParamDecl *param : *params) {
      visit(param);
      Type paramType = param->getTypeLoc().getType();
      assert(paramType && "Function parameter doesn't have a type");
      paramTypes.push_back(paramType);
    }

    // Compute the function type and assign it to the function
    decl->setValueType(FunctionType::get(paramTypes, returnType));

    // Check the body directly for local functions, else delay it.
    if (decl->isLocal())
      tc.typecheckFunctionBody(decl);
    else
      tc.definedFunctions.push_back(decl);
  }

  void visitLetDecl(LetDecl *decl) {
    RAIIDeclChecking declChecking(decl);

    Pattern *pattern = decl->getPattern();
    assert(pattern && "LetDecl doesn't have a pattern");

    // Collect the variables declared inside the pattern
    SmallVector<ValueDecl *, 4> vars;
    decl->forEachVarDecl([&](VarDecl *var) { vars.push_back(var); });

    // Check if the pattern doesn't contain duplicate bindings
    tc.checkForDuplicateBindingsInList(
        vars,
        // diagnoseDuplicateBinding
        [&](ValueDecl *decl) {
          diagnose(decl->getIdentifierLoc(),
                   diag::identifier_bound_multiple_times_in_same_pat,
                   decl->getIdentifier());
        },
        // noteFirstBinding
        [&](ValueDecl *decl) {
          diagnose(decl->getIdentifierLoc(), diag::identifier_bound_first_here,
                   decl->getIdentifier());
        });

    // Type-check the pattern.
    tc.typecheckPattern(pattern);

    // Type-check the initializer
    if (decl->hasInitializer()) {
      Expr *init = decl->getInitializer();
      init = tc.typecheckExpr(init, decl->getDeclContext());
      decl->setInitializer(init);
    }
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::typecheckDecl(Decl *decl) {
  assert(decl);
  if (decl->isChecked())
    return;
  DeclChecker(*this, decl->getSourceFile()).check(decl);
  assert(decl->isChecked() &&
         "Decl isn't marked as checked after being checked");
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

void TypeChecker::checkIsIllegalRedeclaration(ValueDecl *decl) {
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

  // Emit the diagnostics
  diagnose(decl->getIdentifierLoc(), diag::value_already_defined_in_scope,
           decl->getIdentifier());
  diagnose(earliest->getIdentifierLoc(), diag::previous_def_is_here,
           earliest->getIdentifier());
}

void TypeChecker::checkForDuplicateBindingsInList(
    ArrayRef<ValueDecl *> decls,
    llvm::function_ref<void(ValueDecl *)> diagnoseDuplicateBinding,
    llvm::function_ref<void(ValueDecl *)> noteFirstBinding) {
  // The map of identifiers -> list of bindings.
  llvm::DenseMap<Identifier, SmallVector<ValueDecl *, 2>> bindingsMap;

  // Collect the bindings
  for (ValueDecl *decl : decls)
    bindingsMap[decl->getIdentifier()].push_back(decl);

  // Check them
  for (auto entry : bindingsMap) {
    SmallVectorImpl<ValueDecl *> &bindings = entry.second;
    assert(!bindings.empty() && "Empty set of bindings?");

    // If we got more than one binding for this identifier in the list, we must
    // emit diagnostics.
    if (bindings.size() > 1) {
      // First, sort the vector
      std::sort(bindings.begin(), bindings.end(),
                [&](ValueDecl *lhs, ValueDecl *rhs) {
                  return lhs->getBegLoc() < rhs->getBegLoc();
                });

      // Call diagnoseDuplicateBinding on each decl except the first one.
      for (auto it = bindings.begin() + 1, end = bindings.end(); it != end;
           ++it) {
        diagnoseDuplicateBinding(*it);
        (*it)->setIsIllegalRedeclaration();
      }

      // Call noteFirstBinding on the first one
      noteFirstBinding(bindings.front());
    }
  }
}