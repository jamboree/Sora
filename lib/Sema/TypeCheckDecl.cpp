//===--- TypeCheckDecl.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Declaration Semantic Analysis
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "Sora/AST/ASTScope.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/Types.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace sora;

//===- Utils --------------------------------------------------------------===//

namespace {

/// Checks that a declaration list doesn't bind an identifier more than once.
///
/// First, this checks that an identifier isn't bound more than once in
/// \p decls. Else, \p noteFirst is called for the first binding, and
/// \p diagnoseDuplicateBinding for each duplicate binding.
///
/// This also sets the 'isIllegalRedeclaration' flag on each duplicate
/// binding.
///
/// Finally, note that diagnoseDuplicateBinding/noteFirst are called in
/// groups. For example, in 'let (a, b, a, b)', they're both called for 'a'
/// first and then for 'b' (or for 'b' then for 'a' - that ordering isn't
/// guaranteed)
void checkForDuplicateBindingsInList(
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

//===- DeclChecker --------------------------------------------------------===//

/// Performs semantic analysis of a declaration.
/// Note that this doesn't check if a declaration is an illegal redeclaration -
/// that's handled by checkForRedeclaration which is usually called after this
/// class.
class DeclChecker : public ASTChecker, public DeclVisitor<DeclChecker> {
public:
  SourceFile &file;

  DeclChecker(TypeChecker &tc, SourceFile &file) : ASTChecker(tc), file(file) {}

  // An RAII object that marks a declaratin as being checked on destruction.
  class RAIIDeclChecking {
    Decl *const decl = nullptr;

  public:
    RAIIDeclChecking(Decl *decl) : decl(decl) {}
    ~RAIIDeclChecking() { decl->setChecked(); }
  };

  /// Checks if a ValueDecl is legal, and not an invalid redeclaration.
  /// This sets the decl's 'isIllegalRedeclaration' flag.
  /// This will not check the decl if it's already marked as being an illegal
  /// redeclaration.
  /// \verbatim
  ///   func foo() {} // this foo is valid, it's the first one
  ///   func foo() {} // this one isn't
  /// \endverbatim
  void checkIsIllegalRedeclaration(ValueDecl *decl) {
    assert(decl && "decl is null!");

    // If it's an illegal redeclaration, don't re-check.
    if (decl->isIllegalRedeclaration())
      return;

    UnqualifiedValueLookup uvl(decl->getSourceFile());

    // Since Sora has quite loose shadowing rules, we want lookup to stop after
    // looking into the first scope, unless we're checking a FuncDecl and we're
    // looking into its own scope.
    uvl.options.shouldStop = [&](const ASTScope *scope) {
      if (const FuncDeclScope *fnScope = dyn_cast<FuncDeclScope>(scope))
        return fnScope->getFuncDecl() != decl;
      return true;
    };

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

  void visitVarDecl(VarDecl *decl) {
    RAIIDeclChecking declChecking(decl);
    checkIsIllegalRedeclaration(decl);
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
    checkIsIllegalRedeclaration(decl);

    // Collect the parameters in an array of ValueDecl*s, as
    // tc.checkForDuplicateBindingsInList wants an ArrayRef<ValueDecl*>
    SmallVector<ValueDecl *, 4> paramBindings;
    ParamList *params = decl->getParamList();
    for (ParamDecl *param : *params)
      paramBindings.push_back(param);

    // Check that all parameter names are unique.
    checkForDuplicateBindingsInList(
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

  void visitLetDecl(LetDecl *decl, bool isCondition = false) {
    RAIIDeclChecking declChecking(decl);

    Pattern *pattern = decl->getPattern();
    assert(pattern && "LetDecl doesn't have a pattern");

    // When used as a condition, "let" implicitly look inside "maybe" types,
    // so wrap the LetDecl's pattern in an implicit MaybeValuePattern.
    if (isCondition) {
      Pattern *letPat = decl->getPattern();
      letPat = new (ctxt) MaybeValuePattern(letPat, /*isImplicit*/ true);
      decl->setPattern(letPat);
    }

    // Collect the variables declared inside the pattern
    SmallVector<ValueDecl *, 4> vars;
    decl->forEachVarDecl([&](VarDecl *var) { vars.push_back(var); });

    // Check if the pattern doesn't contain duplicate bindings
    checkForDuplicateBindingsInList(
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

    if (decl->hasInitializer()) {
      Expr *init = decl->getInitializer();

      auto diagnoseInitShouldHaveMaybeType = [&](Type initTy) {
        if (canDiagnose(initTy))
          diagnose(init->getLoc(), diag::cond_binding_must_have_maybe_type,
                   initTy);
      };

      // type-check the initializer & pattern together.
      bool complained = false;
      init = tc.typecheckPatternAndInitializer(
          decl->getPattern(), init, decl->getDeclContext(),
          [&](Type initTy, Type patTy) {
            complained = true;
            // If this is a "let" condition, we should simply complain that the
            // initializer should have a "maybe" type, so we don't confuse the
            // user too much (with the implicit MaybeValuePattern)
            if (isCondition)
              diagnoseInitShouldHaveMaybeType(initTy);
            // Else, just complain that we can't convert the type of the
            // initializer to the the pattern's type.
            else if (canDiagnose(initTy) && canDiagnose(patTy))
              diagnose(init->getLoc(), diag::cannot_convert_value_of_type,
                       initTy, patTy)
                  .highlight(pattern->getLoc())
                  .highlight(init->getSourceRange());
          });
      decl->setInitializer(init);

      // In conditions, also check that the initializer wasn't implicitly
      // converted to the "maybe" type, if that's the case, we're probably
      // facing something like "if let x = 0" which isn't allowed
      if (isCondition && !complained) {
        Expr *rawInit = init->ignoreImplicitConversions();
        Type rawInitTy = rawInit->getType()->getRValue();
        if (!rawInitTy->getDesugaredType()->is<MaybeType>())
          diagnoseInitShouldHaveMaybeType(rawInitTy);
      }
    }
    else {
      // If this is a "let" condition, we should have an initializer. Complain
      // about it!
      if (isCondition)
        diagnose(decl->getLetLoc(),
                 diag::variable_binding_in_cond_requires_initializer)
            .fixitInsertAfter(decl->getEndLoc(), "= <expression>")
            .highlight(decl->getLetLoc());
      tc.typecheckPattern(decl->getPattern(), decl->getDeclContext(),
                          /*canEmitInferenceErrors=*/!isCondition);
    }
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

static bool shouldCheck(Decl *decl) {
  assert(decl);
  return !decl->isChecked();
}

void TypeChecker::typecheckDecl(Decl *decl) {
  if (shouldCheck(decl))
    DeclChecker(*this, decl->getSourceFile()).visit(decl);
  assert(decl->isChecked());
}

void TypeChecker::typecheckLetCondition(LetDecl *decl) {
  if (shouldCheck(decl))
    DeclChecker(*this, decl->getSourceFile())
        .visitLetDecl(decl, /*isCondition*/ true);
  assert(decl->isChecked());
}

void TypeChecker::typecheckFunctionBody(FuncDecl *func) {
  if (func->isBodyChecked())
    return;
  typecheckStmt(func->getBody(), func);
  func->setBodyChecked();
}

void TypeChecker::typecheckDefinedFunctions() {
#ifndef NDEBUG
  size_t numDefinedFunc = definedFunctions.size();
#endif
  for (FuncDecl *func : definedFunctions)
    typecheckFunctionBody(func);
  assert((numDefinedFunc == definedFunctions.size()) &&
         "Extra functions were found while checking the bodies "
         "of defined non-local functions");
  definedFunctions.clear();
}