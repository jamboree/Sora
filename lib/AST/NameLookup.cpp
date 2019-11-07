//===--- NameLookup.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTScope.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"

using namespace sora;

//===- ASTScope Lookup Implementation -------------------------------------===//

namespace {
struct ASTScopeLookup {
  using LookupResultConsumer = ASTScope::LookupResultConsumer;

  ASTScopeLookup(LookupResultConsumer consumer, Identifier ident)
      : consumerFunc(consumer), ident(ident) {}

  LookupResultConsumer consumerFunc;
  Identifier ident;

  void visit(const ASTScope *scope) {
    bool stop;
    switch (scope->getKind()) {
#define SCOPE(KIND)                                                            \
  case ASTScopeKind::KIND:                                                     \
    stop = visit##KIND(static_cast<const KIND##Scope *>(scope));               \
    break;
#include "Sora/AST/ASTScopeKinds.def"
    }
    if (stop)
      return;
    if (ASTScope *parent = scope->getParent())
      visit(parent);
  }

  /// Calls the consumer if \p decls contains at least one decl.
  /// \returns the result of the consumer if it was called, false otherwise.
  bool consume(ArrayRef<ValueDecl *> decls, const ASTScope *scope) {
    return decls.empty() ? false : consumerFunc(decls, scope);
  }

  /// \returns true if \p decl should be considered as a result
  bool shouldConsider(ValueDecl *decl) {
    return considerEveryResult() ? true : decl->getIdentifier() == ident;
  }

  /// \returns true if we are looking for any decl, no matter the name.
  bool considerEveryResult() const { return !ident.isValid(); }

  //===- Visit Methods ----------------------------------------------------===//
  // Visit Methods visit a single scope, and return true when lookup can stop.
  //===--------------------------------------------------------------------===//

  bool visitSourceFile(const SourceFileScope *scope) {
    // Check the contents of the file

    // If we can consider every result, just feed everything to consume()
    SourceFile &sf = scope->getSourceFile();
    if (considerEveryResult())
      return consume(sf.getMembers(), scope);

    // If we can't consider everything, remove everything that can't be
    // considered.
    // FIXME: Ideally, the SourceFile should have a dedicated lookup cache &
    // lookup function.
    SmallVector<ValueDecl *, 4> decls;
    for (ValueDecl *decl : sf.getMembers()) {
      if (shouldConsider(decl))
        decls.push_back(decl);
    }
    return consume(decls, scope);
  }

  bool visitLocalLetDecl(const LocalLetDeclScope *scope) {
    // Collect every variable declared in the "let" and feed them to the
    // consumer.
    SmallVector<ValueDecl *, 4> decls;
    scope->getLetDecl()->forEachVarDecl([&](VarDecl *var) {
      if (shouldConsider(var))
        decls.push_back(var);
    });
    return consume(decls, scope);
  }

  bool visitFuncDecl(const FuncDeclScope *scope) {
    // Consider the params of the function
    // FIXME: Can this be made more efficient?
    FuncDecl *fn = scope->getFuncDecl();
    SmallVector<ValueDecl *, 4> decls;
    for (ParamDecl *param : *fn->getParamList()) {
      if (shouldConsider(param))
        decls.push_back(param);
    }
    return consume(decls, scope);
  }

  bool visitBlockStmt(const BlockStmtScope *scope) {
    // We only need to search for local func declarations.
    // FIXME: Can this be made more efficient? e.g. by automatically skipping
    // BlockStmts known to contain no FuncDecls.
    // Perhaps it'd be a good idea to add a "hasFuncDecl" flag to BlockStmt
    // (which would be extended to "hasFuncOrTypeDecl") later.
    SmallVector<ValueDecl *, 4> decls;
    for (ASTNode node : scope->getBlockStmt()->getElements()) {
      Decl *decl = node.dyn_cast<Decl *>();
      if (!decl)
        continue;
      FuncDecl *func = dyn_cast<FuncDecl>(decl);
      if (func && shouldConsider(func))
        decls.push_back(func);
    }
    return consume(decls, scope);
  }

  bool visitIfStmt(const IfStmtScope *scope) {
    // Nothing to do (if the cond declares something, it'll be handled by the
    // implicit LocalLetDeclScope)
    return false;
  }

  bool visitWhileStmt(const WhileStmtScope *scope) {
    // Nothing to do (if the cond declares something, it'll be handled by the
    // implicit LocalLetDeclScope)
    return false;
  }
};
} // namespace

//===- ASTScope::lookup ---------------------------------------------------===//

void ASTScope::lookup(ASTScope::LookupResultConsumer consumer,
                      Identifier ident) const {
  ASTScopeLookup(consumer, ident).visit(this);
}

//===- UnqualifiedValueLookup ---------------------------------------------===//

void UnqualifiedValueLookup::lookupImpl(SourceLoc loc, Identifier ident) {
  assert(loc && "SourceLoc can't be invalid!");
  assert(sourceFile.contains(loc) && "loc doesn't belong to this file!");
  assert(results.empty() &&
         "Consecutive lookups using the same object aren't supported");
  ASTScope *innermostScope = sourceFile.getScopeMap()->findInnermostScope(loc);
  assert(innermostScope && "findInnermostScope shouldn't return null!");
  auto consumer = [&](ArrayRef<ValueDecl *> candidates,
                      const ASTScope *scope) -> bool {
    // If we have at least one result, and we're not trying to find every
    // visible name, we can stop the lookup.
    // Else, keep looking.
    return addResults(candidates) && ident.isValid();
  };
  ASTScopeLookup(consumer, ident).visit(innermostScope);
}

//===- UnqualifiedTypeLookup ----------------------------------------------===//

void UnqualifiedTypeLookup::lookupImpl(SourceLoc loc, Identifier ident) {
  assert(loc && "SourceLoc can't be invalid!");
  assert(sourceFile.contains(loc) && "loc doesn't belong to this file!");
  assert(results.empty() &&
         "Consecutive lookups using the same object aren't supported");
  // For now, we don't have much work to do since we only have builtin types.
  // Just use ASTContext::lookupBuiltinType if we have an identifier, or use
  // getAllBuiltinTypes if we don't have one (= the caller wants every visible
  // type)
  ASTContext &ctxt = sourceFile.astContext;
  if (ident) {
    if (Type type = ctxt.lookupBuiltinType(ident))
      results.push_back(type);
  }
  else
    ctxt.getAllBuiltinTypes(results);
}