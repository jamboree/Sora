//===--- ASTScope.cpp -------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTScope.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

//===- ASTScope -----------------------------------------------------------===//

void *ASTScope::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ArenaKind::Permanent);
}

void ASTScope::addChild(ASTScope *scope) {
  children.push_back(scope);
  if (!hasCleanup && needsCleanup()) {
    getASTContext().addDestructorCleanup(*this);
    hasCleanup = true;
  }
}

void ASTScope::fullyExpand() {
  for (ASTScope *child : children)
    child->fullyExpand();
}

SourceLoc ASTScope::getBegLoc() const {
  switch (getKind()) {
#define SCOPE(ID)                                                              \
  case ASTScopeKind::ID:                                                       \
    return ASTNodeLoc<ASTScope, ID##Scope, false>::getBegLoc(                  \
        cast<ID##Scope>(this));
#include "Sora/AST/ASTScopeKinds.def"
  }
  llvm_unreachable("unknown ASTScopeKind");
}

SourceLoc ASTScope::getEndLoc() const {
  switch (getKind()) {
#define SCOPE(ID)                                                              \
  case ASTScopeKind::ID:                                                       \
    return ASTNodeLoc<ASTScope, ID##Scope, false>::getEndLoc(                  \
        cast<ID##Scope>(this));
#include "Sora/AST/ASTScopeKinds.def"
  }
  llvm_unreachable("unknown ASTScopeKind");
}

SourceRange ASTScope::getSourceRange() const {
  switch (getKind()) {
#define SCOPE(ID)                                                              \
  case ASTScopeKind::ID:                                                       \
    return ASTNodeLoc<ASTScope, ID##Scope, false>::getSourceRange(             \
        cast<ID##Scope>(this));
#include "Sora/AST/ASTScopeKinds.def"
  }
  llvm_unreachable("unknown ASTScopeKind");
}

ASTContext &ASTScope::getASTContext() const {
  if (const SourceFileScope *scope = dyn_cast<const SourceFileScope>(this))
    return scope->getSourceFile().astContext;
  ASTScope *parent = getParent();
  assert(parent && "ASTScope should always have a parent");
  return parent->getASTContext();
}

void ASTScope::expand() { /* todo*/
}

static const char *getKindStr(ASTScopeKind kind) {
  switch (kind) {
#define SCOPE(KIND)                                                            \
  case ASTScopeKind::KIND:                                                     \
    return #KIND;
#include "Sora/AST/ASTScopeKinds.def"
  }
  llvm_unreachable("unknown ASTScope kind");
}

void ASTScope::dumpImpl(raw_ostream &out, unsigned indent,
                        unsigned curIndent) const {
  out.indent(curIndent);
  const SourceManager &srcMgr = getASTContext().srcMgr;
  // Add a "Scope" suffix to the kind string, and print the range as well.
  out << getKindStr(getKind()) << "Scope range:";
  // Only print the source file for SourceFileScopes.
  getSourceRange().print(out, srcMgr, isa<SourceFileScope>(this));
  out << '\n';
}

SourceFileScope *SourceFileScope::create(SourceFile &sf) {
  return new (sf.astContext) SourceFileScope(sf);
}

SourceLoc SourceFileScope::getBegLoc() const {
  // FIXME: SourceFile should have getBegLoc()
  return sourceFile.getMembers().front()->getBegLoc();
}

SourceLoc SourceFileScope::getEndLoc() const {
  // FIXME: SourceFile should have getEndLoc()
  return sourceFile.getMembers().front()->getEndLoc();
}

FuncDeclScope *FuncDeclScope::create(FuncDecl *func, ASTScope *parent) {
  return new (func->getASTContext()) FuncDeclScope(func, parent);
}

SourceLoc FuncDeclScope::getBegLoc() const { return decl->getBegLoc(); }

SourceLoc FuncDeclScope::getEndLoc() const { return decl->getEndLoc(); }

bool LocalLetDeclScope::isLocalAndNonNull() const {
  return decl && decl->isLocal();
}

LocalLetDeclScope *LocalLetDeclScope::create(LetDecl *decl, ASTScope *parent,
                                             SourceLoc end) {
  return new (decl->getASTContext()) LocalLetDeclScope(decl, parent, end);
}

SourceLoc LocalLetDeclScope::getBegLoc() const {
  return getLetDecl()->getBegLoc();
}

SourceLoc LocalLetDeclScope::getEndLoc() const { return end; }

BlockStmtScope *BlockStmtScope::create(ASTContext &ctxt, BlockStmt *stmt,
                                       ASTScope *parent) {
  return new (ctxt) BlockStmtScope(stmt, parent);
}

SourceLoc BlockStmtScope::getBegLoc() const {
  return getBlockStmt()->getBegLoc();
}

SourceLoc BlockStmtScope::getEndLoc() const {
  return getBlockStmt()->getEndLoc();
}

IfStmtScope *IfStmtScope::create(ASTContext &ctxt, IfStmt *stmt,
                                 ASTScope *parent) {
  return new (ctxt) IfStmtScope(stmt, parent);
}

SourceLoc IfStmtScope::getBegLoc() const { return getIfStmt()->getBegLoc(); }

SourceLoc IfStmtScope::getEndLoc() const { return getIfStmt()->getEndLoc(); }

WhileStmtScope *WhileStmtScope::create(ASTContext &ctxt, WhileStmt *stmt,
                                       ASTScope *parent) {
  return new (ctxt) WhileStmtScope(stmt, parent);
}

SourceLoc WhileStmtScope::getBegLoc() const {
  return getWhileStmt()->getBegLoc();
}

SourceLoc WhileStmtScope::getEndLoc() const {
  return getWhileStmt()->getEndLoc();
}