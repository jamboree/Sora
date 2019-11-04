//===--- ASTScope.cpp -------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTScope.hpp"
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
  case ASTScopeKind::SourceFile:
    return "SourceFileScope";
  case ASTScopeKind::FuncDecl:
    return "FuncDeclScope";
  case ASTScopeKind::LocalLetDecl:
    return "LocalLetDeclScope";
  case ASTScopeKind::BlockStmt:
    return "BlockStmtScope";
  case ASTScopeKind::IfStmt:
    return "IfStmtScope";
  case ASTScopeKind::WhileStmt:
    return "WhileStmtScope";
  }
  llvm_unreachable("unknown ASTScope kind");
}

void ASTScope::dumpImpl(raw_ostream &out, unsigned indent,
                        unsigned curIndent) const {
  out.indent(curIndent);
  out << getKindStr(getKind()) << "\n";
}

SourceFileScope *SourceFileScope::create(SourceFile &sf) {
  return new (sf.astContext) SourceFileScope(sf);
}

FuncDeclScope *FuncDeclScope::create(FuncDecl *func, ASTScope *parent) {
  return new (func->getASTContext()) FuncDeclScope(func, parent);
}

bool LocalLetDeclScope::isLocalAndNonNull() const {
  return decl && decl->isLocal();
}

LocalLetDeclScope *LocalLetDeclScope::create(LetDecl *decl, ASTScope *parent) {
  return new (decl->getASTContext()) LocalLetDeclScope(decl, parent);
}

BlockStmtScope *BlockStmtScope::create(ASTContext &ctxt, BlockStmt *stmt,
                                       ASTScope *parent) {
  return new (ctxt) BlockStmtScope(stmt, parent);
}

IfStmtScope *IfStmtScope::create(ASTContext &ctxt, IfStmt *stmt,
                                 ASTScope *parent) {
  return new (ctxt) IfStmtScope(stmt, parent);
}

WhileStmtScope *WhileStmtScope::create(ASTContext &ctxt, WhileStmt *stmt,
                                       ASTScope *parent) {
  return new (ctxt) WhileStmtScope(stmt, parent);
}