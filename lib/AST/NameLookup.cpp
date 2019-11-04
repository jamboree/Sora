//===--- NameLookup.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Stmt.hpp"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

//===- ASTScope -----------------------------------------------------------===//

void *ASTScope::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ArenaKind::Permanent);
}

ASTScope *ASTScope::create(ASTContext &ctxt, ASTScope *parent, ASTNode node) {
  return new (ctxt) ASTScope(parent, node);
}

ASTScope *ASTScope::createRoot(ASTContext &ctxt, ASTNode node) {
  return new (ctxt) ASTScope(ctxt, node);
}

void ASTScope::addChild(ASTScope *scope) {
  children.push_back(scope);
  if (needsCleanup() && !hasRegisteredCleanup) {
    getASTContext().addDestructorCleanup(*this);
    hasRegisteredCleanup = true;
  }
}

void ASTScope::fullyExpand() {
  expandChildren();
  for (ASTScope *child : children)
    child->fullyExpand();
}

bool ASTScope::isFuncDecl() const {
  if (Decl *decl = node.dyn_cast<Decl *>())
    return isa<FuncDecl>(decl);
  return false;
}

namespace {
const char *getASTNodeKind(ASTNode node) {
  if (Expr *expr = node.dyn_cast<Expr *>()) {
    switch (expr->getKind()) {
#define EXPR(KIND, PARENT)                                                     \
  case ExprKind::KIND:                                                         \
    return #KIND "Expr";
#include "Sora/AST/ExprNodes.def"
    }
    llvm_unreachable("unknown Expr");
  }
  if (Stmt *stmt = node.dyn_cast<Stmt *>()) {
    switch (stmt->getKind()) {
#define STMT(KIND, PARENT)                                                     \
  case StmtKind::KIND:                                                         \
    return #KIND "Stmt";
#include "Sora/AST/StmtNodes.def"
    }
    llvm_unreachable("unknown Stmt");
  }
  if (Decl *decl = node.dyn_cast<Decl *>()) {
    switch (decl->getKind()) {
#define DECL(KIND, PARENT)                                                     \
  case DeclKind::KIND:                                                         \
    return #KIND "decl";
#include "Sora/AST/DeclNodes.def"
    }
    llvm_unreachable("unknown Decl");
  }
  llvm_unreachable("unknown ASTNode");
}
} // namespace

void ASTScope::dumpImpl(raw_ostream &out, unsigned indent,
                        unsigned curIndent) const {
  out << "ASTScope<" << getASTNodeKind(node) << ">\n";
  for (ASTScope *child : children)
    child->dumpImpl(out, indent, curIndent + indent);
}

//===- ASTScope Builder ---------------------------------------------------===//

/// The AST Scope builder, which is tasked with building the AST Scopes of an
/// AST.
class ASTScope::Builder : public SimpleASTVisitor<Builder> {};

//===- ASTScope -----------------------------------------------------------===//

void ASTScope::expandChildren() {
  // TODO
}

// Plan:
//  -> Provide an entry point to create the AST Scope of a source file
//  -> Lazily expand children
//  -> Perhaps make the children vector mutable?