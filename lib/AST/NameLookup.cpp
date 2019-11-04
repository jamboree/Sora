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

ASTScope *ASTScope::createChild(ASTScope *parent, ASTNode node) {
  assert(parent && "Parent can't be null");
  ASTScope *result = new (parent->getASTContext()) ASTScope(parent, node);
  parent->addChild(result);
  return result;
}

ASTScope *ASTScope::createContinuation(ASTScope *parent, ASTNode node) {
  assert(parent && "Parent can't be null");
  ASTScope *result = new (parent->getASTContext()) ASTScope(parent, node);
  parent->addContinuation(result);
  return result;
}

ASTScope *ASTScope::createRoot(ASTContext &ctxt, ASTNode node) {
  return new (ctxt) ASTScope(ctxt, node);
}

void ASTScope::addChild(ASTScope *scope) {
  assert(!hasContinuation &&
         "Can't add another child if we have a continuation");
  children.push_back(scope);
  if (needsCleanup() && !cleanupRegistered) {
    getASTContext().addDestructorCleanup(*this);
    cleanupRegistered = true;
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

void ASTScope::dumpImpl(raw_ostream &out, unsigned indent, unsigned curIndent,
                        bool isContinuation) const {
  out.indent(curIndent);
  if (isContinuation)
    out << "(cont) ";
  out << "ASTScope<" << getASTNodeKind(node) << ">\n";
  for (ASTScope *child : children)
    child->dumpImpl(out, indent, curIndent + indent, /*isContinuation*/ false);
  if (hasContinuation())
    getContinuation()->dumpImpl(out, indent, curIndent + indent,
                                /*isContinuation*/ true);
}

//===- ASTScope ChildrenBuilder -------------------------------------------===//

/// The ASTScope ChildrenBuilder, which expands the children of an ASTScope.
/// This will create the children ASTScopes and continuations in some cases.
class ASTScope::ChildrenBuilder
    : public SimpleASTVisitor<ChildrenBuilder, void, ASTScope *> {
public:
  using Parent = SimpleASTVisitor<ChildrenBuilder, void, ASTScope *>;

  void visit(ASTScope *scope) {
    assert(!scope->childrenAreExpanded &&
           "Children have already been expanded");
    Parent::visit(scope->node, scope);
  }

#define VISIT_IMPOSSIBLE(T)                                                    \
  void visit##T(T *, ASTScope *) {                                             \
    llvm_unreachable(#T " doesn't have children scopes!");                     \
  }

  VISIT_IMPOSSIBLE(Expr)

  void visitFuncDecl(FuncDecl *decl, ASTScope *scope);
  VISIT_IMPOSSIBLE(ParamDecl)
  VISIT_IMPOSSIBLE(LetDecl)
  VISIT_IMPOSSIBLE(VarDecl)

  VISIT_IMPOSSIBLE(ContinueStmt)
  VISIT_IMPOSSIBLE(BreakStmt)
  VISIT_IMPOSSIBLE(ReturnStmt)

  void visitBlockStmt(BlockStmt *stmt, ASTScope *scope);
  void visitIfStmt(IfStmt *stmt, ASTScope *scope);
  void visitWhileStmt(WhileStmt *stmt, ASTScope *scope);

#undef VISIT_IMPOSSIBLE
};

/// For FuncDecls, we just create a BraceStmt children
void ASTScope::ChildrenBuilder::visitFuncDecl(FuncDecl *decl, ASTScope *scope) {
  assert(decl->getBody() && "no func body?");
  ASTScope::createChild(scope, decl->getBody());
}

void ASTScope::ChildrenBuilder::visitBlockStmt(BlockStmt *stmt,
                                               ASTScope *scope) {}

void ASTScope::ChildrenBuilder::visitIfStmt(IfStmt *stmt, ASTScope *scope) {

}

void ASTScope::ChildrenBuilder::visitWhileStmt(WhileStmt *stmt,
                                               ASTScope *scope) {
  // If the condition is a Decl, create a continuation
  if (LetDecl *cond = stmt->getCond().getLetDeclOrNull())
    scope = ASTScope::createContinuation(scope, stmt);
  // Create a scope for the body
  ASTScope::createChild(scope, stmt->getBody());
}

//===- ASTScope -----------------------------------------------------------===//

void ASTScope::expandChildren() { ChildrenBuilder().visit(this); }

// Plan:
//  -> Provide an entry point to create the AST Scope of a source file
//  -> Lazily expand children
//  -> Perhaps make the children vector mutable?