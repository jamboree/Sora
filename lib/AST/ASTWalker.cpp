//===--- ASTWalker.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/TypeRepr.hpp"

using namespace sora;

namespace {
/// The ASTWalker's actual implementation
struct Traversal : public SimpleASTVisitor<Traversal> {
  using Action = ASTWalker::Action;

  Traversal(ASTWalker &walker) : walker(walker) {}

  /// The ASTWalker
  ASTWalker &walker;

  /// Whether the walk was stopped.
  bool stopped = false;

  /// Performs the walk and returns true if it was successful, false if it ended
  /// prematurely.
  template <typename Node> bool walk(Node node) {
    doIt(node);
    return !stopped;
  }

  //===- Visit Methods ----------------------------------------------------===//
  // - For each children of the node we're visiting
  //  - Call doIt
  //  - (For Exprs) If the return value of doIt is non-nullptr, replace the
  //    child with the return value.
  //
  // visit methods should never assume that the AST is well formed.
  //===--------------------------------------------------------------------===//

#define TRIVIAL_VISIT(TYPE)                                                    \
  void visit##TYPE(TYPE *) {}

  //===- Decl -------------------------------------------------------------===//

  void visitVarDecl(VarDecl *decl) { doIt(decl->getTypeLoc()); }

  void visitParamDecl(ParamDecl *decl) { doIt(decl->getTypeLoc()); }

  void visitFuncDecl(FuncDecl *decl) {
    doIt(decl->getParamList());
    doIt(decl->getReturnTypeLoc());
    doIt(decl->getBody());
  }

  void visitPatternBindingDecl(PatternBindingDecl *decl) {
    doIt(decl->getPattern());
    if (Expr *expr = doIt(decl->getInitializer()))
      decl->setInitializer(expr);
  }

  //===- Expr -------------------------------------------------------------===//

  TRIVIAL_VISIT(UnresolvedDeclRefExpr)
  TRIVIAL_VISIT(DiscardExpr)
  TRIVIAL_VISIT(AnyLiteralExpr)
  TRIVIAL_VISIT(ErrorExpr)

  void visitTupleIndexingExpr(TupleIndexingExpr *expr) {
    if (Expr *base = doIt(expr->getBase()))
      if (Expr *base = doIt(expr->getBase()))
        expr->setBase(base);
    if (auto idx = dyn_cast_or_null<IntegerLiteralExpr>(doIt(expr->getIndex())))
      expr->setIndex(idx);
  }

  void visitTupleExpr(TupleExpr *expr) {
    for (unsigned k = 0; k < expr->getNumElements(); ++k)
      if (Expr *elem = doIt(expr->getElement(k)))
        expr->setElement(k, elem);
  }

  void visitParenExpr(ParenExpr *expr) {
    if (Expr *sub = doIt(expr->getSubExpr()))
      expr->setSubExpr(sub);
  }

  void visitCallExpr(CallExpr *expr) {
    if (Expr *fn = doIt(expr->getFn()))
      expr->setFn(fn);
    if (auto *args = dyn_cast_or_null<TupleExpr>(doIt(expr->getArgs())))
      expr->setArgs(args);
  }

  void visitBinaryExpr(BinaryExpr *expr) {
    if (Expr *lhs = doIt(expr->getLHS()))
      expr->setLHS(expr);
    if (Expr *rhs = doIt(expr->getRHS()))
      expr->setRHS(rhs);
  }

  void visitUnaryExpr(UnaryExpr *expr) {
    if (Expr *sub = doIt(expr->getSubExpr()))
      expr->setSubExpr(sub);
  }

  //===- Pattern ----------------------------------------------------------===//

  void visitVarPattern(VarPattern *pattern) { doIt(pattern->getVarDecl()); }

  TRIVIAL_VISIT(DiscardPattern)

  void visitMutPattern(MutPattern *pattern) { doIt(pattern->getSubPattern()); }

  void visitTuplePattern(TuplePattern *pattern) {
    for (Pattern *elem : pattern->getElements())
      doIt(elem);
  }

  void visitTypedPattern(TypedPattern *pattern) {
    doIt(pattern->getSubPattern());
    doIt(pattern->getTypeRepr());
  }

  //===- ParamList --------------------------------------------------------===//

  void visitParamList(ParamList *paramList) {
    for (ParamDecl *elem : paramList->getParams())
      doIt(elem);
  }

  //===- TypeRepr ---------------------------------------------------------===//

  TRIVIAL_VISIT(IdentifierTypeRepr)

  void visitTupleTypeRepr(TupleTypeRepr *tyRepr) {
    for (TypeRepr *elem : tyRepr->getElements())
      doIt(elem);
  }

  void visitPointerTypeRepr(PointerTypeRepr *tyRepr) {
    doIt(tyRepr->getSubTypeRepr());
  }

  //===- Stmt -------------------------------------------------------------===//

  TRIVIAL_VISIT(ContinueStmt)

  TRIVIAL_VISIT(BreakStmt)

  void visitReturnStmt(ReturnStmt *stmt) {
    if (Expr *expr = doIt(stmt->getResult()))
      stmt->setResult(expr);
  }

  void visitBlockStmt(BlockStmt *stmt) {
    for (ASTNode &node : stmt->getElements())
      doIt(node);
  }

  void visitIfStmt(IfStmt *stmt) {
    doIt(stmt->getCond());
    doIt(stmt->getThen());
    doIt(stmt->getElse());
  }

  void visitWhileStmt(WhileStmt *stmt) {
    doIt(stmt->getCond());
    doIt(stmt->getBody());
  }

#undef TRIVIAL_VISIT

  //===- DoIt Methods -----------------------------------------------------===//
  // - Call walkToXPre
  // - Handle the result (and possibly return)
  // - If needed, call visit(X) to visit the children
  // - Call walkToXPost
  // - Handle the result
  // Note: doIt method should accept nullptr arguments (just return directly)
  //===--------------------------------------------------------------------===//

  void doIt(ASTNode &node) {
    if (stopped)
      return;
    if (auto expr = node.dyn_cast<Expr *>()) {
      if (Expr *replacement = doIt(expr))
        node = replacement;
    }
    else if (auto stmt = node.dyn_cast<Stmt *>())
      doIt(stmt);
    else if (auto decl = node.dyn_cast<Decl *>())
      doIt(decl);
    else
      llvm_unreachable("unhandled ASTNode kind!");
  }

  void doIt(Decl *decl) {
    if (!decl || stopped)
      return;

    // Call walker.walkToDeclPre and handle the result.
    Action action = walker.walkToDeclPre(decl);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return;
    }

    // Visit the node's children.
    visit(decl);

    // Call walkToDeclPost and handle the result
    if (!stopped)
      stopped = !walker.walkToDeclPost(decl);
  }

  Expr *doIt(Expr *expr) {
    if (!expr || stopped)
      return nullptr;

    // The node that should take \p expr's place.
    Expr *replacement = nullptr;

    // Call walker.walkToExprPre and handle the result.
    Action action;
    std::tie(action, replacement) = walker.walkToExprPre(expr);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return replacement;
    }

    // Visit the node's children.
    visit(replacement ? replacement : expr);
    if (stopped)
      return replacement;

    // Call walkToExprPost and handle the result
    auto result = walker.walkToExprPost(expr);
    stopped = !result.first;
    return (result.second ? result.second : replacement);
  }

  void doIt(ParamList *paramList) {
    if (!paramList || stopped)
      return;

    // Call walker.walkToParamListPre and handle the result.
    Action action = walker.walkToParamListPre(paramList);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return;
    }

    // Visit the node's children.
    visit(paramList);

    // Call walkToParamListPost and handle the result
    if (!stopped)
      stopped = !walker.walkToParamListPost(paramList);
  }

  void doIt(Pattern *pattern) {
    if (!pattern || stopped)
      return;

    // Call walker.walkToPatternPre and handle the result.
    Action action = walker.walkToPatternPre(pattern);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return;
    }

    // Visit the node's children.
    visit(pattern);

    // Call walkToPatternPost and handle the result
    if (!stopped)
      stopped = !walker.walkToPatternPost(pattern);
  }

  void doIt(Stmt *stmt) {
    if (!stmt || stopped)
      return;

    // Call walker.walkToStmtPre and handle the result.
    Action action = walker.walkToStmtPre(stmt);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return;
    }

    // Visit the node's children.
    visit(stmt);

    // Call walkToStmtPost and handle the result
    if (!stopped)
      stopped = !walker.walkToStmtPost(stmt);
  }

  void doIt(StmtCondition &cond) {
    if (cond.isExpr()) {
      if (Expr *expr = doIt(cond.getExpr()))
        cond.setExpr(expr);
    }
    else
      llvm_unreachable("Unhandled StmtCondition Kind!");
  }

  void doIt(TypeLoc &tyLoc) {
    if (stopped)
      return;

    // Call walker.walkToTypeLocPre and handle the result.
    Action action = walker.walkToTypeLocPre(tyLoc);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return;
    }

    // Visit the TypeLoc's TypeRepr, if it has one.
    if (TypeRepr *tyRepr = tyLoc.getTypeRepr())
      doIt(tyRepr);

    // Call walkToTypeLocPost and handle the result
    if (!stopped)
      stopped = !walker.walkToTypeLocPost(tyLoc);
  }

  void doIt(TypeRepr *tyRepr) {
    if (!tyRepr || stopped)
      return;

    // Call walker.walkToTypeReprPre and handle the result.
    Action action = walker.walkToTypeReprPre(tyRepr);
    if (action != Action::Continue) {
      stopped = (action == Action::Stop);
      return;
    }

    // Visit the node's children.
    visit(tyRepr);

    // Call walkToTypeReprPost and handle the result
    if (!stopped)
      stopped = !walker.walkToTypeReprPost(tyRepr);
  }
}; // namespace
} // namespace

void ASTWalker::anchor() {}

// Walk implementations 

bool ASTNode::walk(ASTWalker &walker) { return Traversal(walker).walk(*this); }
bool Decl::walk(ASTWalker &walker) { return Traversal(walker).walk(this); }
bool Expr::walk(ASTWalker &walker) { return Traversal(walker).walk(this); }
bool Pattern::walk(ASTWalker &walker) { return Traversal(walker).walk(this); }
bool Stmt::walk(ASTWalker &walker) { return Traversal(walker).walk(this); }
bool TypeRepr::walk(ASTWalker &walker) { return Traversal(walker).walk(this); }