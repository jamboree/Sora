//===--- ASTVisitor.hpp - AST Nodes Visitor ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTNode.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Stmt.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <utility>

namespace sora {
template <typename Derived, typename ExprRtrTy = void,
          typename DeclRtrTy = void, typename StmtRtrTy = void,
          typename... Args>
class ASTVisitor {
public:
  using this_type = ASTVisitor;

  void visit(ASTNode node, Args... args) {
    assert(node && "node cannot be null!");
    if (node.is<Decl *>())
      visit(node.get<Decl *>(), ::std::forward<Args>(args)...);
    else if (node.is<Expr *>())
      visit(node.get<Expr *>(), ::std::forward<Args>(args)...);
    else if (node.is<Stmt *>())
      visit(node.get<Stmt *>(), ::std::forward<Args>(args)...);
    else
      llvm_unreachable("Unsupported ASTNode variant");
  }

  ExprRtrTy visit(Expr *expr, Args... args) {
    assert(expr && "Cannot be used on a null pointer");
    switch (expr->getKind()) {
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return static_cast<Derived *>(this)->visit##ID##Expr(                      \
        static_cast<ID##Expr *>(expr), ::std::forward<Args>(args)...);
#include "Sora/AST/ExprNodes.def"
    default:
      llvm_unreachable("Unknown node");
    }
  };

  DeclRtrTy visit(Decl *decl, Args... args) {
    assert(decl && "Cannot be used on a null pointer");
    switch (decl->getKind()) {
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return static_cast<Derived *>(this)->visit##ID##Decl(                      \
        static_cast<ID##Decl *>(decl), ::std::forward<Args>(args)...);
#include "Sora/AST/DeclNodes.def"
    default:
      llvm_unreachable("Unknown node");
    }
  };

  StmtRtrTy visit(Stmt *stmt, Args... args) {
    assert(stmt && "Cannot be used on a null pointer");
    switch (stmt->getKind()) {
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return static_cast<Derived *>(this)->visit##ID##Stmt(                      \
        static_cast<ID##Stmt *>(stmt), ::std::forward<Args>(args)...);
#include "Sora/AST/StmtNodes.def"
    default:
      llvm_unreachable("Unknown node");
    }
  };

  void visit(ParamList *paramList) {
    return static_cast<Derived *>(this)->visitParamList();
  }

  void visitParamList(ParamList *paramList) {}

  // Add default implementations that chain back to their base class, so we
  // require full coverage of the AST by visitors but we also allow them to
  // visit only a common base (like ValueDecl) and handle all derived classes.
#define VISIT_METHOD(RTR, NODE, PARENT)                                        \
  RTR visit##NODE(NODE *node, Args... args) {                                  \
    return static_cast<Derived *>(this)->visit##PARENT(                        \
        node, ::std::forward<Args>(args)...);                                  \
  }

#define DECL(ID, PARENT) VISIT_METHOD(DeclRtrTy, ID##Decl, PARENT)
#define ABSTRACT_DECL(ID, PARENT) VISIT_METHOD(DeclRtrTy, ID##Decl, PARENT)
#include "DeclNodes.def"
#define STMT(ID, PARENT) VISIT_METHOD(StmtRtrTy, ID##Stmt, PARENT)
#define ABSTRACT_STMT(ID, PARENT) VISIT_METHOD(StmtRtrTy, ID##Stmt, PARENT)
#include "StmtNodes.def"
#define EXPR(ID, PARENT) VISIT_METHOD(ExprRtrTy, ID##Expr, PARENT)
#define ABSTRACT_EXPR(ID, PARENT) VISIT_METHOD(ExprRtrTy, ID##Expr, PARENT)
#include "ExprNodes.def"
};

/// Typealias for simple AST visitors with a unique return type.
template <typename Derived, typename RtrTy = void, typename... Args>
using SimpleASTVisitor = ASTVisitor<Derived, RtrTy, RtrTy, RtrTy, Args...>;

/// Typealias for visitors that are only interested in expressions
template <typename Derived, typename RtrTy = void, typename... Args>
using ExprVisitor = ASTVisitor<Derived, RtrTy, void, void, Args...>;

/// Typealias for visitors that are only interested in declarations
template <typename Derived, typename RtrTy = void, typename... Args>
using DeclVisitor = ASTVisitor<Derived, void, RtrTy, void, Args...>;

/// Typealias for visitors that are only interested in statements
template <typename Derived, typename RtrTy = void, typename... Args>
using StmtVisitor = ASTVisitor<Derived, void, void, RtrTy, Args...>;
} // namespace sora