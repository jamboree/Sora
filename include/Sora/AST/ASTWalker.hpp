//===--- ASTWalker.hpp - AST Traversal Tool ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <memory>

namespace sora {
class Decl;
class Expr;
class ParamList;
class Pattern;
class Stmt;
class TypeRepr;
class TypeLoc;

/// The Sora AST Walker which allows smooth traversal of the AST.
///
/// The ASTWalker calls "walkToXPre" when visiting a X node in pre-order (before
/// visiting its children) and "walkToXPost" when visiting it in post-order
/// (after visiting its children). Both methods return an "action" that allows
/// you to control what to do next.
class ASTWalker {

protected:
  ASTWalker() = default;
  ASTWalker(const ASTWalker &) = default;
  virtual ~ASTWalker() = default;
  virtual void anchor();

public:
  /// Kinds of actions that can be performed during a pre-order walk.
  enum class Action : uint8_t {
    /// Continue the walk
    Continue,
    /// Continue the walk, but skip the node's children.
    SkipChildren,
    /// Stop the walk
    Stop
  };

  /// Called when first visiting \p decl before walking into its children.
  /// \returns the next action to take (default: Action::Continue)
  virtual Action walkToDeclPre(Decl *decl) { return Action::Continue; }

  /// Called after visiting \p decl's children
  /// If this returns false, the rest of the walk is cancelled and returns
  /// failure. The default implementation returns true.
  virtual bool walkToDeclPost(Decl *decl) { return true; }

  /// Called when first visiting \p expr before walking into its children.
  /// \returns a pair indicating the next action to take and a replacement node.
  /// \p expr is only replaced if the second element of the pair is not
  /// nullptr, and if the replacement is valid.
  /// For example, the if you're visiting the TupleExpr (args) of a CallExpr
  /// and you try to replace it with an ErrorExpr, the replacement will not be
  /// performed. The default implementation returns {Action::Continue, nullptr}
  virtual std::pair<Action, Expr *> walkToExprPre(Expr *expr) {
    return {Action::Continue, nullptr};
  }

  /// Called after visiting \p expr's children
  /// \returns a pair indicating whether to continue the walk and a replacement
  /// node. If the former is false, the remaining walk is cancelled and returns
  /// failure.
  /// NOTE: \p expr is only replaced if the Expr* is not nullptr, and
  /// if the replacement is valid.
  virtual std::pair<bool, Expr *> walkToExprPost(Expr *expr) {
    return {true, nullptr};
  }

  /// Called when first visiting \p paramList before walking into its children.
  /// \returns the next action to take (by default, Action::Continue)
  virtual Action walkToParamListPre(ParamList *paramList) {
    return Action::Continue;
  }

  /// Called after visiting \p paramList's children
  /// If this returns false, the rest of the walk is cancelled and returns
  /// failure. The default implementation returns true.
  virtual bool walkToParamListPost(ParamList *paramList) { return true; }

  /// Called when first visiting \p pattern before walking into its children.
  /// \returns the next action to take (by default, Action::Continue)
  virtual Action walkToPatternPre(Pattern *pattern) { return Action::Continue; }

  /// Called after visiting \p pattern's children
  /// If this returns false, the rest of the walk is cancelled and returns
  /// failure. The default implementation returns true.
  virtual bool walkToPatternPost(Pattern *pattern) { return true; }

  /// Called when first visiting \p stmt before walking into its children.
  /// \returns the next action to take (by default, Action::Continue)
  virtual Action walkToStmtPre(Stmt *stmt) { return Action::Continue; }

  /// Called after visiting \p stmt's children
  /// If this returns false, the rest of the walk is cancelled and returns
  /// failure. The default implementation returns true.
  virtual bool walkToStmtPost(Stmt *stmt) { return true; }

  /// Called when first visiting \p tyLoc before walking into its TypeRepr.
  /// \returns the next action to take (by default, Action::Continue)
  virtual Action walkToTypeLocPre(TypeLoc &tyLoc) { return Action::Continue; }

  /// Called after visiting \p tyLoc's TypeRepr
  /// If this returns false, the rest of the walk is cancelled and returns
  /// failure. The default implementation returns true.
  virtual bool walkToTypeLocPost(TypeLoc &tyLoc) { return true; }

  /// Called when first visiting \p tyRepr before walking into its children.
  /// \returns the next action to take (by default, Action::Continue)
  virtual Action walkToTypeReprPre(TypeRepr *tyRepr) {
    return Action::Continue;
  }

  /// Called after visiting \p tyRepr's children
  /// If this returns false, the rest of the walk is cancelled and returns
  /// failure. The default implementation returns true.
  virtual bool walkToTypeReprPost(TypeRepr *tyRepr) { return true; }
};
} // namespace sora