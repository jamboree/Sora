//===--- ASTNodeLoc.hpp - AST Node Source Loc Information Utils -*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Decl, Stmt and Expr all share some logic regarding the implementation of
// getBegLoc/getEndLoc and getSourceRange. We put that logic here to minimize
// code repetition.
//
// FIXME: "ASTNodeLoc" is kind of a weird name isn't it?
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/SourceLoc.hpp"
#include <type_traits>

namespace sora {
namespace detail {
template <typename Base, typename Rtr, typename Ty>
constexpr bool isOverriden(Rtr (Ty::*)() const) {
  return !std::is_same<Base, Ty>::value;
}
} // namespace detail

template <typename Base, typename Derived, bool usesGetLoc = true>
struct ASTNodeLoc {
  static constexpr bool hasGetRange =
      detail::isOverriden<Base>(&Derived::getSourceRange);
  static constexpr bool hasGetBeg =
      detail::isOverriden<Base>(&Derived::getBegLoc);
  static constexpr bool hasGetEnd =
      detail::isOverriden<Base>(&Derived::getEndLoc);

  /// Nodes must override (getSourceRange) or (getBegLoc & getEndLoc) or both
  static_assert(hasGetRange || (hasGetBeg && hasGetEnd),
                "AST Nodes must override (getSourceRange) or "
                "(getBegLoc/getEndLoc) or both.");

  static SourceRange getSourceRange(const Derived *node) {
    return hasGetRange ? node->getSourceRange()
                       : SourceRange(node->getBegLoc(), node->getEndLoc());
  }

  static SourceLoc getBegLoc(const Derived *node) {
    return hasGetBeg ? node->getBegLoc() : node->getSourceRange().begin;
  }

  template <typename = std::enable_if<usesGetLoc, void>>
  static SourceLoc getLoc(const Derived *node) {
    static constexpr bool hasGetLoc =
        detail::isOverriden<Base>(&Derived::getLoc);
    // Prefer to use the override if it exists, else use getBegLoc.
    return hasGetLoc ? node->getLoc() : getBegLoc(node);
  }

  static SourceLoc getEndLoc(const Derived *node) {
    return hasGetEnd ? node->getEndLoc() : node->getSourceRange().end;
  }
};
} // namespace sora