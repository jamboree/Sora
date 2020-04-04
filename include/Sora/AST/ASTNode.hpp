//===--- ASTNode.hpp - Pointer Union Of All Major AST Nodes  ----*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PointerUnion.h"

namespace sora {
class ASTWalker;
class SourceRange;
class SourceLoc;
class Expr;
class Stmt;
class Decl;

/// The ASTNode is a intrusive pointer union that can contain a pointer
/// to any major AST Node: Declaration, Statements and Expressions.
struct ASTNode : public llvm::PointerUnion<Expr *, Stmt *, Decl *> {
  using Base = llvm::PointerUnion<Expr *, Stmt *, Decl *>;

  // Inherit the constructors from PointerUnion
  using llvm::PointerUnion<Expr *, Stmt *, Decl *>::PointerUnion;

  /// \returns the SourceRange of the Decl/Expr/Stmt
  SourceRange getSourceRange() const;
  /// \returns the SourceLoc of the start of the node
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the end of the node
  SourceLoc getEndLoc() const;

  /// Traverse this ASTNode using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
};
} // namespace sora

namespace llvm {
template <> struct DenseMapInfo<::sora::ASTNode> {
  static inline ::sora::ASTNode getEmptyKey() {
    return DenseMapInfo<::sora::Expr *>::getEmptyKey();
  }

  static inline ::sora::ASTNode getTombstoneKey() {
    return DenseMapInfo<::sora::Expr *>::getTombstoneKey();
  }

  static unsigned getHashValue(const ::sora::ASTNode &Node) {
    intptr_t key = (intptr_t)Node.getOpaqueValue();
    return DenseMapInfo<intptr_t>::getHashValue(key);
  }

  static bool isEqual(const ::sora::ASTNode &LHS, const ::sora::ASTNode &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm