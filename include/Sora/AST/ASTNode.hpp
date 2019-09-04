//===--- ASTNode.hpp - Pointer Union Of All Major AST Nodes  ----*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/PointerUnion.h"

namespace sora {
class SourceRange;
class SourceLoc;
class Expr;
class Stmt;
class Decl;

/// The ASTNode is a intrusive pointer union that can contain a pointer
/// to any major AST Node: Declaration, Statements and Expressions.
struct ASTNode : public llvm::PointerUnion3<Expr *, Stmt *, Decl *> {
  // Inherit the constructors from PointerUnion
  using llvm::PointerUnion3<Expr *, Stmt *, Decl *>::PointerUnion3;

  /// \returns the SourceRange of the Decl/Expr/Stmt
  SourceRange getSourceRange() const;
  /// \returns the SourceLoc of the start of the node
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the end of the node
  SourceLoc getEndLoc() const;
};
} // namespace sora