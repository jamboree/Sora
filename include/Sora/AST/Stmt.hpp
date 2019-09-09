//===--- Stmt.hpp - Statement ASTs -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include <cassert>
#include <stdint.h>

namespace sora {
class Expr;
class ASTContext;

/// Kinds of Statements
enum class StmtKind : uint8_t {
#define STMT(KIND, PARENT) KIND,
#include "Sora/AST/StmtNodes.def"
};

/// Base class for every Statement node.
class alignas(StmtAlignement) Stmt {
  // Disable vanilla new/delete for statements
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  StmtKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Stmt(StmtKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of statements using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Stmt));

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const;
  /// \returns the full range of this statement
  SourceRange getSourceRange() const;

  /// \return the kind of statement this is
  StmtKind getKind() const { return kind; }
};
} // namespace sora