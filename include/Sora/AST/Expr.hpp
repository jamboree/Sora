//===--- Expr.hpp - Expression ASTs -----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
/// Kinds of Expressions
enum class ExprKind : uint8_t {
#define EXPR(KIND, PARENT) KIND,
#define EXPR_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/ExprNodes.def"
};

class alignas(sora::ExprAlignement) Expr {
  // Disable vanilla new/delete for expressions
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *) noexcept = delete;

  ExprKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Expr(ExprKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of expressions using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Expr));

  /// \return the kind of expression this is
  ExprKind getKind() const { return kind; }

private:
};
} // namespace sora