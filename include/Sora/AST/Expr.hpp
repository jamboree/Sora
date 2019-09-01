//===--- Expr.hpp - Expression ASTs -----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/PointerIntPair.h"
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
  void operator delete(void *)noexcept = delete;

  ExprKind kind;
  llvm::PointerIntPair<Type, 1, bool> typeAndIsImplicit;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Expr(ExprKind kind) : kind(kind) {}

public:
  /// \returns the starting location of this expression
  SourceLoc getStartLoc() const;
  /// \returns the ending location of this expression
  SourceLoc getEndLoc() const;
  /// \returns the range of this expression
  /// FIXME: Can this be made more efficient?
  SourceRange getSourceRange() const {
    return SourceRange(getStartLoc(), getEndLoc());
  }
  
  /// Marks this expression as being implicit (or not)
  bool setImplicit(bool implicit = true) { typeAndIsImplicit.setInt(implicit); }
  /// \returns whether this expression is implicit or not
  bool isImplicit() const { return typeAndIsImplicit.getInt(); }

  /// \returns the type of this expression
  Type getType() const { return typeAndIsImplicit.getPointer(); }

  /// \returns true if this expression has a type
  bool hasType() const { return (bool)getType(); }

  /// Sets the type of this expression to \p type
  void setType(Type type) { typeAndIsImplicit.setPointer(type); }

  // Publicly allow allocation of expressions using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Expr));

  /// \return the kind of expression this is
  ExprKind getKind() const { return kind; }
};

/// Common base class for "unresolved" expressions.
///
/// Unresolved expressions are expressions that the parser can't
/// resolve. (For instance, the use of an identifier. The parser doesn't know
/// what it resolves to). Theses are allocated in their own memory pool and are
/// replaced by
class UnresolvedExpr : public Expr {
protected:
  UnresolvedExpr(ExprKind kind) : Expr(kind) {}

public:
  // Publicly allow allocation of unresolved expressions using the ASTContext's
  // "unresolved" allocator.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Expr));
};

class UnresolvedDeclRefExpr : public UnresolvedExpr {
  Identifier ident;
  SourceLoc identLoc;

public:
  UnresolvedDeclRefExpr(Identifier ident, SourceLoc identLoc)
      : UnresolvedExpr(ExprKind::UnresolvedDeclRef), ident(ident),
        identLoc(identLoc) {}

  /// \returns the starting location of this expression
  SourceLoc getStartLoc() const { return identLoc; }
  /// \returns the ending location of this expression
  SourceLoc getEndLoc() const { return identLoc; }
};
} // namespace sora