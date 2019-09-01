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
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringRef.h"
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
  /// \returns the location of the beginning of the expression
  SourceLoc getBegLoc() const;
  /// \returns the location of the end of the expression
  SourceLoc getEndLoc() const;
  /// \returns the full range of this expression
  SourceRange getSourceRange() const;

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

  static bool classof(const Expr *expr) {
    return (expr->getKind() >= ExprKind::First_Unresolved) &&
           (expr->getKind() <= ExprKind::Last_Unresolved);
  }
};

/// Represents an unresolved reference to a declaration. This is always a
/// single identifier.
class UnresolvedDeclRefExpr final : public UnresolvedExpr {
  Identifier ident;
  SourceLoc identLoc;

public:
  UnresolvedDeclRefExpr(Identifier ident, SourceLoc identLoc)
      : UnresolvedExpr(ExprKind::UnresolvedDeclRef), ident(ident),
        identLoc(identLoc) {}

  /// \returns the location of the beginning of the expression
  SourceLoc getBegLoc() const { return identLoc; }
  /// \returns the location of the end of the expression
  SourceLoc getEndLoc() const { return identLoc; }
  /// \returns the identifier referenced
  Identifier getIdentifier() const { return ident; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::UnresolvedDeclRef;
  }
};

/// Represents the "discard" variable, which is a write-only variable
/// whose name is an underscore '_'.
class DiscardExpr final : public Expr {
  SourceLoc loc;

public:
  DiscardExpr(SourceLoc loc) : Expr(ExprKind::Discard), loc(loc) {}

  /// \returns the location of the beginning of the expression
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the location of the end of the expression
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Discard;
  }
};

/// Common base class for literal expressions.
class AnyLiteralExpr : public Expr {
  /// Currently, literals are always single tokens, so we can simply
  /// store the loc here.
  SourceLoc loc;

protected:
  AnyLiteralExpr(ExprKind kind, SourceLoc loc) : Expr(kind), loc(loc) {}

public:
  /// \returns the location of the beginning of the expression
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the location of the end of the expression
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Expr *expr) {
    return (expr->getKind() >= ExprKind::First_AnyLiteral) &&
           (expr->getKind() <= ExprKind::Last_AnyLiteral);
  }
};

/// Represents an integer literal (42, 320, etc.)
class IntegerLiteralExpr : public AnyLiteralExpr {
  /// Store the literal as a StringRef because APInt isn't trivially
  /// destructible.
  StringRef strValue;

public:
  /// \param strValue the string value as written by the user
  /// \param loc the loc of the literal
  IntegerLiteralExpr(StringRef strValue, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::IntegerLiteral, loc), strValue(strValue) {}

  /// \returns the string literal as written by the user
  StringRef asString() const { return strValue; }

  /// \returns the raw integer value (that doesn't respect the target's type bit
  /// width)
  ///
  /// This is relatively expensive (parsing isn't cached), so don't abuse it
  /// where performance matters.
  APInt getRawValue() const;

  /// \returns the value as an APInt that respects the target's type.
  ///
  /// e.g. if this has a i32 type, this returns a 32 bit integer, if it has a
  /// u64 type, this returns a unsigned 64 bits integer, etc.
  ///
  /// This is relatively expensive (parsing isn't cached), so don't abuse it
  /// where performance matters.
  /* unimplemented for now - requires llvm::fltSemantics which should be
     fetched from the type */
  // APFloat getValue() const;

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::IntegerLiteral;
  }
};

/// Represents a floating-point literal (3.14, 42.42, etc.)
class FloatLiteralExpr : public AnyLiteralExpr {
  /// Store the literal as a StringRef because APInt isn't trivially
  /// destructible.
  StringRef strValue;

public:
  /// \param strValue the string value as written by the user
  /// \param loc the loc of the literal
  FloatLiteralExpr(StringRef strValue, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::FloatLiteral, loc), strValue(strValue) {}

  /// \returns the string literal as written by the user
  StringRef asString() const { return strValue; }

  /// \returns the value as an APFloat that respects the target's type.
  ///
  /// e.g. if this has a f32 type, this returns a single precision float, if
  /// this has a f64 type, it returns a double-precision float.
  ///
  /// This is relatively expensive (parsing isn't cached), so don't abuse it
  /// where performance matters.
  /* unimplemented for now - requires llvm::fltSemantics which should be
     fetched from the FloatType */
  // APFloat getValue() const;

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::FloatLiteral;
  }
};

/// Represents a boolean literal (true or false)
class BooleanLiteralExpr : public AnyLiteralExpr {
  bool value;

public:
  /// \param value the value of the literal
  /// \param loc the loc of the literal (of the "true" or "false" keyword
  BooleanLiteralExpr(bool value, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::BooleanLiteral, loc), value(value) {}

  /// \returns the value of the literal
  bool getValue() const { return value; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::BooleanLiteral;
  }
};

/// Represents a null pointer literal (null).
class NullLiteralExpr : public AnyLiteralExpr {
public:
  /// \param loc the loc of the "null" keyword
  NullLiteralExpr(SourceLoc loc) : AnyLiteralExpr(ExprKind::NullLiteral, loc) {}

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::NullLiteral;
  }
};

/// Represents an error expr.
///
/// This is created when Sema cannot resolve an UnresolvedExpr.
///
/// e.g. you got a UnresolvedDeclRefExpr "foo", and Sema can't find anything
/// named Foo. It'll simply create one of those to replace the
/// UnresolvedDeclRefExpr.
class ErrorExpr : public Expr {
  /// The original range of the node that couldn't be resolved
  SourceRange range;

public:
  /// \param loc the loc of the "null" keyword
  ErrorExpr(SourceRange range) : Expr(ExprKind::Error), range(range) {}

  /// Create an ErrorExpr from an UnresolvedExpr.
  /// This is simply a convenience method to avoid doing
  /// "ErrorExpr(theExpr->getSourceRange())" which can be quite repetitive.
  /// \param expr the expression that couldn't be resolved
  ErrorExpr(UnresolvedExpr *expr)
      : Expr(ExprKind::Error), range(expr->getSourceRange()) {}

  /// \returns the location of the beginning of the expression
  SourceLoc getBegLoc() const { return range.begin; }
  /// \returns the location of the end of the expression
  SourceLoc getEndLoc() const { return range.end; }
  /// \returns the full range of this expression
  SourceRange getSourceRange() const { return range; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Error;
  }
};
} // namespace sora