//===--- Expr.hpp - Expression ASTs -----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/OperatorKinds.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/TrailingObjects.h"
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

/// Base class for every Expression node.
class alignas(ExprAlignement) Expr {
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
  // Publicly allow allocation of expressions using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Expr));

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const;
  /// \returns the full range of this expression
  SourceRange getSourceRange() const;

  void setImplicit(bool implicit = true) { typeAndIsImplicit.setInt(implicit); }
  bool isImplicit() const { return typeAndIsImplicit.getInt(); }

  Type getType() const { return typeAndIsImplicit.getPointer(); }
  bool hasType() const { return (bool)getType(); }
  void setType(Type type) { typeAndIsImplicit.setPointer(type); }

  /// Recursively ignores the ParenExprs that might surround this expression.
  /// \returns the first expression found that isn't a ParenExpr
  Expr *ignoreParens();

  /// Recursively ignores the ParenExprs that might surround this expression.
  /// \returns the first expression found that isn't a ParenExpr
  const Expr *ignoreParens() const {
    return const_cast<Expr *>(this)->ignoreParens();
  }

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

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return identLoc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const { return identLoc; }

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

  SourceLoc getLoc() const { return loc; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Discard;
  }
};

/// Common base class for literal expressions.
class AnyLiteralExpr : public Expr {
  /// Currently, literals are always represented using a single token, so we can
  /// simply store the loc of the token here.
  SourceLoc loc;

protected:
  AnyLiteralExpr(ExprKind kind, SourceLoc loc) : Expr(kind), loc(loc) {}

public:
  SourceLoc getLoc() const { return loc; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Expr *expr) {
    return (expr->getKind() >= ExprKind::First_AnyLiteral) &&
           (expr->getKind() <= ExprKind::Last_AnyLiteral);
  }
};

/// Represents an integer literal (42, 320, etc.)
class IntegerLiteralExpr final : public AnyLiteralExpr {
  /// Store the literal as a StringRef because APInt isn't trivially
  /// destructible.
  StringRef strValue;

public:
  IntegerLiteralExpr(StringRef strValue, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::IntegerLiteral, loc), strValue(strValue) {}

  /// \returns the string version of the literal as written by the user
  StringRef getString() const { return strValue; }

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
class FloatLiteralExpr final : public AnyLiteralExpr {
  /// Store the literal as a StringRef because APInt isn't trivially
  /// destructible.
  StringRef strValue;

public:
  FloatLiteralExpr(StringRef strValue, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::FloatLiteral, loc), strValue(strValue) {}

  /// \returns the string literal as written by the user
  StringRef getString() const { return strValue; }

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
class BooleanLiteralExpr final : public AnyLiteralExpr {
  bool value;

public:
  BooleanLiteralExpr(bool value, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::BooleanLiteral, loc), value(value) {}

  bool getValue() const { return value; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::BooleanLiteral;
  }
};

/// Represents a null pointer literal (null).
class NullLiteralExpr final : public AnyLiteralExpr {
public:
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
class ErrorExpr final : public Expr {
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

  /// \returns the full range of this expression. (This is the original
  /// SourceRange of the expression that couldn't be resolved)
  SourceRange getSourceRange() const { return range; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Error;
  }
};

/// Represents a tuple indexing expression (base.index)
///
/// e.g. tuple.0, (0, 1, 2).2, etc.
///
/// The base (the tuple) can be any expression, and the index is always an
/// IntegerLiteralExpr (of type usize).
class TupleIndexingExpr final : public Expr {
  Expr *base;
  SourceLoc dotLoc;
  IntegerLiteralExpr *index;

public:
  TupleIndexingExpr(Expr *base, SourceLoc dotLoc, IntegerLiteralExpr *index)
      : Expr(ExprKind::TupleIndexing), base(base), dotLoc(dotLoc),
        index(index) {}

  Expr *getBase() const { return base; }
  void setBase(Expr *base) { this->base = base; }

  IntegerLiteralExpr *getIndex() const { return index; }
  void setIndex(IntegerLiteralExpr *index) { this->index = index; }

  SourceLoc getDotLoc() const { return dotLoc; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const {
    assert(base && "no base expr");
    return base->getBegLoc();
  }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const {
    assert(index && "no index expr");
    return index->getEndLoc();
  }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::TupleIndexing;
  }
};

/// Represents a Tuple Expression, which is a list of expressions between
/// parentheses.
///
/// Note that single-element tuples that aren't part of a call aren't
/// represented as a TupleExpr but as a ParenExpr. For instance, (0) is
/// represented at a ParenExpr, but in foo(0), the (0) is represented as a
/// TupleExpr.
///
/// The expressions and the SourceLocs of the commas are stored as trailing
/// objects.
///
/// \verbatim
/// Example: for the expression (0, 1, 2)
///   (     ->      getBegLoc()     or getLParenLoc()
///   0     ->      getElement(0)   or getElements()[0]
///   ,     ->      getCommaLoc(0)  or getCommaLocs()[0]
///   1     ->      getElement(1)   or getElements()[1]
///   ,     ->      getCommaLoc(1)  or getCommaLocs()[1]
///   2     ->      getElement(2)   or getElements()[2]
///   )     ->      getEndLoc()     or getRParenloc()
/// \endverbatim
class TupleExpr final
    : public Expr,
      private llvm::TrailingObjects<TupleExpr, Expr *, SourceLoc> {
  friend llvm::TrailingObjects<TupleExpr, Expr *, SourceLoc>;

  TupleExpr(SourceLoc lParenLoc, ArrayRef<Expr *> exprs,
            ArrayRef<SourceLoc> locs, SourceLoc rParenLoc);

  size_t numTrailingObjects(OverloadToken<Expr *>) const { return numElements; }

  size_t numTrailingObjects(OverloadToken<SourceLoc>) const {
    return getNumCommas();
  }

  SourceLoc lParenLoc, rParenLoc;
  size_t numElements = 0;

public:
  /// Creates a TupleExpr with one or more element.
  ///
  /// Note that the size of the \p commaLocs array must be that of the \p exprs
  /// array minus one, or zero if \p expr's size is zero.
  ///
  /// \verbatim
  ///   if exprs.size() = 0, commaLocs.size() must be 0
  ///   if exprs.size() = 1, commaLocs.size() must be 0
  ///   if exprs.size() = 2, commaLocs.size() must be 1
  ///   if exprs.size() = 30, commaLocs.size() must be 29
  /// \endverbatim
  static TupleExpr *create(ASTContext &ctxt, SourceLoc lParenLoc,
                           ArrayRef<Expr *> exprs,
                           ArrayRef<SourceLoc> commaLocs, SourceLoc rParenLoc);

  /// Creates an empty TupleExpr.
  static TupleExpr *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                SourceLoc rParenLoc);

  size_t getNumElements() const { return numElements; }
  MutableArrayRef<Expr *> getElements();
  ArrayRef<Expr *> getElements() const;
  Expr *getElement(size_t n);
  void setElement(size_t n, Expr *expr);

  size_t getNumCommas() const { return numElements ? numElements - 1 : 0; }
  SourceLoc getCommaLoc(size_t n) const;
  ArrayRef<SourceLoc> getCommaLocs() const;
  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  bool isEmpty() const { return numElements == 0; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return lParenLoc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Tuple;
  }
};

/// Represents a parenthesized expression
///
/// e.g. (0), (foo), etc.
class ParenExpr final : public Expr {
  Expr *subExpr;
  SourceLoc lParenLoc, rParenLoc;

public:
  ParenExpr(SourceLoc lParenLoc, Expr *subExpr, SourceLoc rParenLoc)
      : Expr(ExprKind::Paren), subExpr(subExpr), lParenLoc(lParenLoc),
        rParenLoc(rParenLoc) {}

  Expr *getSubExpr() const { return subExpr; }
  void setSubExpr(Expr *subExpr) { this->subExpr = subExpr; }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return lParenLoc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Paren;
  }
};

/// Represents a function call
///
/// e.g. foo(0), bar()
class CallExpr final : public Expr {
  /// The function being called
  Expr *fn;
  /// The arguments passed to the function
  TupleExpr *args;

public:
  CallExpr(Expr *fn, TupleExpr *args)
      : Expr(ExprKind::Call), fn(fn), args(args) {}

  Expr *getFn() const { return fn; }
  void setFn(Expr *base) { this->fn = base; }

  TupleExpr *getArgs() const { return args; }
  void setArgs(TupleExpr *args) { this->args = args; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const {
    assert(fn && "no fn");
    return fn->getBegLoc();
  }

  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const {
    assert(args && "no args");
    return args->getEndLoc();
  }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Call;
  }
};

/// Represents an infix binary operation
///
/// e.g a = b, 1 + 2, c += d, 32 ^ 56
class BinaryExpr final : public Expr {
public:
  using OpKind = BinaryOperatorKind;

private:
  Expr *lhs;
  Expr *rhs;
  OpKind op;
  SourceLoc opLoc;

public:
  BinaryExpr(Expr *lhs, OpKind op, SourceLoc opLoc, Expr *rhs)
      : Expr(ExprKind::Binary), lhs(lhs), rhs(rhs), op(op), opLoc(opLoc) {}

  Expr *getLHS() const { return lhs; }
  void setLHS(Expr *lhs) { this->lhs = lhs; }

  Expr *getRHS() const { return rhs; }
  void setRHS(Expr *rhs) { this->rhs = rhs; }

  SourceLoc getOpLoc() const { return opLoc; }
  OpKind getOpKind() const { return op; }
  /// \returns the spelling of the operator (e.g. "+" for Add)
  const char *getOpSpelling() const { return getSpelling(op); }

  /// \returns true if \p op is + or -
  bool isAdditiveOp() const { return sora::isAdditiveOp(op); }
  /// \returns true if \p op is * / or %
  bool isMultiplicativeOp() const { return sora::isMultiplicativeOp(op); }
  /// \returns true if \p op is << or >>
  bool isShiftOp() const { return sora::isShiftOp(op); }
  /// \returns true if \p op is | & or ^
  bool isBitwiseOp() const { return sora::isBitwiseOp(op); }
  /// \returns true if \p op is == or !=
  bool isEqualityOp() const { return sora::isEqualityOp(op); }
  /// \returns true if \p op is < <= > or >=
  bool isRelationalOp() const { return sora::isRelationalOp(op); }
  /// \returns true if \p op is || or &&
  bool isLogicalOp() const { return sora::isLogicalOp(op); }
  /// \returns true if \p op is any assignement operator
  bool isAssignementOp() const { return sora::isAssignementOp(op); }
  /// \returns true if \p op is a compound assignement operator
  bool isCompoundAssignementOp() const {
    return sora::isCompoundAssignementOp(op);
  }

  /// \returns the operator of a compound assignement. e.g. for AddAssign this
  /// returns Add.
  OpKind getOpForCompoundAssignementOp() const {
    return sora::getOpForCompoundAssignementOp(op);
  }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const {
    assert(lhs && "no lhs");
    return lhs->getBegLoc();
  }

  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const {
    assert(rhs && "no rhs");
    return rhs->getEndLoc();
  }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Binary;
  }
};

/// Represents a prefix unary operation
///
/// e.g. +1, -2, &foo
class UnaryExpr final : public Expr {
public:
  using OpKind = UnaryOperatorKind;

private:
  Expr *subExpr;
  OpKind op;
  SourceLoc opLoc;

public:
  UnaryExpr(OpKind op, SourceLoc opLoc, Expr *subExpr)
      : Expr(ExprKind::Unary), subExpr(subExpr), op(op), opLoc(opLoc) {}

  Expr *getSubExpr() const { return subExpr; }
  void setSubExpr(Expr *expr) { subExpr = expr; }

  SourceLoc getOpLoc() const { return opLoc; }
  OpKind getOpKind() const { return op; }
  /// \returns the spelling of the operator
  const char *getOpSpelling() const { return getSpelling(op); }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return opLoc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const {
    assert(subExpr && "no subExpr");
    return subExpr->getEndLoc();
  }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Unary;
  }
};
} // namespace sora