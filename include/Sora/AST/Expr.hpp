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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
class ASTWalker;

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

  llvm::PointerIntPair<Type, 1, bool> typeAndIsImplicit;
  ExprKind kind;
  /// Make use of the padding bits by allowing derived class to store data here.
  /// NOTE: Derived classes are expected to initialize the bitfields.
  LLVM_PACKED(union Bits {
    Bits() : raw() {}
    // Raw bits (to zero-init the union)
    char raw[7];
    // UnresolvedMemberRefExpr bits
    struct {
      // whether the operator used was the '->' operator
      bool isArrow;
    } unresMembRefExpr;
    // BooleanLiteralExpr bits
    struct {
      bool value;
    } booleanLiteralExpr;
    /// TupleElementExpr bits
    struct {
      // whether the operator used was the '->' operator
      bool isArrow;
      // the index of the element in the tuple
      unsigned index;
    } tupleEltExpr;
    /// TupleExpr bits
    struct {
      uint32_t numElements;
    } tupleExpr;
    /// BinaryExpr bits
    struct {
      BinaryOperatorKind opKind;
    } binaryExpr;
    /// UnaryExpr bits
    struct {
      UnaryOperatorKind opKind;
    } unaryExpr;
  });
  static_assert(sizeof(Bits) == 7, "Bits is too large!");

protected:
  Bits bits;

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

  void setImplicit(bool implicit = true) { typeAndIsImplicit.setInt(implicit); }
  bool isImplicit() const { return typeAndIsImplicit.getInt(); }

  Type getType() const { return typeAndIsImplicit.getPointer(); }
  bool hasType() const { return (bool)getType(); }
  void setType(Type type) { typeAndIsImplicit.setPointer(type); }

  /// Skips parentheses around this Expr: If this is a ParenExpr, returns
  /// getSubExpr()->ignoreParens(), else returns this.
  Expr *ignoreParens();
  const Expr *ignoreParens() const {
    return const_cast<Expr *>(this)->ignoreParens();
  }

  /// Traverse this Expr using \p walker.
  /// \returns a pair, the first element indicates whether the walk completed
  /// successfully (true = success), and the second element, if non-null, is the
  /// Expr that should replace this Expr in the tree.
  std::pair<bool, Expr *> walk(ASTWalker &walker);
  std::pair<bool, Expr *> walk(ASTWalker &&walker) { return walk(walker); }

  /// Dumps this expression to \p out
  void dump(raw_ostream &out, const SourceManager &srcMgr,
            unsigned indent = 2) const;

  /// \return the kind of expression this is
  ExprKind getKind() const { return kind; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const;
  /// \returns the preffered SourceLoc for diagnostics. This is defaults to
  /// getBegLoc but nodes can override it as they please.
  SourceLoc getLoc() const;
  /// \returns the full range of this expression
  SourceRange getSourceRange() const;
};

/// We should only use 16 bytes (2 pointers) max in 64 bits mode.
/// One pointer for the type (+ the "packed" isImplicit flag) and
/// one for the kind + packed bits.
static_assert(sizeof(Expr) <= 16, "Expr is too big!");

/// Common base class for "unresolved" expressions.
///
/// UnresolvedExpr expressions are expressions that the parser can't
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

/// Represents an unresolved reference to a declaration.
/// This is simply an identifier and a SourceLoc.
class UnresolvedDeclRefExpr final : public UnresolvedExpr {
  Identifier ident;
  SourceLoc identLoc;

public:
  UnresolvedDeclRefExpr(Identifier ident, SourceLoc identLoc)
      : UnresolvedExpr(ExprKind::UnresolvedDeclRef), ident(ident),
        identLoc(identLoc) {}

  SourceLoc getBegLoc() const { return identLoc; }
  SourceLoc getEndLoc() const { return identLoc; }

  Identifier getIdentifier() const { return ident; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::UnresolvedDeclRef;
  }
};

/// Represents an unresolved member access on an expression.
/// This represents both unresolved member accesses on struct, and unresolved
/// tuple indexing. In both cases, the "RHS" is stored as an identifier.
///
/// \verbatim
///   foo.bar
///   foo.0
///   foo->bar
///   foo->0
/// \endverbatim
class UnresolvedMemberRefExpr final : public UnresolvedExpr {
  Expr *base = nullptr;
  SourceLoc opLoc, memberIdentLoc;
  Identifier memberIdent;

public:
  UnresolvedMemberRefExpr(Expr *base, SourceLoc opLoc, bool isArrow,
                          SourceLoc memberIdentLoc, Identifier memberIdent)
      : UnresolvedExpr(ExprKind::UnresolvedMemberRef), base(base), opLoc(opLoc),
        memberIdentLoc(memberIdentLoc), memberIdent(memberIdent) {
    bits.unresMembRefExpr.isArrow = isArrow;
  }

  Expr *getBase() const { return base; }
  void setBase(Expr *expr) { base = expr; }

  /// \returns the SourceLoc of the '.' or '->'
  SourceLoc getOpLoc() const { return opLoc; }

  /// \returns true if the operator used was '->', false if it was '.'
  bool isArrow() const { return bits.unresMembRefExpr.isArrow; }
  /// \returns true if the operator used was '.', false if it was '->'
  bool isDot() const { return !isArrow(); }

  SourceLoc getMemberIdentifierLoc() const { return memberIdentLoc; }
  Identifier getMemberIdentifier() const { return memberIdent; }

  SourceLoc getBegLoc() const { return base->getBegLoc(); }
  SourceLoc getEndLoc() const { return memberIdentLoc; }
  SourceLoc getLoc() const { return opLoc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::UnresolvedMemberRef;
  }
};

/// Represents the "discard" variable, which is a write-only variable
/// whose name is an underscore '_'. This always has a contextual l-value type.
class DiscardExpr final : public Expr {
  SourceLoc loc;

public:
  DiscardExpr(SourceLoc loc) : Expr(ExprKind::Discard), loc(loc) {}

  SourceLoc getBegLoc() const { return loc; }
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
public:
  BooleanLiteralExpr(bool value, SourceLoc loc)
      : AnyLiteralExpr(ExprKind::BooleanLiteral, loc) {
    bits.booleanLiteralExpr.value = value;
  }

  bool getValue() const { return bits.booleanLiteralExpr.value; }

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

/// Represents an expression that refers to an element of a tuple
///
/// e.g. tuple.0, (0, 1, 2).2, etc.
class TupleElementExpr final : public Expr {
  Expr *base;
  SourceLoc opLoc, indexLoc;

public:
  TupleElementExpr(Expr *base, SourceLoc opLoc, bool isArrow,
                   SourceLoc indexLoc, unsigned index)
      : Expr(ExprKind::TupleElement), base(base), opLoc(opLoc),
        indexLoc(indexLoc) {
    bits.tupleEltExpr.index = index;
  }

  Expr *getBase() const { return base; }
  void setBase(Expr *base) { this->base = base; }

  unsigned getIndex() const { return bits.tupleEltExpr.index; }
  SourceLoc getIndexLoc() const { return indexLoc; }

  /// \returns the SourceLoc of the '.' or '->'
  SourceLoc getOpLoc() const { return opLoc; }

  /// \returns true if the operator used was '->', false if it was '.'
  bool isArrow() const { return bits.tupleEltExpr.isArrow; }
  /// \returns true if the operator used was '.', false if it was '->'
  bool isDot() const { return !isArrow(); }

  SourceLoc getBegLoc() const {
    assert(base && "no base expr");
    return base->getBegLoc();
  }
  SourceLoc getEndLoc() const { return indexLoc; }
  SourceLoc getLoc() const { return opLoc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::TupleElement;
  }
};

/// Represents a list of zero or more expressions in parentheses.
/// Note that there are no single-element tuples, so expressions like "(0)" are
/// represented using ParenExpr.
class TupleExpr final : public Expr,
                        private llvm::TrailingObjects<TupleExpr, Expr *> {
  friend llvm::TrailingObjects<TupleExpr, Expr *>;

  SourceLoc lParenLoc, rParenLoc;

  TupleExpr(SourceLoc lParenLoc, ArrayRef<Expr *> exprs, SourceLoc rParenLoc)
      : Expr(ExprKind::Tuple), lParenLoc(lParenLoc), rParenLoc(rParenLoc) {
    assert(exprs.size() != 1 &&
           "Single-element tuples don't exist - Use ParenExpr!");
    bits.tupleExpr.numElements = exprs.size();
    std::uninitialized_copy(exprs.begin(), exprs.end(),
                            getTrailingObjects<Expr *>());
  }

public:
  /// Creates a TupleExpr. Note that \p exprs must contain either zero elements,
  /// or 2+ elements. It can't contain a single element as one-element tuples
  /// don't exist in Sora (There's no way to write them). Things like
  /// "(expr)" are represented using a ParenExpr instead.
  static TupleExpr *create(ASTContext &ctxt, SourceLoc lParenLoc,
                           ArrayRef<Expr *> exprs, SourceLoc rParenLoc);

  static TupleExpr *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                SourceLoc rParenLoc) {
    return create(ctxt, lParenLoc, {}, rParenLoc);
  }

  bool isEmpty() const { return getNumElements() == 0; }
  size_t getNumElements() const { return bits.tupleExpr.numElements; }
  MutableArrayRef<Expr *> getElements() {
    return {getTrailingObjects<Expr *>(), getNumElements()};
  }
  ArrayRef<Expr *> getElements() const {
    return {getTrailingObjects<Expr *>(), getNumElements()};
  }
  Expr *getElement(size_t n) { return getElements()[n]; }
  void setElement(size_t n, Expr *expr) { getElements()[n] = expr; }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  /// \returns the SourceLoc of the first token of the expression
  SourceLoc getBegLoc() const { return lParenLoc; }
  /// \returns the SourceLoc of the last token of the expression
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Tuple;
  }
};

/// Represents a simple parenthesized expression.
/// This always has the same type as its subexpression.
///
/// \verbatim
/// (foo)
/// foo(3) // (3) is a ParenExpr
/// \endverbatim
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
  /// \returns the preffered SourceLoc for diagnostics
  SourceLoc getLoc() const { return subExpr->getBegLoc(); }

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
  /// The arguments passed to the function. This is usually a ParenExpr, a
  /// TupleExpr or an ErrorExpr.
  Expr *args;

public:
  CallExpr(Expr *fn, Expr *args) : Expr(ExprKind::Call), fn(fn), args(args) {}

  Expr *getFn() const { return fn; }
  void setFn(Expr *base) { this->fn = base; }

  Expr *getArgs() const { return args; }
  void setArgs(Expr *args) { this->args = args; }

  SourceLoc getBegLoc() const {
    assert(fn && "no fn");
    return fn->getBegLoc();
  }
  SourceLoc getEndLoc() const {
    assert(args && "no args");
    return args->getEndLoc();
  }
  SourceLoc getLoc() const { return fn->getLoc(); }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Call;
  }
};

/// Represents a conditional (AKA ternary) expression.
/// e.g. foo ? a : b
class ConditionalExpr final : public Expr {
  Expr *condExpr, *thenExpr, *elseExpr;
  SourceLoc questionLoc, colonLoc;

public:
  ConditionalExpr(Expr *condExpr, SourceLoc questionLoc, Expr *thenExpr,
                  SourceLoc colonLoc, Expr *elseExpr)
      : Expr(ExprKind::Conditional), condExpr(condExpr), thenExpr(thenExpr),
        elseExpr(elseExpr), questionLoc(questionLoc), colonLoc(colonLoc) {}

  SourceLoc getQuestionLoc() const { return questionLoc; }
  SourceLoc getColonLoc() const { return colonLoc; }

  Expr *getCond() const { return condExpr; }
  void setCond(Expr *expr) { condExpr = expr; }

  Expr *getThen() const { return thenExpr; }
  void setThen(Expr *expr) { thenExpr = expr; }

  Expr *getElse() const { return elseExpr; }
  void setElse(Expr *expr) { elseExpr = expr; }

  SourceLoc getBegLoc() const { return condExpr->getBegLoc(); }
  SourceLoc getEndLoc() const { return elseExpr->getEndLoc(); }
  SourceLoc getLoc() const { return questionLoc; }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Conditional;
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
  SourceLoc opLoc;

public:
  BinaryExpr(Expr *lhs, OpKind opKind, SourceLoc opLoc, Expr *rhs)
      : Expr(ExprKind::Binary), lhs(lhs), rhs(rhs), opLoc(opLoc) {
    bits.binaryExpr.opKind = opKind;
  }

  Expr *getLHS() const { return lhs; }
  void setLHS(Expr *lhs) { this->lhs = lhs; }

  Expr *getRHS() const { return rhs; }
  void setRHS(Expr *rhs) { this->rhs = rhs; }

  SourceLoc getOpLoc() const { return opLoc; }
  OpKind getOpKind() const { return bits.binaryExpr.opKind; }
  const char *getOpKindStr() const { return to_string(getOpKind()); }
  /// \returns the spelling of the operator (e.g. "+" for Add)
  const char *getOpSpelling() const { return sora::getSpelling(getOpKind()); }

  /// \returns true if \p op is + or -
  bool isAdditiveOp() const { return sora::isAdditiveOp(getOpKind()); }
  /// \returns true if \p op is * / or %
  bool isMultiplicativeOp() const {
    return sora::isMultiplicativeOp(getOpKind());
  }
  /// \returns true if \p op is << or >>
  bool isShiftOp() const { return sora::isShiftOp(getOpKind()); }
  /// \returns true if \p op is | & or ^
  bool isBitwiseOp() const { return sora::isBitwiseOp(getOpKind()); }
  /// \returns true if \p op is == or !=
  bool isEqualityOp() const { return sora::isEqualityOp(getOpKind()); }
  /// \returns true if \p op is < <= > or >=
  bool isRelationalOp() const { return sora::isRelationalOp(getOpKind()); }
  /// \returns true if \p op is || or &&
  bool isLogicalOp() const { return sora::isLogicalOp(getOpKind()); }
  /// \returns true if \p op is any assignement operator
  bool isAssignementOp() const { return sora::isAssignementOp(getOpKind()); }
  /// \returns true if \p op is a compound assignement operator
  bool isCompoundAssignementOp() const {
    return sora::isCompoundAssignementOp(getOpKind());
  }

  /// \returns the operator of a compound assignement. e.g. for AddAssign this
  /// returns Add.
  OpKind getOpForCompoundAssignementOp() const {
    return sora::getOpForCompoundAssignementOp(getOpKind());
  }

  SourceLoc getBegLoc() const { return lhs->getBegLoc(); }
  SourceLoc getEndLoc() const { return rhs->getEndLoc(); }
  SourceLoc getLoc() const { return opLoc; }

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
  SourceLoc opLoc;

public:
  UnaryExpr(OpKind opKind, SourceLoc opLoc, Expr *subExpr)
      : Expr(ExprKind::Unary), subExpr(subExpr), opLoc(opLoc) {
    bits.unaryExpr.opKind = opKind;
  }

  Expr *getSubExpr() const { return subExpr; }
  void setSubExpr(Expr *expr) { subExpr = expr; }

  SourceLoc getOpLoc() const { return opLoc; }
  OpKind getOpKind() const { return bits.unaryExpr.opKind; }
  const char *getOpKindStr() const { return to_string(getOpKind()); }
  /// \returns the spelling of the operator
  const char *getOpSpelling() const { return sora::getSpelling(getOpKind()); }

  SourceLoc getBegLoc() const { return opLoc; }
  SourceLoc getEndLoc() const { return subExpr->getEndLoc(); }

  static bool classof(const Expr *expr) {
    return expr->getKind() == ExprKind::Unary;
  }
};
} // namespace sora