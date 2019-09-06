//===--- Expr.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Expr.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include <type_traits>

using namespace sora;

/// Check that all expressions are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define EXPR(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Expr>::value,               \
                #ID "Expr is not trivially destructible.");
#include "Sora/AST/ExprNodes.def"

namespace {
template <typename Rtr, typename Class>
constexpr bool isOverridenFromExpr(Rtr (Class::*)() const) {
  return true;
}

template <typename Rtr>
constexpr bool isOverridenFromExpr(Rtr (Expr::*)() const) {
  return false;
}

/// Expressions can override (getSourceRange) or (getBegLoc & getEndLoc) or
/// both. We adapt automatically based on what's available.
template <typename Ty> struct ExprFetchLoc {
  static constexpr bool hasGetRange = isOverridenFromExpr(&Ty::getSourceRange);
  static constexpr bool hasGetBeg = isOverridenFromExpr(&Ty::getBegLoc);
  static constexpr bool hasGetEnd = isOverridenFromExpr(&Ty::getEndLoc);

  static_assert(hasGetRange || (hasGetBeg && hasGetEnd),
                "Statements must override (getSourceRange) or "
                "(getBegLoc/getEndLoc) or both.");

  static SourceRange getSourceRange(const Ty *expr) {
    if (hasGetRange)
      return expr->getSourceRange();
    return {expr->getBegLoc(), expr->getEndLoc()};
  }

  static SourceLoc getBegLoc(const Ty *expr) {
    if (hasGetBeg)
      return expr->getBegLoc();
    return expr->getSourceRange().begin;
  }

  static SourceLoc getEndLoc(const Ty *expr) {
    if (hasGetEnd)
      return expr->getEndLoc();
    return expr->getSourceRange().end;
  }
};
} // namespace

SourceLoc Expr::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ExprFetchLoc<ID##Expr>::getBegLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
}

SourceLoc Expr::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ExprFetchLoc<ID##Expr>::getEndLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
}
SourceRange Expr::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ExprFetchLoc<ID##Expr>::getSourceRange(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
}

void *Expr::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

Expr *Expr::ignoreParens() {
  if (ParenExpr *paren = dyn_cast<ParenExpr>(this))
    return paren->getSubExpr()->ignoreParens();
  return this;
}

void *UnresolvedExpr::operator new(size_t size, ASTContext &ctxt,
                                   unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::UnresolvedNodes);
}

APInt IntegerLiteralExpr::getRawValue() const {
  APInt result;
  /// For now Sora only has base 10 literals.
  unsigned radix = 10;
  /// Parse it (true = error)
  if (strValue.getAsInteger(radix, result))
    llvm_unreachable("Integer Parsing Error - Ill-formed integer token?");
  return result;
}

TupleExpr::TupleExpr(SourceLoc lParenLoc, ArrayRef<Expr *> exprs,
                     ArrayRef<SourceLoc> locs, SourceLoc rParenLoc)
    : Expr(ExprKind::Tuple), lParenLoc(lParenLoc), rParenLoc(rParenLoc),
      numElements(exprs.size()) {
  assert(exprs.size() ? (locs.size() == (exprs.size() - 1)) : true);
  std::uninitialized_copy(exprs.begin(), exprs.end(),
                          getTrailingObjects<Expr *>());
  std::uninitialized_copy(locs.begin(), locs.end(),
                          getTrailingObjects<SourceLoc>());
}

TupleExpr *TupleExpr::create(ASTContext &ctxt, SourceLoc lParenLoc,
                             ArrayRef<Expr *> exprs, ArrayRef<SourceLoc> locs,
                             SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<Expr *, SourceLoc>(exprs.size(), locs.size());
  void *mem = ctxt.allocate(size, alignof(TupleExpr));
  return new (mem) TupleExpr(lParenLoc, exprs, locs, rParenLoc);
}

TupleExpr *TupleExpr::createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                  SourceLoc rParenLoc) {
  return create(ctxt, lParenLoc, {}, {}, rParenLoc);
}

MutableArrayRef<Expr *> TupleExpr::getElements() {
  return {getTrailingObjects<Expr *>(), getNumElements()};
}

ArrayRef<Expr *> TupleExpr::getElements() const {
  return {getTrailingObjects<Expr *>(), getNumElements()};
}

Expr *TupleExpr::getElement(size_t n) { return getElements()[n]; }

SourceLoc TupleExpr::getCommaLoc(size_t n) const { return getCommaLocs()[n]; }

ArrayRef<SourceLoc> TupleExpr::getCommaLocs() const {
  return {getTrailingObjects<SourceLoc>(), getNumCommas()};
}