//===--- Expr.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Expr.hpp"
#include "ASTNodeLoc.hpp"
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

SourceLoc Expr::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getBegLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
}

SourceLoc Expr::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getEndLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
}

SourceLoc Expr::getLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
}
SourceRange Expr::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getSourceRange(cast<ID##Expr>(this));
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
                     ArrayRef<SourceLoc> commaLocs, SourceLoc rParenLoc)
    : Expr(ExprKind::Tuple), lParenLoc(lParenLoc), rParenLoc(rParenLoc),
      numElements(exprs.size()), numCommas(commaLocs.size()) {
  assert((exprs.size() ? (commaLocs.size() == (exprs.size() - 1)) : true) &&
         "There must be N expressions and N-1 comma Source locations (or 0 of "
         "both)");
  std::uninitialized_copy(exprs.begin(), exprs.end(),
                          getTrailingObjects<Expr *>());
  std::uninitialized_copy(commaLocs.begin(), commaLocs.end(),
                          getTrailingObjects<SourceLoc>());
}

TupleExpr *TupleExpr::create(ASTContext &ctxt, SourceLoc lParenLoc,
                             ArrayRef<Expr *> exprs,
                             ArrayRef<SourceLoc> commaLocs,
                             SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size =
      totalSizeToAlloc<Expr *, SourceLoc>(exprs.size(), commaLocs.size());
  void *mem = ctxt.allocate(size, alignof(TupleExpr));
  return new (mem) TupleExpr(lParenLoc, exprs, commaLocs, rParenLoc);
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

void TupleExpr::setElement(size_t n, Expr *expr) { getElements()[n] = expr; }

#include <llvm/Support/raw_ostream.h>

SourceLoc TupleExpr::getCommaLocForExpr(Expr *expr) const {
#ifndef NDEBUG
  bool found = false;
  for (auto elem : getElements()) {
    if (elem == expr) {
      found = true;
      break;
    }
  }
  assert(found && "Expr does not belong to this Tuple!");
#endif
  SourceLoc exprEnd = expr->getEndLoc();
  assert(exprEnd && "invalid end loc for expr");
  for (auto comma : getCommaLocs()) {
    assert(comma && "invalid comma loc!");
    if (comma > exprEnd)
      return comma;
  }
  return SourceLoc();
}

ArrayRef<SourceLoc> TupleExpr::getCommaLocs() const {
  return {getTrailingObjects<SourceLoc>(), getNumCommas()};
}