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
} // namespace

#define DISPATCH_OVERRIDEN(CLASSNAME, METHOD)                                  \
  static_assert(isOverridenFromExpr(&CLASSNAME::METHOD),                       \
                #CLASSNAME " does not override Expr::" #METHOD);               \
  return static_cast<const CLASSNAME *>(this)->METHOD()

// FIXME: Rework this so it always use getSourceRange or getStartLoc/getEndLoc.
// It should auto-adapt (support either getSourceRange or the
// getStartLoc/getEndLoc combo). Once that's done change ErrorExpr to remove
// getStartLoc/getEndLoc.

SourceLoc Expr::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    DISPATCH_OVERRIDEN(ID##Expr, getBegLoc);
#include "Sora/AST/ExprNodes.def"
  }
}

SourceLoc Expr::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    DISPATCH_OVERRIDEN(ID##Expr, getEndLoc);
#include "Sora/AST/ExprNodes.def"
  }
}
SourceRange Expr::getSourceRange() const {
  // For getSourceRange, we use the getSourceRange of the derived expression
  // if it reimplements it, else we simply create the SourceRange using
  // getBegLoc & getEndLoc
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return isOverridenFromExpr(&ID##Expr::getSourceRange)                      \
               ? static_cast<const ID##Expr *>(this)->getSourceRange()         \
               : SourceRange(getBegLoc(), getEndLoc());
#include "Sora/AST/ExprNodes.def"
  }
}
#undef DISPATCH_OVERRIDEN

void *Expr::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
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
