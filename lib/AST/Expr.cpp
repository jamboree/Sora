//===--- Expr.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Expr.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
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

SourceLoc Expr::getBegLoc() const {
  switch (getKind()) {
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getBegLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
  llvm_unreachable("unknown ExprKind");
}

SourceLoc Expr::getEndLoc() const {
  switch (getKind()) {
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getEndLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
  llvm_unreachable("unknown ExprKind");
}

SourceLoc Expr::getLoc() const {
  switch (getKind()) {
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getLoc(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
  llvm_unreachable("unknown ExprKind");
}
SourceRange Expr::getSourceRange() const {
  switch (getKind()) {
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return ASTNodeLoc<Expr, ID##Expr>::getSourceRange(cast<ID##Expr>(this));
#include "Sora/AST/ExprNodes.def"
  }
  llvm_unreachable("unknown ExprKind");
}

void *Expr::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ArenaKind::Permanent);
}

Expr *Expr::ignoreParens() {
  if (ParenExpr *paren = dyn_cast<ParenExpr>(this))
    return paren->getSubExpr()->ignoreParens();
  return this;
}

Expr *Expr::ignoreImplicitConversions() {
  if (ImplicitConversionExpr *conv = dyn_cast<ImplicitConversionExpr>(this))
    return conv->getSubExpr()->ignoreImplicitConversions();
  return this;
}

void *UnresolvedExpr::operator new(size_t size, ASTContext &ctxt,
                                   unsigned align) {
  return ctxt.allocate(size, align, ArenaKind::UnresolvedExpr);
}

Identifier DeclRefExpr::getIdentifier() const { return decl->getIdentifier(); }

DeclRefExpr::DeclRefExpr(UnresolvedDeclRefExpr *udre, ValueDecl *decl)
    : DeclRefExpr(udre->getIdentifierLoc(), decl) {
  assert(udre->getIdentifier() == decl->getIdentifier() &&
         "Incorrect Resolution!");
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

TupleExpr *TupleExpr::create(ASTContext &ctxt, SourceLoc lParenLoc,
                             ArrayRef<Expr *> exprs, SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<Expr *>(exprs.size());
  void *mem = ctxt.allocate(size, alignof(TupleExpr));
  return new (mem) TupleExpr(lParenLoc, exprs, rParenLoc);
}

CallExpr *CallExpr::create(ASTContext &ctxt, Expr *fn, SourceLoc lParen,
                           ArrayRef<Expr *> args, SourceLoc rParen) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<Expr *>(args.size());
  void *mem = ctxt.allocate(size, alignof(CallExpr));
  return new (mem) CallExpr(fn, lParen, args, rParen);
}