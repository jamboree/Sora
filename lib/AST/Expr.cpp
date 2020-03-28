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
#include "Sora/AST/Types.hpp"
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
  IntegerWidth::Status status;
  APInt result = IntegerWidth::arbitrary().parse(
      getString(), /*isNegative*/ false, 0, &status);
  assert(status != IntegerWidth::Status::Error &&
         "Integer Parsing Error - Ill-formed integer token?");
  return result;
}

APInt IntegerLiteralExpr::getValue() const {
  Type type = getType();
  assert(type && type->is<IntegerType>());
  IntegerWidth intWidth = type->castTo<IntegerType>()->getWidth();

  IntegerWidth::Status status;
  APInt result = intWidth.parse(getString(), /*isNegative*/ false, 0, &status);
  assert(status != IntegerWidth::Status::Error &&
         "Integer Parsing Error - Ill-formed integer token?");
  return result;
}

APFloat FloatLiteralExpr::getValue() const {
  Type type = getType();
  assert(type && type->is<FloatType>());
  return APFloat(type->castTo<FloatType>()->getAPFloatSemantics(), getString());
}

TupleExpr *TupleExpr::create(ASTContext &ctxt, SourceLoc lParenLoc,
                             ArrayRef<Expr *> exprs, SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<Expr *>(exprs.size());
  void *mem = ctxt.allocate(size, alignof(TupleExpr));
  return new (mem) TupleExpr(lParenLoc, exprs, rParenLoc);
}

SourceLoc TupleExpr::getBegLoc() const {
  if (lParenLoc.isValid())
    return lParenLoc;
  // If we have no '(' loc, use the first valid beg loc in the elements array
  for (Expr *expr : getElements())
    if (SourceLoc loc = expr->getBegLoc())
      return loc;
  return {};
}

SourceLoc TupleExpr::getEndLoc() const {
  if (rParenLoc.isValid())
    return rParenLoc;
  // If we have no ')' loc, use the last valid end loc in the elements array
  auto elems = getElements();
  for (auto it = elems.rbegin(); it != elems.rend(); ++it)
    if (SourceLoc loc = (*it)->getEndLoc())
      return loc;
  return {};
}

CallExpr *CallExpr::create(ASTContext &ctxt, Expr *fn, SourceLoc lParen,
                           ArrayRef<Expr *> args, SourceLoc rParen) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<Expr *>(args.size());
  void *mem = ctxt.allocate(size, alignof(CallExpr));
  return new (mem) CallExpr(fn, lParen, args, rParen);
}