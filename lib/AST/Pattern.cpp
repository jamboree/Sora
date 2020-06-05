//===--- Pattern.cpp --------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Pattern.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include <type_traits>

using namespace sora;

/// Check that all patterns are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define PATTERN(ID, PARENT)                                                    \
  static_assert(std::is_trivially_destructible<ID##Pattern>::value,            \
                #ID "Pattern is not trivially destructible.");
#include "Sora/AST/PatternNodes.def"

void *Pattern::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ArenaKind::Permanent);
}

Pattern *Pattern::ignoreParens() {
  if (auto paren = dyn_cast<ParenPattern>(this))
    return paren->getSubPattern()->ignoreParens();
  return this;
}

bool Pattern::isRefutable() const {
  bool foundRefutablePattern = false;
  const_cast<Pattern *>(this)->forEachNode([&](Pattern *pattern) {
    if (isa<RefutablePattern>(pattern))
      foundRefutablePattern = true;
  });
  return foundRefutablePattern;
}

void Pattern::forEachVarDecl(llvm::function_ref<void(VarDecl *)> fn) const {
  using Kind = PatternKind;
  switch (getKind()) {
  case Kind::Var:
    fn(cast<VarPattern>(this)->getVarDecl());
    break;
  case Kind::Discard:
    break;
  case Kind::Mut:
    if (Pattern *sub = cast<MutPattern>(this)->getSubPattern())
      sub->forEachVarDecl(fn);
    break;
  case Kind::Paren:
    if (Pattern *sub = cast<ParenPattern>(this)->getSubPattern())
      sub->forEachVarDecl(fn);
    break;
  case Kind::Tuple: {
    const TuplePattern *tuple = cast<TuplePattern>(this);
    for (Pattern *elem : tuple->getElements())
      if (elem)
        elem->forEachVarDecl(fn);
    break;
  }
  case Kind::Typed:
    if (Pattern *sub = cast<TypedPattern>(this)->getSubPattern())
      sub->forEachVarDecl(fn);
    break;
  case Kind::MaybeValue:
    if (Pattern *sub = cast<MaybeValuePattern>(this)->getSubPattern())
      sub->forEachVarDecl(fn);
    break;
  }
}

void Pattern::forEachNode(llvm::function_ref<void(Pattern *)> fn) {
  using Kind = PatternKind;
  fn(this);
  switch (getKind()) {
  case Kind::Var:
  case Kind::Discard:
    break;
  case Kind::Mut:
    if (Pattern *sub = cast<MutPattern>(this)->getSubPattern())
      sub->forEachNode(fn);
    break;
  case Kind::Paren:
    if (Pattern *sub = cast<ParenPattern>(this)->getSubPattern())
      sub->forEachNode(fn);
    break;
  case Kind::Tuple: {
    const TuplePattern *tuple = cast<TuplePattern>(this);
    for (Pattern *elem : tuple->getElements())
      if (elem)
        elem->forEachNode(fn);
    break;
  }
  case Kind::Typed:
    if (Pattern *sub = cast<TypedPattern>(this)->getSubPattern())
      sub->forEachNode(fn);
    break;
  case Kind::MaybeValue:
    if (Pattern *sub = cast<MaybeValuePattern>(this)->getSubPattern())
      sub->forEachNode(fn);
    break;
  }
}

bool Pattern::hasVarPattern() const {
  using Kind = PatternKind;
  switch (getKind()) {
  case Kind::Var:
    return true;
  case Kind::Discard:
    return false;
  case Kind::Mut:
    return cast<MutPattern>(this)->getSubPattern()->hasVarPattern();
  case Kind::Paren:
    return cast<ParenPattern>(this)->getSubPattern()->hasVarPattern();
  case Kind::Tuple:
    for (Pattern *pattern : cast<TuplePattern>(this)->getElements())
      if (pattern->hasVarPattern())
        return true;
    return false;
  case Kind::Typed:
    return cast<TypedPattern>(this)->getSubPattern()->hasVarPattern();
  case Kind::MaybeValue:
    return cast<MaybeValuePattern>(this)->getSubPattern()->hasVarPattern();
  }
}

SourceLoc Pattern::getBegLoc() const {
  switch (getKind()) {
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getBegLoc(cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
  llvm_unreachable("unknown PatternKind");
}

SourceLoc Pattern::getEndLoc() const {
  switch (getKind()) {
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getEndLoc(cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
  llvm_unreachable("unknown PatternKind");
}

SourceLoc Pattern::getLoc() const {
  switch (getKind()) {
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getLoc(cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
  llvm_unreachable("unknown PatternKind");
}

SourceRange Pattern::getSourceRange() const {
  switch (getKind()) {
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getSourceRange(                   \
        cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
  llvm_unreachable("unknown PatternKind");
}

Identifier VarPattern::getIdentifier() const {
  return varDecl->getIdentifier();
}

SourceLoc VarPattern::getBegLoc() const { return varDecl->getIdentifierLoc(); }

SourceLoc VarPattern::getEndLoc() const { return varDecl->getIdentifierLoc(); }

TuplePattern *TuplePattern::create(ASTContext &ctxt, SourceLoc lParenLoc,
                                   ArrayRef<Pattern *> patterns,
                                   SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<Pattern *>(patterns.size());
  void *mem = ctxt.allocate(size, alignof(TuplePattern));
  return new (mem) TuplePattern(lParenLoc, patterns, rParenLoc);
}

SourceLoc TypedPattern::getBegLoc() const { return subPattern->getBegLoc(); }

SourceLoc TypedPattern::getEndLoc() const { return typeRepr->getEndLoc(); }