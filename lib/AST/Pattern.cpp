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
  return ctxt.allocate(size, align, AllocatorKind::Permanent);
}

Pattern *Pattern::ignoreParens() {
  if (auto paren = dyn_cast<ParenPattern>(this))
    return paren->getSubPattern()->ignoreParens();
  return this;
}

void Pattern::forEachVarDecl(llvm::function_ref<void(VarDecl *)> fn) const {
  using Kind = PatternKind;
  switch (getKind()) {
  default:
  case Kind::Var:
    fn(cast<VarPattern>(this)->getVarDecl());
    break;
  case Kind::Discard:
    break;
  case Kind::Mut:
    cast<MutPattern>(this)->getSubPattern()->forEachVarDecl(fn);
    break;
  case Kind::Paren:
    cast<ParenPattern>(this)->getSubPattern()->forEachVarDecl(fn);
    break;
  case Kind::Tuple: {
    const TuplePattern *tuple = cast<TuplePattern>(this);
    for (Pattern *elem : tuple->getElements())
      elem->forEachVarDecl(fn);
    break;
  }
  case Kind::Typed:
    cast<TypedPattern>(this)->getSubPattern()->forEachVarDecl(fn);
    break;
  }
}

SourceLoc Pattern::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown PatternKind");
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getBegLoc(cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
}

SourceLoc Pattern::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown PatternKind");
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getEndLoc(cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
}

SourceLoc Pattern::getLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown PatternKind");
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getLoc(cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
}

SourceRange Pattern::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown PatternKind");
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return ASTNodeLoc<Pattern, ID##Pattern>::getSourceRange(                   \
        cast<ID##Pattern>(this));
#include "Sora/AST/PatternNodes.def"
  }
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