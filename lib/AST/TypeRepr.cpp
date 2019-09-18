//===--- TypeRepr.cpp ---------------------------------------------*- C++*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/TypeRepr.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

/// Check that all TypeReprs are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.

#define TYPEREPR(ID, PARENT)                                                   \
  static_assert(std::is_trivially_destructible<ID##TypeRepr>::value,           \
                #ID "TypeRepr is not trivially destructible.");
#include "Sora/AST/TypeReprNodes.def"

void *TypeRepr::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

SourceLoc TypeRepr::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown TypeReprKind");
#define TYPEREPR(ID, PARENT)                                                   \
  case TypeReprKind::ID:                                                       \
    return ASTNodeLoc<TypeRepr, ID##TypeRepr>::getBegLoc(                      \
        cast<ID##TypeRepr>(this));
#include "Sora/AST/TypeReprNodes.def"
  }
}

SourceLoc TypeRepr::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown TypeReprKind");
#define TYPEREPR(ID, PARENT)                                                   \
  case TypeReprKind::ID:                                                       \
    return ASTNodeLoc<TypeRepr, ID##TypeRepr>::getEndLoc(                      \
        cast<ID##TypeRepr>(this));
#include "Sora/AST/TypeReprNodes.def"
  }
}

SourceLoc TypeRepr::getLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown TypeReprKind");
#define TYPEREPR(ID, PARENT)                                                   \
  case TypeReprKind::ID:                                                       \
    return ASTNodeLoc<TypeRepr, ID##TypeRepr>::getLoc(cast<ID##TypeRepr>(this));
#include "Sora/AST/TypeReprNodes.def"
  }
}

SourceRange TypeRepr::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown TypeReprKind");
#define TYPEREPR(ID, PARENT)                                                   \
  case TypeReprKind::ID:                                                       \
    return ASTNodeLoc<TypeRepr, ID##TypeRepr>::getSourceRange(                 \
        cast<ID##TypeRepr>(this));
#include "Sora/AST/TypeReprNodes.def"
  }
}

TupleTypeRepr *TupleTypeRepr::create(ASTContext &ctxt, SourceLoc lParenLoc,
                                     ArrayRef<TypeRepr *> elements,
                                     SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<TypeRepr *>(elements.size());
  void *mem = ctxt.allocate(size, alignof(TupleTypeRepr));
  return new (mem) TupleTypeRepr(lParenLoc, elements, rParenLoc);
}