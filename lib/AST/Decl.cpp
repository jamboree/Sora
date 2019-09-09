//===--- Decl.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Decl.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

/// Check that all declarations are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define DECL(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Decl>::value,               \
                #ID "Decl is not trivially destructible.");
#include "Sora/AST/DeclNodes.def"

namespace {
template <typename Rtr, typename Class>
constexpr bool isOverridenFromDecl(Rtr (Class::*)() const) {
  return true;
}

template <typename Rtr>
constexpr bool isOverridenFromDecl(Rtr (Decl::*)() const) {
  return false;
}

/// Declarations can override (getSourceRange) or (getBegLoc & getEndLoc) or
/// both. We adapt automatically based on what's available.
template <typename Ty> struct DeclFetchLoc {
  static constexpr bool hasGetRange = isOverridenFromDecl(&Ty::getSourceRange);
  static constexpr bool hasGetBeg = isOverridenFromDecl(&Ty::getBegLoc);
  static constexpr bool hasGetEnd = isOverridenFromDecl(&Ty::getEndLoc);

  static_assert(hasGetRange || (hasGetBeg && hasGetEnd),
                "Declarations must override (getSourceRange) or "
                "(getBegLoc/getEndLoc) or both.");

  static SourceRange getSourceRange(const Ty *decl) {
    if (hasGetRange)
      return decl->getSourceRange();
    return {decl->getBegLoc(), stmt->getEndLoc()};
  }

  static SourceLoc getBegLoc(const Ty *decl) {
    if (hasGetBeg)
      return decl->getBegLoc();
    return decl->getSourceRange().begin;
  }

  static SourceLoc getEndLoc(const Ty *decl) {
    if (hasGetEnd)
      return decl->getEndLoc();
    return decl->getSourceRange().end;
  }
};
} // namespace

void *Decl::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

SourceLoc Decl::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return DeclFetchLoc<ID##Decl>::getBegLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceLoc Decl::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return DeclFetchLoc<ID##Decl>::getEndLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}
SourceRange Decl::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return DeclFetchLoc<ID##Decl>::getSourceRange(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}