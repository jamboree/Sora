//===--- Decl.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Decl.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

/// Check that all declarations are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define DECL(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Decl>::value,               \
                #ID "Decl is not trivially destructible.");
#include "Sora/AST/DeclNodes.def"

void *Decl::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

SourceLoc Decl::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getBegLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceLoc Decl::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getEndLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceLoc Decl::getLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceRange Decl::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getSourceRange(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

Type ValueDecl::getValueType() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ValueDecl kind");
#define VALUE_DECL(ID, PARENT)                                                 \
  case DeclKind::ID:                                                           \
    return cast<ID##Decl>(this)->getValueType();
#include "Sora/AST/DeclNodes.def"
  }
}