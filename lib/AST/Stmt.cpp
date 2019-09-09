//===--- Stmt.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "ASTNodeLoc.hpp"

using namespace sora;

/// Check that all statements are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define STMT(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Stmt>::value,               \
                #ID "Stmt is not trivially destructible.");
#include "Sora/AST/StmtNodes.def"

void *Stmt::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

SourceLoc Stmt::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return Fetchloc<Stmt, ID##Stmt>::getBegLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}

SourceLoc Stmt::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return Fetchloc<Stmt, ID##Stmt>::getEndLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}
SourceRange Stmt::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return Fetchloc<Stmt, ID##Stmt>::getSourceRange(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}