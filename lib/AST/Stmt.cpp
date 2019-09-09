//===--- Stmt.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

/// Check that all statements are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define STMT(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Stmt>::value,               \
                #ID "Stmt is not trivially destructible.");
#include "Sora/AST/StmtNodes.def"

namespace {
template <typename Rtr, typename Class>
constexpr bool isOverridenFromStmt(Rtr (Class::*)() const) {
  return true;
}

template <typename Rtr>
constexpr bool isOverridenFromStmt(Rtr (Stmt::*)() const) {
  return false;
}

/// Statements can override (getSourceRange) or (getBegLoc & getEndLoc) or both.
/// We adapt automatically based on what's available.
template <typename Ty> struct StmtFetchLoc {
  static constexpr bool hasGetRange = isOverridenFromStmt(&Ty::getSourceRange);
  static constexpr bool hasGetBeg = isOverridenFromStmt(&Ty::getBegLoc);
  static constexpr bool hasGetEnd = isOverridenFromStmt(&Ty::getEndLoc);

  static_assert(hasGetRange || (hasGetBeg && hasGetEnd),
                "Statements must override (getSourceRange) or "
                "(getBegLoc/getEndLoc) or both.");

  static SourceRange getSourceRange(const Ty *stmt) {
    if (hasGetRange)
      return stmt->getSourceRange();
    return {stmt->getBegLoc(), stmt->getEndLoc()};
  }

  static SourceLoc getBegLoc(const Ty *stmt) {
    if (hasGetBeg)
      return stmt->getBegLoc();
    return stmt->getSourceRange().begin;
  }

  static SourceLoc getEndLoc(const Ty *stmt) {
    if (hasGetEnd)
      return stmt->getEndLoc();
    return stmt->getSourceRange().end;
  }
};
} // namespace

void *Stmt::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

SourceLoc Stmt::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return StmtFetchLoc<ID##Stmt>::getBegLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}

SourceLoc Stmt::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return StmtFetchLoc<ID##Stmt>::getEndLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}
SourceRange Stmt::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return StmtFetchLoc<ID##Stmt>::getSourceRange(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}