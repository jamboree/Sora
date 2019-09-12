//===--- Stmt.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Stmt.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Expr.hpp"
#include "llvm/ADT/ArrayRef.h"

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
    return ASTNodeLoc<Stmt, ID##Stmt>::getBegLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}

SourceLoc Stmt::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return ASTNodeLoc<Stmt, ID##Stmt>::getEndLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}
SourceRange Stmt::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return ASTNodeLoc<Stmt, ID##Stmt>::getSourceRange(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
}

SourceLoc ReturnStmt::getBegLoc() const { return returnLoc; }

SourceLoc ReturnStmt::getEndLoc() const {
  return result ? result->getEndLoc() : returnLoc;
}

BlockStmt::BlockStmt(SourceLoc lCurlyLoc, ArrayRef<ASTNode> nodes,
                     SourceLoc rCurlyLoc)
    : Stmt(StmtKind::Block), lCurlyLoc(lCurlyLoc), rCurlyLoc(rCurlyLoc),
      numElem(nodes.size()) {
  std::uninitialized_copy(nodes.begin(), nodes.end(),
                          getTrailingObjects<ASTNode>());
}

BlockStmt *BlockStmt::create(ASTContext &ctxt, SourceLoc lCurlyLoc,
                             ArrayRef<ASTNode> nodes, SourceLoc rCurlyLoc) {
  auto size = totalSizeToAlloc<ASTNode>(nodes.size());
  void *mem = ctxt.allocate(size, alignof(BlockStmt));
  return new (mem) BlockStmt(lCurlyLoc, nodes, rCurlyLoc);
}

BlockStmt *BlockStmt::createEmpty(ASTContext &ctxt, SourceLoc lCurlyLoc,
                                  SourceLoc rCurlyLoc) {
  return create(ctxt, lCurlyLoc, {}, rCurlyLoc);
}

ArrayRef<ASTNode> BlockStmt::getElements() const {
  return {getTrailingObjects<ASTNode>(), numElem};
}

MutableArrayRef<ASTNode> BlockStmt::getElements() {
  return {getTrailingObjects<ASTNode>(), numElem};
}

ASTNode BlockStmt::getElement(size_t n) const { return getElements()[n]; }

void BlockStmt::setElement(size_t n, ASTNode node) { getElements()[n] = node; }

SourceLoc StmtCondition::getBegLoc() const {
  if (isExpr())
    return expr->getBegLoc();
  llvm_unreachable("unknown condition kind");
}

SourceLoc StmtCondition::getEndLoc() const {
  if (isExpr())
    return expr->getEndLoc();
  llvm_unreachable("unknown condition kind");
}