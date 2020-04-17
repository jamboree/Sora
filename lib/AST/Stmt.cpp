//===--- Stmt.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Stmt.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
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
  return ctxt.allocate(size, align, ArenaKind::Permanent);
}

SourceLoc Stmt::getBegLoc() const {
  switch (getKind()) {
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return ASTNodeLoc<Stmt, ID##Stmt>::getBegLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
  llvm_unreachable("unknown StmtKind");
}

SourceLoc Stmt::getEndLoc() const {
  switch (getKind()) {
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return ASTNodeLoc<Stmt, ID##Stmt>::getEndLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
  llvm_unreachable("unknown StmtKind");
}

SourceLoc Stmt::getLoc() const {
  switch (getKind()) {
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return ASTNodeLoc<Stmt, ID##Stmt>::getLoc(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
  llvm_unreachable("unknown StmtKind");
}

SourceRange Stmt::getSourceRange() const {
  switch (getKind()) {
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return ASTNodeLoc<Stmt, ID##Stmt>::getSourceRange(cast<ID##Stmt>(this));
#include "Sora/AST/StmtNodes.def"
  }
  llvm_unreachable("unknown StmtKind");
}

SourceLoc ReturnStmt::getBegLoc() const { return returnLoc; }

SourceLoc ReturnStmt::getEndLoc() const {
  return result ? result->getEndLoc() : returnLoc;
}

SourceRange BlockStmtElement::getSourceRange() const {
  if (is<Expr *>())
    return get<Expr *>()->getSourceRange();
  if (is<Decl *>())
    return get<Decl *>()->getSourceRange();
  if (is<Stmt *>())
    return get<Stmt *>()->getSourceRange();
  llvm_unreachable("unknown node");
}

SourceLoc BlockStmtElement::getBegLoc() const {
  if (is<Expr *>())
    return get<Expr *>()->getBegLoc();
  if (is<Decl *>())
    return get<Decl *>()->getBegLoc();
  if (is<Stmt *>())
    return get<Stmt *>()->getBegLoc();
  llvm_unreachable("unknown node");
}

SourceLoc BlockStmtElement::getEndLoc() const {
  if (is<Expr *>())
    return get<Expr *>()->getEndLoc();
  if (is<Decl *>())
    return get<Decl *>()->getEndLoc();
  if (is<Stmt *>())
    return get<Stmt *>()->getEndLoc();
  llvm_unreachable("unknown node");
}

BlockStmt::BlockStmt(SourceLoc lCurlyLoc, ArrayRef<BlockStmtElement> elts,
                     SourceLoc rCurlyLoc)
    : Stmt(StmtKind::Block), lCurlyLoc(lCurlyLoc), rCurlyLoc(rCurlyLoc) {
  bits.BlockStmt.numElements = elts.size();
  assert(getNumElements() == elts.size() && "Bits dropped");
  std::uninitialized_copy(elts.begin(), elts.end(),
                          getTrailingObjects<BlockStmtElement>());
}

BlockStmt *BlockStmt::create(ASTContext &ctxt, SourceLoc lCurlyLoc,
                             ArrayRef<BlockStmtElement> elts,
                             SourceLoc rCurlyLoc) {
  auto size = totalSizeToAlloc<BlockStmtElement>(elts.size());
  void *mem = ctxt.allocate(size, alignof(BlockStmt));
  return new (mem) BlockStmt(lCurlyLoc, elts, rCurlyLoc);
}

BlockStmt *BlockStmt::createEmpty(ASTContext &ctxt, SourceLoc lCurlyLoc,
                                  SourceLoc rCurlyLoc) {
  return create(ctxt, lCurlyLoc, {}, rCurlyLoc);
}

SourceLoc StmtCondition::getBegLoc() const {
  switch (getKind()) {
  case StmtConditionKind::Expr:
    return getExpr()->getBegLoc();
  case StmtConditionKind::LetDecl:
    return getLetDecl()->getBegLoc();
  }
  llvm_unreachable("unknown condition kind");
}

SourceLoc StmtCondition::getEndLoc() const {
  switch (getKind()) {
  case StmtConditionKind::Expr:
    return getExpr()->getEndLoc();
  case StmtConditionKind::LetDecl:
    return getLetDecl()->getEndLoc();
  }
  llvm_unreachable("unknown condition kind");
}