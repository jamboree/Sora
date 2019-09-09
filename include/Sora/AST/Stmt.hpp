//===--- Stmt.hpp - Statement ASTs -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/ASTNode.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class Expr;
class ASTContext;

/// Kinds of Statements
enum class StmtKind : uint8_t {
#define STMT(KIND, PARENT) KIND,
#include "Sora/AST/StmtNodes.def"
};

/// Base class for every Statement node.
class alignas(StmtAlignement) Stmt {
  // Disable vanilla new/delete for statements
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  StmtKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Stmt(StmtKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of statements using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Stmt));

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const;
  /// \returns the full range of this statement
  SourceRange getSourceRange() const;

  /// \return the kind of statement this is
  StmtKind getKind() const { return kind; }
};


/// Represents a simple "break" statement indicating that we can break out of
/// the innermost loop.
class BreakStmt final : public Stmt {
  SourceLoc loc;

public:
  /// \param loc the SourceLoc of the "break" keyword
  BreakStmt(SourceLoc loc) : Stmt(StmtKind::Break), loc(loc) {}

  /// \returns the SourceLoc of the "break" keyword
  SourceLoc getLoc() const { return loc; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::Break;
  }
};

/// Represents a simple "continue" statement, indicating that
/// we can skip the rest of the innermost loop and move on to the next
/// iteration (re-evaluating the condition first).
class ContinueStmt final : public Stmt {
  SourceLoc loc;

public:
  /// \param loc the SourceLoc of the "continue" keyword
  ContinueStmt(SourceLoc loc) : Stmt(StmtKind::Continue), loc(loc) {}

  /// \returns the SourceLoc of the "continue" keyword
  SourceLoc getLoc() const { return loc; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::Continue;
  }
};

/// Represents a "Block" statement, which is a group of statements (ast nodes)
/// enclosed in curly brackets.
///
/// \verbatim
/// {
///   let x = 3*3
///   x += 2
/// }
/// \endverbatim
class BlockStmt final : public Stmt,
                        private llvm::TrailingObjects<BlockStmt, ASTNode> {
  friend llvm::TrailingObjects<BlockStmt, ASTNode>;
  size_t numTrailingObjects(OverloadToken<ASTNode>) { return numElem; }

  SourceLoc lCurlyLoc, rCurlyLoc;
  size_t numElem = 0;

  BlockStmt(SourceLoc lCurlyLoc, ArrayRef<ASTNode> nodes, SourceLoc rCurlyLoc);

public:
  /// Creates a Block Stmt
  /// \param ctxt the ASTContext in which memory will be allocated
  /// \param lCurlyLoc the SourceLoc of the left (opening) curly bracket {
  /// \param nodes the elements of the block statement
  /// \param rCurlyLoc the SourceLoc of the right (closing) curly bracket }
  static BlockStmt *create(ASTContext &ctxt, SourceLoc lCurlyLoc,
                           ArrayRef<ASTNode> nodes, SourceLoc rCurlyLoc);

  /// Creates an empty Block Stmt
  /// \param ctxt the ASTContext in which memory will be allocated
  /// \param lCurlyLoc the SourceLoc of the left (opening) curly bracket {
  /// \param rCurlyLoc the SourceLoc of the right (closing) curly bracket }
  static BlockStmt *createEmpty(ASTContext &ctxt, SourceLoc lCurlyLoc,
                                SourceLoc rCurlyLoc);

  /// \returns the SourceLoc of the left (opening) curly bracket {
  SourceLoc getLeftCurlyLoc() const { return lCurlyLoc; }
  /// \returns the SourceLoc of the right (closing) curly bracket }
  SourceLoc getRightCurlyLoc() const { return rCurlyLoc; }

  /// \returns the number of elements in the BlockStmt
  size_t getNumElements() const { return numElem; }
  /// \returns a view of the array of elements
  ArrayRef<ASTNode> getElements() const;
  /// \returns the array of elements
  MutableArrayRef<ASTNode> getElements();
  /// \returns the element at index \p n
  ASTNode getElement(size_t n) const;
  /// Replaces the element at index \p n with \p node
  void setElement(size_t n, ASTNode node);

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return lCurlyLoc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const { return rCurlyLoc; }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::Block;
  }
};
} // namespace sora