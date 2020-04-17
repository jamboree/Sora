//===--- Stmt.hpp - Statement ASTs -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/Common/InlineBitfields.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
class ASTWalker;
class Expr;
class LetDecl;

/// Kinds of Statements
enum class StmtKind : uint8_t {
#define STMT(KIND, PARENT) KIND,
#define STMT_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#define LAST_STMT(KIND) Last_Stmt = KIND
#include "Sora/AST/StmtNodes.def"
};

/// Base class for every Statement node.
class alignas(StmtAlignement) Stmt {
  // Disable vanilla new/delete for statements
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

protected:
  /// Number of bits needed for StmtKind
  static constexpr unsigned kindBits =
      countBitsUsed((unsigned)StmtKind::Last_Stmt);

  union Bits {
    Bits() : rawBits(0) {}
    uint64_t rawBits;

    // clang-format off

    // Stmt
    SORA_INLINE_BITFIELD_BASE(Stmt, kindBits, 
      kind : kindBits
    );

    // BlockStmt 
    SORA_INLINE_BITFIELD_FULL(BlockStmt, Stmt, 32, 
      : NumPadBits, 
      numElements : 32
    );

    // clang-format on
  } bits;
  static_assert(sizeof(Bits) == 8, "Bits is too large!");

  // Derived classes should be able to use placement new, as it is needed for
  // classes with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Stmt(StmtKind kind) { bits.Stmt.kind = (uint64_t)kind; }

public:
  // Publicly allow allocation of statements using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Stmt));

  /// Traverse this Stmt using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
  bool walk(ASTWalker &&walker) { return walk(walker); }

  /// Dumps this statement to \p out
  void dump(raw_ostream &out, const SourceManager &srcMgr,
            unsigned indent = 2) const;

  /// \return the kind of statement this is
  StmtKind getKind() const { return StmtKind(bits.Stmt.kind); }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const;
  /// \returns the preffered SourceLoc for diagnostics. This is defaults to
  /// getBegLoc but nodes can override it as they please.
  SourceLoc getLoc() const;
  /// \returns the full range of this statement
  SourceRange getSourceRange() const;
};

/// Stmt should only be one pointer in size (kind + padding bits)
static_assert(sizeof(Stmt) <= 8, "Stmt is too large!");

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
  ContinueStmt(SourceLoc loc) : Stmt(StmtKind::Continue), loc(loc) {}

  SourceLoc getLoc() const { return loc; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return loc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::Continue;
  }
};

/// Represents a "return" statement
///
/// \verbatim
///   return
///   return 0
/// \endverbatim
class ReturnStmt : public Stmt {
  SourceLoc returnLoc;
  Expr *result = nullptr;

public:
  ReturnStmt(SourceLoc returnLoc, Expr *result = nullptr)
      : Stmt(StmtKind::Return), returnLoc(returnLoc), result(result) {}

  SourceLoc getReturnLoc() const { return returnLoc; }

  bool hasResult() const { return (bool)result; }
  Expr *getResult() const { return result; }
  void setResult(Expr *expr) { this->result = expr; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const;

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::Return;
  }
};

/// An element of a block statement.
struct BlockStmtElement : public llvm::PointerUnion<Expr *, Stmt *, Decl *> {
  using Base = llvm::PointerUnion<Expr *, Stmt *, Decl *>;

  using llvm::PointerUnion<Expr *, Stmt *, Decl *>::PointerUnion;

  SourceRange getSourceRange() const;
  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  /// Traverse this element using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
};

/// Represents a "Block" statement, which is a group of declaration, expression
/// and/or statements enclosed in curly brackets.
///
/// \verbatim
/// {
///   let x = 3*3
///   x += 2
/// }
/// \endverbatim
class BlockStmt final
    : public Stmt,
      private llvm::TrailingObjects<BlockStmt, BlockStmtElement> {
  friend llvm::TrailingObjects<BlockStmt, BlockStmtElement>;

  SourceLoc lCurlyLoc, rCurlyLoc;

  BlockStmt(SourceLoc lCurlyLoc, ArrayRef<BlockStmtElement> nodes,
            SourceLoc rCurlyLoc);

public:
  static BlockStmt *create(ASTContext &ctxt, SourceLoc lCurlyLoc,
                           ArrayRef<BlockStmtElement> elts,
                           SourceLoc rCurlyLoc);

  static BlockStmt *createEmpty(ASTContext &ctxt, SourceLoc lCurlyLoc,
                                SourceLoc rCurlyLoc);

  SourceLoc getLeftCurlyLoc() const { return lCurlyLoc; }
  SourceLoc getRightCurlyLoc() const { return rCurlyLoc; }

  size_t getNumElements() const { return (size_t)bits.BlockStmt.numElements; }

  ArrayRef<BlockStmtElement> getElements() const {
    return {getTrailingObjects<BlockStmtElement>(), getNumElements()};
  }

  MutableArrayRef<BlockStmtElement> getElements() {
    return {getTrailingObjects<BlockStmtElement>(), getNumElements()};
  }

  BlockStmtElement getElement(size_t n) const { return getElements()[n]; }
  void setElement(size_t n, BlockStmtElement elt) { getElements()[n] = elt; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return lCurlyLoc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const { return rCurlyLoc; }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::Block;
  }
};

/// Kinds of StmtConditions
enum class StmtConditionKind {
  /// An expression condition, e.g. 'foo || bar' in `'if foo || bar'
  Expr,
  /// A declaration condition, e.g. 'let x = x' in 'if let x = x'
  LetDecl
};

/// Represents the condition of a loop or an if statement.
/// This can be a boolean expression, or a let-declaration which can be used to
/// create conditional bindings for 'maybe' types.
///
/// This is intended to be a thin pointer-sized wrapper around a PointerUnion,
/// providing convenience methods like getKind(), getBegLoc() & getEndLoc().
class StmtCondition final {
  llvm::PointerUnion<Expr *, LetDecl *> cond;

public:
  /// Creates a null (empty) condition
  StmtCondition() : StmtCondition(nullptr) {}

  /// Creates a null (empty) condition
  /*implicit*/ StmtCondition(std::nullptr_t) : cond(nullptr) {}

  /// Creates an expression condition
  /*implicit*/ StmtCondition(Expr *expr) : cond(expr) {}

  /// Creates a let-declaration condition
  /*implicit*/ StmtCondition(LetDecl *decl) : cond(decl) {}

  bool isNull() const { return cond.isNull(); }

  /// \returns true if this condition is not null
  explicit operator bool() const { return !cond.isNull(); }

  /// \returns the kind of StmtCondition this is
  StmtConditionKind getKind() const {
    return cond.is<Expr *>() ? StmtConditionKind::Expr
                             : StmtConditionKind::LetDecl;
  }

  /// \returns the Expr* or nullptr.
  Expr *getExprOrNull() const { return cond.dyn_cast<Expr *>(); }
  /// \returns the Expr*. Can only be used if this is an Expr.
  Expr *getExpr() const { return cond.get<Expr *>(); }

  /// \returns the LetDecl* or nullptr.
  LetDecl *getLetDeclOrNull() const { return cond.dyn_cast<LetDecl *>(); }
  /// \returns the LetDecl*. Can only be used if this is an LetDecl.
  LetDecl *getLetDecl() const { return cond.get<LetDecl *>(); }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;
};

/// Common base class for conditional statements ("if-then-else" & "while")
class ConditionalStmt : public Stmt {
  StmtCondition cond;

protected:
  ConditionalStmt(StmtKind kind, StmtCondition cond) : Stmt(kind), cond(cond) {}

public:
  StmtCondition getCond() const { return cond; }
  void setCond(StmtCondition cond) { this->cond = cond; }

  static bool classof(const Stmt *stmt) {
    return (stmt->getKind() >= StmtKind::First_Conditional) &&
           (stmt->getKind() <= StmtKind::Last_Conditional);
  }
};

/// Represents an "if-then-else" statement
///
/// \verbatim
/// if cond {
///   /* do something */
/// }
///
/// if cond {
///   /* do something */
/// }
/// else {
///   /* do something */
/// }
///
/// if cond {
///   /* do something */
/// }
/// else if cond {
///   /* do something */
/// }
///
/// /*etc*/
/// \endverbatim
class IfStmt final : public ConditionalStmt {
  BlockStmt *thenStmt = nullptr;
  Stmt *elseStmt = nullptr;
  SourceLoc ifLoc;
  SourceLoc elseLoc;

public:
  IfStmt(SourceLoc ifLoc, StmtCondition cond, BlockStmt *thenStmt,
         SourceLoc elseLoc, Stmt *elseStmt)
      : ConditionalStmt(StmtKind::If, cond), thenStmt(thenStmt),
        elseStmt(elseStmt), ifLoc(ifLoc), elseLoc(elseLoc) {}

  IfStmt(SourceLoc ifLoc, StmtCondition cond, BlockStmt *thenStmt)
      : IfStmt(ifLoc, cond, thenStmt, SourceLoc(), nullptr) {}

  SourceLoc getIfLoc() const { return ifLoc; }
  SourceLoc getElseLoc() const { return elseLoc; }

  BlockStmt *getThen() const { return thenStmt; }
  void setThen(BlockStmt *stmt) { thenStmt = stmt; }

  Stmt *getElse() const { return elseStmt; }
  void setElse(Stmt *stmt) { elseStmt = stmt; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return ifLoc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const {
    return elseStmt ? elseStmt->getEndLoc() : thenStmt->getEndLoc();
  }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::If;
  }
};

/// Represents a "while" loop
///
/// \verbatim
/// while cond {
///   /* do something */
/// }
/// \endverbatim
class WhileStmt final : public ConditionalStmt {
  SourceLoc whileLoc;
  BlockStmt *body = nullptr;

public:
  WhileStmt(SourceLoc whileLoc, StmtCondition cond, BlockStmt *body)
      : ConditionalStmt(StmtKind::While, cond), whileLoc(whileLoc), body(body) {
  }

  SourceLoc getWhileLoc() const { return whileLoc; }

  BlockStmt *getBody() const { return body; }
  void setBody(BlockStmt *body) { this->body = body; }

  /// \returns the SourceLoc of the first token of the statement
  SourceLoc getBegLoc() const { return whileLoc; }
  /// \returns the SourceLoc of the last token of the statement
  SourceLoc getEndLoc() const { return body->getEndLoc(); }

  static bool classof(const Stmt *stmt) {
    return stmt->getKind() == StmtKind::While;
  }
};
} // namespace sora