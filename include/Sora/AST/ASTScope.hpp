//===--- ASTScope.hpp - AST Scope -------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/ASTNode.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/PointerIntpair.h"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class ASTContext;
class ASTScope;
class FuncDecl;
class BlockStmt;
class IfStmt;
class WhileStmt;
class LetDecl;
class SourceFile;

/// Kinds of ASTScope
enum class ASTScopeKind : uint8_t {
  /// SourceFile
  SourceFile,
  /// Decl
  FuncDecl,
  LocalLetDecl,
  /// Stmt
  BlockStmt,
  IfStmt,
  WhileStmt,

  Last_Kind = WhileStmt
};

/// Base class for AST Scopes.
///
/// ASTScopes form a tree and represent every scope in the AST in some way, and
/// the root is always a SourceFileScope.
class alignas(ASTScopeAlignement) ASTScope {
  // ASTScopes shouldn't be copyable
  ASTScope(const ASTScope &) = delete;
  ASTScope &operator=(const ASTScope &) = delete;

  // Disable vanilla new/delete for ASTScopes
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;
  // Disable placement new as well
  void *operator new(size_t, void *mem) noexcept = delete;

  // The parent ASTScope, or the ASTContext for root ASTScopes.
  static_assert(
      unsigned(ASTScopeKind::Last_Kind) <= (1 << 3),
      "Not enough bits in parentAndKind to represent every ASTScopeKind");
  llvm::PointerIntPair<ASTScope *, 3, ASTScopeKind> parentAndKind;
  SmallVector<ASTScope *, 4> children;
  bool hasCleanup = false;
  /// Whether this scope has built its children scope.
  bool expanded = true;

protected:
  /// \returns true if this ASTScope needs cleanup
  bool needsCleanup() const {
    const char *beg = reinterpret_cast<const char *>(this);
    const char *end = beg + sizeof(children);
    const char *data = reinterpret_cast<const char *>(children.data());
    // If the SmallVector didn't allocate any data, it doesn't need cleanups.
    // FIXME: Using .isSmall() would be so much cleaner, but we don't have
    // access to it.
    if ((beg <= data) && (data <= end)) {
      assert(children.size() <= 4);
      return true;
    }
    return false;
  }

  ASTScope(ASTScopeKind kind, ASTScope *parent) : parentAndKind(parent, kind) {
    assert((parent || (kind == ASTScopeKind::SourceFile)) &&
           "Every scope except SourceFileScopes must have a parent");
  }

  // Allow allocation of ASTScopes using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(ASTScope));

public:
  /// \returns the kind of scope this is
  ASTScopeKind getKind() const { return parentAndKind.getInt(); }

  void addChild(ASTScope *scope);
  ArrayRef<ASTScope *> getChildren() const { return children; }
  /// \param canExpand if true, the ASTScope will expand if needed.
  /// \returns the children of this ASTScope
  MutableArrayRef<ASTScope *> getChildren(bool canExpand = false) {
    if (expanded && canExpand)
      expand();
    return children;
  }

  /// \returns the ASTContext in which this scope has been allocated
  ASTContext &getASTContext() const;

  /// \returns the parent of this ASTScope.
  /// This returns null for SourceFileScopes.
  ASTScope *getParent() const { return parentAndKind.getPointer(); }

  /// \returns true if this ASTScope has been expanded
  bool isExpanded() const { return expanded; }

  /// Expands this ASTScope if it hasn't been done yet, building its children
  /// scope.
  void expand();

  /// Recursively expand the scope until the full ASTScope tree has been
  /// generated. This is expensive, and should be used only for testing
  /// purposes.
  void fullyExpand();

  /// \returns true if this ASTScope represents a FuncDecl's scope.
  bool isFuncDecl() const;

  /// Dumps this ASTScope to \p out
  void dump(raw_ostream &out, unsigned indent = 2) const {
    dumpImpl(out, indent, 0);
  }

private:
  void dumpImpl(raw_ostream &out, unsigned indent, unsigned curIndent) const;
};

/// Represents the scope of a SourceFile
class SourceFileScope final : public ASTScope {
  SourceFileScope(SourceFile &file)
      : ASTScope(ASTScopeKind::SourceFile, nullptr), sourceFile(file) {}

  SourceFile &sourceFile;

public:
  static SourceFileScope *create(SourceFile &sf);

  SourceFile &getSourceFile() const { return sourceFile; }

  static bool classof(ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::SourceFile;
  }
};

/// Represents the scope of a FuncDecl
class FuncDeclScope final : public ASTScope {
  FuncDeclScope(FuncDecl *decl, ASTScope *parent)
      : ASTScope(ASTScopeKind::FuncDecl, parent), decl(decl) {
    assert(parent && "FuncDeclScope must have a valid parent");
    assert(decl && "FuncDeclScope must have a non-null FuncDecl*");
  }

  FuncDecl *const decl;

public:
  static FuncDeclScope *create(FuncDecl *decl, ASTScope *parent);

  FuncDecl *getFuncDecl() const { return decl; }

  static bool classof(ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::FuncDecl;
  }
};

/// Represents the scope of a local LetDecl.
/// This is needed because order matters inside the body of functions.
/// For example, the following code
/// \verbatim
///   func foo() {
///     let (a, b)
///     if false {}
///     if true [}
///     let c
///     let d
///     { a }
///   }
/// \endverbatim
///
/// will have this scope map
///
/// \verbatim
///   FuncDeclScope
///     LocalLetDeclScope
///       IfStmtScope
///       IfStmtScope
///       LocalLetDeclScope
///         LocalLetDeclScope
///           BraceStmtScope
/// \endverbatim
///
/// Note that LocalLetDecls implicitly introduce a new scope.
///
/// That way, when we perform lookup for 'a', we correctly look at all of the
/// 'let's (as they are the parents of the BraceStmtScope)
class LocalLetDeclScope final : public ASTScope {
  LocalLetDeclScope(LetDecl *decl, ASTScope *parent)
      : ASTScope(ASTScopeKind::LocalLetDecl, parent), decl(decl) {
    assert(parent && "LocalLetDeclScope must have a valid parent");
    assert(isLocalAndNonNull() &&
           "LocalLetDeclScope must have a non-null, local LetDecl*");
  }

  bool isLocalAndNonNull() const;

  LetDecl *const decl;

public:
  static LocalLetDeclScope *create(LetDecl *decl, ASTScope *parent);

  LetDecl *getLetDecl() const { return decl; }

  static bool classof(ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::LocalLetDecl;
  }
};

/// Represents the scope of a BlockStmt.
/// This can have any number of children and may have a continuation.
class BlockStmtScope final : public ASTScope {
  BlockStmtScope(BlockStmt *stmt, ASTScope *parent)
      : ASTScope(ASTScopeKind::BlockStmt, parent), stmt(stmt) {
    assert(parent && "BlockStmtScope must have a valid parent");
    assert(stmt && "BlockStmtScope must have a non-null BlockStmt*");
  }

  BlockStmt *const stmt;

public:
  static BlockStmtScope *create(ASTContext &ctxt, BlockStmt *stmt,
                                ASTScope *parent);

  BlockStmt *getBlockStmt() const { return stmt; }

  static bool classof(ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::BlockStmt;
  }
};

/// Represents the scope of a IfStmt
/// This can have at most 2 children scopes: the body of the if, and, perhaps,
/// the body of the else.
class IfStmtScope final : public ASTScope {
  IfStmtScope(IfStmt *stmt, ASTScope *parent)
      : ASTScope(ASTScopeKind::BlockStmt, parent), stmt(stmt) {
    assert(parent && "IfStmtScope must have a valid parent");
    assert(stmt && "IfStmtScope must have a non-null BlockStmt*");
  }

  IfStmt *const stmt;

public:
  static IfStmtScope *create(ASTContext &ctxt, IfStmt *stmt, ASTScope *parent);

  IfStmt *getIfStmt() const { return stmt; }

  static bool classof(ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::BlockStmt;
  }
};

/// Represents the scope of a WhileStmt
/// This can have a single child: the body of the while.
class WhileStmtScope final : public ASTScope {
  WhileStmtScope(WhileStmt *stmt, ASTScope *parent)
      : ASTScope(ASTScopeKind::WhileStmt, parent), stmt(stmt) {
    assert(parent && "WhileStmtScope must have a valid parent");
    assert(stmt && "WhileStmtScope must have a non-null WhileStmt*");
  }

  WhileStmt *const stmt;

public:
  static WhileStmtScope *create(ASTContext &ctxt, WhileStmt *stmt,
                                ASTScope *parent);

  WhileStmt *getWhileStmt() const { return stmt; }

  static bool classof(ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::WhileStmt;
  }
};

} // namespace sora