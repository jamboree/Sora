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
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/PointerIntPair.h"
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
#define SCOPE(KIND) KIND,
#define LAST_SCOPE(KIND) Last_Scope = KIND
#include "Sora/AST/ASTScopeKinds.def"
};

/// Base class for AST Scopes.
///
/// ASTScopes form a tree and represent every scope in the AST.
/// The root of an ASTScope is always a SourceFileScope.
class alignas(ASTScopeAlignement) ASTScope {
  // ASTScopes shouldn't be copyable
  ASTScope(const ASTScope &) = delete;
  ASTScope &operator=(const ASTScope &) = delete;

  // Disable vanilla new/delete for ASTScopes
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;
  // Disable placement new as well
  void *operator new(size_t, void *mem) noexcept = delete;

  /// The Parent ASTScope, and the kind of ASTScope this is.
  llvm::PointerIntPair<ASTScope *, 3, ASTScopeKind> parentAndKind;
  static_assert(
      unsigned(ASTScopeKind::Last_Scope) <= (1 << 3),
      "Not enough bits in parentAndKind to represent every ASTScopeKind");
  /// The Children Scopes
  SmallVector<ASTScope *, 4> children;
  /// Whether a cleanup for this ASTScope has been registered
  bool hasCleanup = false;
  /// Whether this scope has built its children scope.
  bool expanded = true;

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

protected:
  ASTScope(ASTScopeKind kind, ASTScope *parent) : parentAndKind(parent, kind) {
    assert((parent || (kind == ASTScopeKind::SourceFile)) &&
           "Every scope except SourceFileScopes must have a parent");
  }

  // Allow derived classes to allocate ASTScopes using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(ASTScope));

public:
  /// \returns the kind of scope this is
  ASTScopeKind getKind() const { return parentAndKind.getInt(); }

  /// Adds a new child scope
  void addChild(ASTScope *scope);

  /// \returns the children scopes of this ASTScope.
  /// This will never expand this ASTScope if it's not expanded.
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

  /// Recursively expand the scope until the full ASTScope map has been
  /// generated for the subtree. This is expensive, and should be used only for
  /// testing purposes.
  void fullyExpand();

  /// \returns the SourceLoc of the first token of this scope
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of this scope
  SourceLoc getEndLoc() const;
  /// \returns the full range of this scope
  SourceRange getSourceRange() const;

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

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::SourceFile;
  }
};

/// Represents the scope of a FuncDecl
/// This should only have a single child: the BraceStmtScope of the body.
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

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::FuncDecl;
  }
};

/// Represents the scope of a local LetDecl.
/// Local LetDecls always introduce a new scope: they close the scope that
/// included everything before them and open a new scope for everything that
/// comes after them.
/// \verbatim
/// // code
///   func foo() {
///     let (a, b)
///     if false {}
///     if true [}
///     let c
///     let d
///     { a }
///     if false {}
///   }
/// /// scopes
///   FuncDeclScope
///     LocalLetDeclScope
///       IfStmtScope
///       IfStmtScope
///       LocalLetDeclScope
///         LocalLetDeclScope
///           BraceStmtScope
///           IfStmtScope
/// \endverbatim
///
/// That way, when we perform lookup for 'a', we correctly look at all of the
/// 'let's (as they are the parents of the BraceStmtScope)
///
/// Note that this also means that the end loc of the scope isn't the end loc of
/// the LetDecl.
class LocalLetDeclScope final : public ASTScope {
  LocalLetDeclScope(LetDecl *decl, ASTScope *parent, SourceLoc end)
      : ASTScope(ASTScopeKind::LocalLetDecl, parent), decl(decl), end(end) {
    assert(parent && "LocalLetDeclScope must have a valid parent");
    assert(isLocalAndNonNull() &&
           "LocalLetDeclScope must have a non-null, local LetDecl*");
  }

  bool isLocalAndNonNull() const;

  LetDecl *const decl;
  // The actual end of the implicit scope of the LetDecl.
  SourceLoc end;

public:
  static LocalLetDeclScope *create(LetDecl *decl, ASTScope *parent,
                                   SourceLoc end);

  LetDecl *getLetDecl() const { return decl; }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::LocalLetDecl;
  }
};

/// Represents the scope of a BlockStmt.
/// This can have any number of children scopes.
class BlockStmtScope final : public ASTScope {
  BlockStmtScope(BlockStmt *stmt, ASTScope *parent)
      : ASTScope(ASTScopeKind::BlockStmt, parent),
        bodyAndLookupKind(stmt, LookupKind::Unknown) {
    assert(parent && "BlockStmtScope must have a valid parent");
    assert(stmt && "BlockStmtScope must have a non-null BlockStmt*");
  }

  enum class LookupKind {
    /// This BlockStmt has interesting declarations (other than LetDecl) that we
    /// need to consider (e.g. FuncDecl, types, etc.)
    Mandatory,
    /// This BlockStmt isn't interesting and we don't need to lookup into it
    Skippable,
    /// Unknown LookupKind
    Unknown
  };

  llvm::PointerIntPair<BlockStmt *, 2, LookupKind> bodyAndLookupKind;

public:
  static BlockStmtScope *create(ASTContext &ctxt, BlockStmt *stmt,
                                ASTScope *parent);

  BlockStmt *getBlockStmt() const { return bodyAndLookupKind.getPointer(); }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::BlockStmt;
  }
};

/// Represents the scope of a IfStmt
/// This can have at most 2 children scopes: the body of the if (or the
/// condition's declaration), and the body of the else.
/// \verbatim
/// // code
/// if false {} else {}
/// if let x = x {} else {}
/// // scopes
/// IfStmtScope
///   BlockStmtScope
///   BlockStmtScope
/// IfStmtScope
///   LocalLetDeclScope
///     BlockStmtScope
///   BlockStmtScope
/// \endverbatim
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

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::BlockStmt;
  }
};

/// Represents the scope of a WhileStmt
/// This will usually have a single child, which will be the body of the while
/// or the condition's declaration.
///
/// \verbatim
/// // code
/// while false {}
/// while let x = x {}
/// // scopes
/// WhileStmtScope
///   BlockStmtScope
/// WhileStmtScope
///   LocalLetDeclScope
///     BlockStmtScope
/// \endverbatim
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

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::WhileStmt;
  }
};

} // namespace sora