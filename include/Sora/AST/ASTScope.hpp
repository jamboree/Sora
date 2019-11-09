//===--- ASTScope.hpp - AST Scope -------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/ASTNode.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
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
struct UnqualifiedLookupOptions;
class ValueDecl;

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
  /// The sorted vector of children scopes
  SmallVector<ASTScope *, 4> children;
  /// Whether a cleanup for this ASTScope has been registered
  bool hasCleanup = false;
  /// Whether this scope has built its children scope.
  bool expanded = false;

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

  /// A Lookup Result Consider, which is called once per AST Scope with the list
  /// of candidates found in the scope and the scope in which the results were
  /// found. This isn't called when there are no results.
  /// If the consumer returns true, lookup stops.
  ///
  /// As this is a function_ref, it is generally not safe to store.
  using LookupResultConsumer =
      llvm::function_ref<bool(ArrayRef<ValueDecl *>, const ASTScope *)>;

  /// Performs lookup in this scope and parent scopes.
  /// \param consumer the consumer function, which is called once per scope.
  /// When it returns true, lookup stops. Note that the consumer is never called
  /// with an empty array.
  /// \param options the lookup options to use
  /// \param ident the identifier to look for. If it's null, simply returns a
  /// list of every decl visible in the scope. When an identifier is provided,
  /// this will try to use cached lookup maps when possible, making the search
  /// more efficient.
  void lookup(LookupResultConsumer consumer,
              const UnqualifiedLookupOptions &options,
              Identifier ident = Identifier()) const;

  /// \returns the innermost scope around \p loc, or this if no this is the
  /// innermost scope.
  /// \p loc must be within this scope's SourceRange.
  ASTScope *findInnermostScope(SourceLoc loc);

  /// \returns true if this scope overlaps \p other
  bool overlaps(const ASTScope *other) const;

  /// Adds a new child scope.
  /// \p scope can't be null and its range must not overlap with other existing
  /// scopes' range.
  void addChild(ASTScope *scope);

  /// \returns the sorted array of children scopes
  /// This won't expand this ASTScope if it's not expanded.
  ArrayRef<ASTScope *> getChildren() const { return children; }

  /// \param canExpand if true, the ASTScope will expand if needed.
  /// \returns the sorted array of children scopes
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
///
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
/// This is needed so order is respected when performing local lookup (can't use
/// the var before it has been declared).
///
/// Note that the range of a LocalLetDeclScope doesn't depend in any way on the
/// range of the LetDecl. The range of the scope is given by the parent.
///
/// For LetDecls inside BlockStmts, the range of the scope is a range that
/// begins past-the-end of the LetDecl and ends at the '}'.
///
/// \verbatim
///   {
///     let x: i32 // scope begins on the whitespace after the 'i32' token
///     x = 0
///   }           // scope ends at the '}'
/// \endverbatim
///
/// For LetDecls inside conditions, the range of the scope is the range of the
/// body (the "then" block for IfStmts and the loop's body for WhileStmts)
class LocalLetDeclScope final : public ASTScope {
  LocalLetDeclScope(LetDecl *decl, ASTScope *parent, SourceRange range)
      : ASTScope(ASTScopeKind::LocalLetDecl, parent), decl(decl), range(range) {
    assert(parent && "LocalLetDeclScope must have a valid parent");
    assert(range && "range isn't valid");
    assert(isLocalAndNonNull() &&
           "LocalLetDeclScope must have a non-null, local LetDecl*");
  }

  bool isLocalAndNonNull() const;

  LetDecl *const decl;
  SourceRange range;

public:
  static LocalLetDeclScope *create(LetDecl *decl, ASTScope *parent,
                                   SourceRange range);

  LetDecl *getLetDecl() const { return decl; }

  SourceRange getSourceRange() const { return range; }

  static bool classof(const ASTScope *scope) {
    return scope->getKind() == ASTScopeKind::LocalLetDecl;
  }
};

/// Represents the scope of a BlockStmt.
/// This can have any number of children scopes.
class BlockStmtScope final : public ASTScope {
  BlockStmtScope(BlockStmt *stmt, ASTScope *parent)
      : ASTScope(ASTScopeKind::BlockStmt, parent), block(stmt) {
    assert(parent && "BlockStmtScope must have a valid parent");
    assert(stmt && "BlockStmtScope must have a non-null BlockStmt*");
  }

  BlockStmt *const block;

public:
  static BlockStmtScope *create(ASTContext &ctxt, BlockStmt *stmt,
                                ASTScope *parent);

  BlockStmt *getBlockStmt() const { return block; }

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
      : ASTScope(ASTScopeKind::IfStmt, parent), stmt(stmt) {
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
    return scope->getKind() == ASTScopeKind::IfStmt;
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