//===--- NameLookup.hpp - AST Name Lookup -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTNode.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class ASTContext;

/// Represents a single scope in the Sora AST.
class ASTScope final {
  // ASTScopes shouldn't be copyable
  ASTScope(const ASTScope &) = delete;
  ASTScope &operator=(const ASTScope &) = delete;

  // Disable vanilla new/delete for ASTScopes
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;
  // Disable placement new as well
  void *operator new(size_t, void *mem) noexcept = delete;
  // Allow allocation of ASTScopes using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(ASTScope));

  // The parent ASTScope, or the ASTContext for root ASTScopes.
  const llvm::PointerUnion<ASTContext *, ASTScope *> parentOrCtxt;
  /// Children of this scope.
  /// The last element of this vector may not be a children, it could be a
  /// continuation, see \c hasContinuation.
  /// FIXME: Find a better name for this variable
  SmallVector<ASTScope *, 4> children;
  /// Whether a cleanup has already been registered for this ASTScope.
  bool cleanupRegistered = false;
  /// Whether the children of this scope have been expanded.
  bool childrenAreExpanded = false;
  /// Whether this scope's last children is a continuation.
  /// FIXME: Find a better name for this variable
  bool isContinued = false;

  /// \returns true if this ASTScope needs cleanup.
  bool needsCleanup() const {
    const char *beg = reinterpret_cast<const char *>(&children);
    const char *end = beg + sizeof(children);
    const char *data = reinterpret_cast<const char *>(children.data());
    // If the SmallVector allocated its data inside itself, it doesn't need
    // cleanups.
    // FIXME: Using .isSmall() would be so much cleaner, but we don't have
    // access to it.
    if ((beg <= data) && (data <= end)) {
      assert(children.size() <= 4);
      return true;
    }
    return false;
  }

  /// The ASTScope's constructors are private because the class uses ASTContext
  /// cleanups, so it's dangerous to allocate an ASTScope on the stack.
  ASTScope(ASTScope *scope, ASTNode node) : parentOrCtxt(scope), node(node) {
    assert(scope && "scope can't be null");
  }

  ASTScope(ASTContext &ctxt, ASTNode node) : parentOrCtxt(&ctxt), node(node) {}

  class ChildrenBuilder;

public:
  /// Creates an ASTScope with \p scope as parent (& register it as a child of
  /// \p scope)
  static ASTScope *createChild(ASTScope *scope, ASTNode node);
  /// Creates an ASTScope which is a continuation of \p scope.
  /// Once this scope has been created, no extra children may be added to \p
  /// scope.
  static ASTScope *createContinuation(ASTScope *scope, ASTNode node);
  /// Creates a root ASTScope
  static ASTScope *createRoot(ASTContext &ctxt, ASTNode node);

  ASTContext &getASTContext() const {
    if (ASTScope *parent = getParent())
      return parent->getASTContext();
    ASTContext *ctxt = parentOrCtxt.get<ASTContext *>();
    assert(ctxt && "ctxt ptr is null");
    return *ctxt;
  }

  /// \returns the parent of this ASTScope, or null in the case of a root scope.
  ASTScope *getParent() const { return parentOrCtxt.dyn_cast<ASTScope *>(); }
  /// \returns true if this is a root ASTScope (with no parent).
  bool isRoot() const { return getParent(); }

  /// Adds a child to this ASTSCope.
  void addChild(ASTScope *scope);

  /// Adds a continuation to this ASTScope.
  /// Once a continuation has been added, no more children can be added.
  void addContinuation(ASTScope *scope) {
    assert(!hasContinuation && "Already has a continuation");
    addChild(scope);
    isContinued = true;
  }

  /// \returns the children of this ASTScope. This will never expand the
  /// children if they're not expanded yet.
  ArrayRef<ASTScope *> getChildren() const {
    return const_cast<ASTScope *>(this)->getChildren(false);
  }

  /// \param allowExpansion is true, the children of this ASTScope will be
  /// expanded if it hasn't been done yet.
  /// \returns the children of this AST Node.
  MutableArrayRef<ASTScope *> getChildren(bool allowExpansion = false) {
    if (!childrenAreExpanded && allowExpansion)
      expandChildren();
    MutableArrayRef<ASTScope *> result = children;
    // If we got a continuation, the last element of the "children" vector is
    // actually a continuation, so ignore it.
    return hasContinuation() ? result.drop_back() : result;
  }

  /// Whether this ASTScope has a continuation
  bool hasContinuation() const { return isContinued; }

  /// Returns the continuation of this ASTScope.
  /// Can only be used if hasContinuation() returns true.
  ASTScope *getContinuation() const {
    assert(hasContinuation() && "Doesn't have a continuation");
    assert(children.size() && "Continuation without children?");
    return children.back();
  }

  /// Expands the Children of this ASTScope if it hasn't been done yet.
  void expandChildren();

  /// Recursively expand the children of this ASTScope until the full ASTScope
  /// tree is generated.
  /// This is mostly used for testing purposes.
  void fullyExpand();

  /// \returns true if this ASTScope represents a FuncDecl's scope.
  bool isFuncDecl() const;

  /// The AST node behind this scope
  const ASTNode node;

  /// Dumps this ASTScope to \p out
  void dump(raw_ostream &out, unsigned indent = 2) const {
    dumpImpl(out, indent, 0, /*isContinuation*/ false);
  }

private:
  void dumpImpl(raw_ostream &out, unsigned indent, unsigned curIndent,
                bool isContinuation) const;
};
} // namespace sora