//===--- DeclContext.hpp - Declaration Contexts -----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/ErrorHandling.h"
#include <stdint.h>

namespace sora {
class FuncDecl;

/// Kinds of DeclContext
enum class DeclContextKind : uint8_t {
  FuncDecl,
  Last_LocalDeclContext = FuncDecl,

  SourceFile,
  Last_DeclContext = SourceFile
};

/// DeclContexts are semantic containers for declarations, acting like
/// "anchors" in the AST for declarations.
/// Declarations keep track of their parent DeclContext, and they can
/// use them to gather information and access the root SourceFile.
class alignas(DeclContextAlignement) DeclContext {
  llvm::PointerIntPair<DeclContext *, DeclContextFreeLowBits, DeclContextKind>
      parentAndKind;

  static_assert(
      unsigned(DeclContextKind::Last_DeclContext) <=
          (1 << DeclContextFreeLowBits),
      "Too many DeclContextKind exceeds bits available in DeclContext*");

protected:
  DeclContext(DeclContextKind kind, DeclContext *parent)
      : parentAndKind(parent, kind) {}

public:
  /// \returns true if this is a local DeclContext
  bool isLocalContext() const {
    return getDeclContextKind() <= DeclContextKind::Last_LocalDeclContext;
  }

  /// \returns the parent of this DeclContext
  DeclContext *getParent() const { return parentAndKind.getPointer(); }

  /// \returns the parent SourceFile of this DeclContext, or nullptr if not
  /// found.
  SourceFile *getParentSourceFile() const;

  /// \returns true if this DeclContext has a parent
  bool hasParent() const { return getParent(); }

  /// \returns the kind of this DeclContext
  DeclContextKind getDeclContextKind() const { return parentAndKind.getInt(); }

  // See Decl.hpp
  static bool classof(const Decl *decl);
};
} // namespace sora