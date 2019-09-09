//===--- Decl.hpp - Declarations ASTs ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;

/// Kinds of Declarations
enum class DeclKind : uint8_t {
#define DECL(KIND, PARENT) KIND,
#define DECL_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/DeclNodes.def"
};

/// Base class for every Declaration node.
class alignas(DeclAlignement) Decl {
  // Disable vanilla new/delete for declarations
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  DeclKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Decl(DeclKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of declaration using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Decl));

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const;
  /// \returns the full range of this declaration
  SourceRange getSourceRange() const;

  /// \return the kind of declaration this is
  DeclKind getKind() const { return kind; }
};
} // namespace sora