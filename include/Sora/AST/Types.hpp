//===--- Types.hpp - Types ASTs ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the whole "type" hierarchy
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/IntegerWidth.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/Support/Error.h"

namespace sora {
class ASTContext;

/// Kinds of Types
enum class TypeKind : uint8_t {
#define TYPE(KIND, PARENT) KIND,
#define TYPE_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/TypeNodes.def"
};

/// Common base class for Types.
class alignas(TypeBaseAlignement) TypeBase {
  // Disable vanilla new/delete for types
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  TypeKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  // Also allow allocation of Types using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(TypeBase));

  TypeBase(TypeKind kind) : kind(kind) {}

public:
  /// \returns the kind of type this is
  TypeKind getKind() const { return kind; }
};
} // namespace sora