//===--- Type.hpp - Type Object ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the "Type" object which is a wrapper around TypeBase
// pointers. This is needed to disable direct type pointer comparison, which
// can be deceiving due to the canonical/non-canonical type distinction.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include <cassert>

namespace sora {
class TypeBase;

class Type {
  TypeBase *ptr = nullptr;

public:
  /// Creates an invalid Type
  Type() = default;

  /// Creates a Type from a TypeBase pointer
  /*implicit*/ Type(TypeBase *ptr) : ptr(ptr) {}

  TypeBase *getPtr() { return ptr; }
  const TypeBase *getPtr() const { return ptr; }

  TypeBase *operator->() {
    assert(ptr && "cannot use this on a null pointer");
    return ptr;
  }

  const TypeBase *operator->() const {
    assert(ptr && "cannot use this on a null pointer");
    return ptr;
  }

  TypeBase &operator*() {
    assert(ptr && "cannot use this on a null pointer");
    return *ptr;
  }

  const TypeBase &operator*() const {
    assert(ptr && "cannot use this on a null pointer");
    return *ptr;
  }

  explicit operator bool() const { return ptr != nullptr; }

  // for STL containers
  bool operator<(const Type other) const { return ptr < other.ptr; }
};
} // namespace sora

namespace llvm {
// A Type is just a wrapper around a TypeBase*, so it can safely be considered
// as pointer-like.
template <> struct PointerLikeTypeTraits<::sora::Type> {
public:
  enum { NumLowBitsAvailable = ::sora::TypeBaseFreeLowBits };

  static inline void *getAsVoidPointer(::sora::Type type) {
    return type.getPtr();
  }

  static inline ::sora::Type getFromVoidPointer(void *ptr) {
    return (::sora::TypeBase *)ptr;
  }
};
} // namespace llvm