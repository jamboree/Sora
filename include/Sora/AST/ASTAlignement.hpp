//===--- ASTAlignement.hpp - AST Nodes Alignement Values --------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>

namespace sora {
class Decl;
class Expr;
class Stmt;
class TypeBase;

// Declare the FreeLowBits and Alignment variables
#define DECLARE(CLASS, FREE_BITS_DESIRED)                                      \
  constexpr std::size_t CLASS##FreeLowBits = FREE_BITS_DESIRED##u;             \
  constexpr std::size_t CLASS##Alignement = 1u << FREE_BITS_DESIRED##u

DECLARE(TypeBase, 1);
DECLARE(Expr, 3);
DECLARE(Decl, 3);
DECLARE(Stmt, 3);
#undef DECLARE
} // namespace sora

// Specialize llvm::PointerLikeTypeTraits for each class. This is important for
// multiple LLVM ADT classes, such as PointerUnion
namespace llvm {
template <class T> struct PointerLikeTypeTraits;
#define LLVM_DEFINE_PLTT(CLASS)                                                \
  template <> struct PointerLikeTypeTraits<::sora::CLASS *> {                  \
    enum { NumLowBitsAvailable = ::sora::CLASS##FreeLowBits };                 \
    static inline void *getAsVoidPointer(::sora::CLASS *ptr) { return ptr; }   \
    static inline ::sora::CLASS *getFromVoidPointer(void *ptr) {               \
      return static_cast<::sora::CLASS *>(ptr);                                \
    }                                                                          \
  }

LLVM_DEFINE_PLTT(TypeBase);
LLVM_DEFINE_PLTT(Expr);
LLVM_DEFINE_PLTT(Decl);
LLVM_DEFINE_PLTT(Stmt);

#undef LLVM_DEFINE_PLTT

// For PointerLikeTypeTraits
static_assert(alignof(void *) >= 2, "void* pointer alignment too small");
} // namespace llvm