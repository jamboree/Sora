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
class DeclContext;
class Expr;
class LetDecl;
class Pattern;
class SourceFile;
class Stmt;
class TypeBase;
class TypeRepr;

// Declare the FreeLowBits and Alignment variables
#define DECLARE(CLASS, FREE_BITS_DESIRED)                                      \
  constexpr std::size_t CLASS##FreeLowBits = FREE_BITS_DESIRED##u;             \
  constexpr std::size_t CLASS##Alignement = 1u << FREE_BITS_DESIRED##u

DECLARE(Decl, 3);
DECLARE(DeclContext, 3);
DECLARE(Expr, 3);
DECLARE(Pattern, 3);
DECLARE(SourceFile, 3);
DECLARE(Stmt, 3);
DECLARE(TypeBase, 1);
DECLARE(TypeRepr, 1);
#undef DECLARE
} // namespace sora

// Specialize llvm::PointerLikeTypeTraits for each class. This is important for
// multiple LLVM ADT classes, such as PointerUnion
namespace llvm {
template <class T> struct PointerLikeTypeTraits;
#define LLVM_DEFINE_PLTT(CLASS, NUMLOWBITS)                                    \
  template <> struct PointerLikeTypeTraits<::sora::CLASS *> {                  \
    enum { NumLowBitsAvailable = NUMLOWBITS };                                 \
    static inline void *getAsVoidPointer(::sora::CLASS *ptr) { return ptr; }   \
    static inline ::sora::CLASS *getFromVoidPointer(void *ptr) {               \
      return static_cast<::sora::CLASS *>(ptr);                                \
    }                                                                          \
  }

LLVM_DEFINE_PLTT(Expr, sora::ExprFreeLowBits);
LLVM_DEFINE_PLTT(Decl, sora::DeclFreeLowBits);
LLVM_DEFINE_PLTT(LetDecl, sora::DeclFreeLowBits);
LLVM_DEFINE_PLTT(Pattern, sora::PatternFreeLowBits);
LLVM_DEFINE_PLTT(SourceFile, sora::SourceFileFreeLowBits);
LLVM_DEFINE_PLTT(Stmt, sora::StmtFreeLowBits);
LLVM_DEFINE_PLTT(TypeBase, sora::TypeBaseFreeLowBits);
LLVM_DEFINE_PLTT(TypeRepr, sora::TypeReprFreeLowBits);

#undef LLVM_DEFINE_PLTT

// For PointerLikeTypeTraits
static_assert(alignof(void *) >= 2, "void* pointer alignment too small");
} // namespace llvm