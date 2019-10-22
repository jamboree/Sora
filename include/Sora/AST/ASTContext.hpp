//===--- ASTContext.hpp - AST Root and Memory Management --------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "llvm/Support/Allocator.h"
#include <functional>
#include <memory>
#include <stdint.h>

namespace llvm {
class Triple;
}

namespace sora {
class SourceManager;
class DiagnosticEngine;

enum class AllocatorKind : uint8_t {
  /// The "permanent" AST allocator that holds long-lived objects such as types,
  /// resolved AST nodes, etc.
  /// This is freed when the ASTContext is deallocated.
  Permanent,
  /// The AST allocator for expressions that inherit from UnresolvedExpr.
  /// This can be freed using "freeUnresolvedExprs()" once the AST has been
  /// fully typechecked.
  UnresolvedExpr,
  /// The Allocator used by the Typechecker. This is where TypeVariables &
  /// constraints are allocated.
  /// This allocator is not active by default, but can be used when a
  /// TypeCheckerAllocatorRAII is active.
  TypeChecker
};

/// An RAII object that enables usage of the TypeChecker's allocator.
/// Once this object is destroyed, everything within the TypeChecker allocator
/// is freed.
struct TypeCheckerAllocatorRAII {
  ASTContext &ctxt;

  /// This object shouldn't be copyable.
  TypeCheckerAllocatorRAII(const TypeCheckerAllocatorRAII &) = delete;
  TypeCheckerAllocatorRAII &
  operator=(const TypeCheckerAllocatorRAII &) = delete;

  TypeCheckerAllocatorRAII(ASTContext &ctxt);
  ~TypeCheckerAllocatorRAII();
};

/// The ASTContext is a large object designed as the core of the AST.
/// It contains the allocator for the AST, many type singletons and references
/// to other important structures like the DiagnosticEngine & SourceManager.
class ASTContext final {
  /// The ASTContext's implementation
  struct Impl;

  ASTContext(const ASTContext &) = delete;
  ASTContext &operator=(const ASTContext &) = delete;

  ASTContext(const SourceManager &srcMgr, DiagnosticEngine &diagEngine);

  llvm::BumpPtrAllocator &
  getAllocator(AllocatorKind kind = AllocatorKind::Permanent);

public:
  /// Members for ASTContext.cpp
  Impl &getImpl();
  const Impl &getImpl() const {
    return const_cast<ASTContext *>(this)->getImpl();
  }

  /// Creates a new ASTContext.
  /// This is a separate method because the ASTContext needs to trail-allocate
  /// its implementation object.
  static std::unique_ptr<ASTContext> create(const SourceManager &srcMgr,
                                            DiagnosticEngine &diagEngine);
  ~ASTContext();

  /// Allocates memory using \p allocator.
  /// \returns a pointer to the allocated memory (aligned to \p align) or
  /// nullptr if \p size == 0
  void *allocate(size_t size, size_t align,
                 AllocatorKind allocator = AllocatorKind::Permanent) {
    return size ? getAllocator(allocator).Allocate(size, align) : nullptr;
  }

  /// Allocates enough memory for an object \p Ty.
  /// This simply calls allocate using sizeof/alignof Ty.
  /// This does not construct the object. You'll need to use placement
  /// new for that.
  template <typename Ty>
  void *allocate(AllocatorKind allocator = AllocatorKind::Permanent) {
    return allocate(sizeof(Ty), alignof(Ty), allocator);
  }

  /// \returns true if the AllocatorKind::TypeChecker allocator is active.
  /// It is only active if there's one active TypeCheckerAllocatorRAII.
  bool isTypeCheckerAllocatorActive() const;

  /// Frees (deallocates) all UnresolvedExprs allocated within this ASTContext.
  void freeUnresolvedExprs();

  /// Adds a cleanup function that'll be run when the ASTContext's memory (&
  /// Permanent Allocator) is freed.
  void addCleanup(std::function<void()> cleanup);

  /// Adds a destructor cleanup function that'll be run when the ASTContext's
  /// memory is freed. This can be used to destroy non trivially-destructible
  /// AST Nodes that are allocated in the ASTContext's Permanent allocator.
  template <typename Ty> void addDestructorCleanup(Ty &obj) {
    addCleanup([&obj]() { obj.~Ty(); });
  }

  /// Interns an identifier string
  /// \returns an identifier object for \p str
  Identifier getIdentifier(StringRef str);

  /// Overrides the default target triple
  void overrideTargetTriple(const llvm::Triple &triple);

  /// \returns the target triple.
  llvm::Triple getTargetTriple() const;

  /// \returns the builtin type with name \p str, or null if no builtin type
  /// with that name was found.
  Type getBuiltinType(StringRef str);

  //===- Common Singletons ------------------------------------------------===//

  /// The SourceManager that owns the source buffers that created this AST.
  const SourceManager &srcMgr;

  /// The DiagnosticEngine used to diagnose errors related to this AST.
  DiagnosticEngine &diagEngine;

  //===- Frequently Used Types --------------------------------------------===//

  const Type i8Type;    /// "i8"
  const Type i16Type;   /// "i16"
  const Type i32Type;   /// "i32"
  const Type i64Type;   /// "i64"
  const Type isizeType; /// "isize"

  const Type u8Type;    /// "u8"
  const Type u16Type;   /// "u16"
  const Type u32Type;   /// "u32"
  const Type u64Type;   /// "u64"
  const Type usizeType; /// "usize"

  const Type f32Type; /// "f32"
  const Type f64Type; /// "f64"

  const Type voidType; /// "void"

  const Type errorType; // The "error" type.

  //===--------------------------------------------------------------------===//
};
} // namespace sora