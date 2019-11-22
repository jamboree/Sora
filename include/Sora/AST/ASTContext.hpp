//===--- ASTContext.hpp - AST Root and Memory Management --------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
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

enum class ArenaKind : uint8_t {
  /// The "permanent" arena that holds long-lived objects such as most
  /// types and AST nodes.
  /// This is freed when the ASTContext is deallocated.
  /// This is the default allocator, and can be used to allocate types.
  Permanent,
  /// The arena for expressions that inherit from UnresolvedExpr.
  /// This can be freed using "freeUnresolvedExprs()" once the AST has been
  /// fully typechecked.
  /// This arena is never used to allocate types;
  UnresolvedExpr,
  /// The arena used by the Typechecker's ConstraintSystem. This is where
  /// TypeVariables, types containing TypeVariables & constraints themselves are
  /// allocated. This allocator is not active by default. It can ONLY be used
  /// when a RAIIConstraintSystemArena object is alive.
  ConstraintSystem
};

/// An RAII object that enables usage of the ConstraintSystem's arena.
/// Once this object is destroyed, everything within the ConstraintSystem arena
/// is freed.
class RAIIConstraintSystemArena final {
private:
  /// This object shouldn't be copyable.
  RAIIConstraintSystemArena(const RAIIConstraintSystemArena &) = delete;
  RAIIConstraintSystemArena &
  operator=(const RAIIConstraintSystemArena &) = delete;

  /// Only the ASTContext can create those
  friend class ASTContext;
  RAIIConstraintSystemArena(ASTContext &ctxt);

public:
  RAIIConstraintSystemArena(RAIIConstraintSystemArena &&) = default;

  ASTContext &ctxt;

  ~RAIIConstraintSystemArena();
};

/// The ASTContext is a large object designed as the core of the AST.
/// It contains the allocator for the AST, many type singletons and references
/// to other important structures like the DiagnosticEngine & SourceManager.
class alignas(ASTContextAlignement) ASTContext final {
  /// The ASTContext's implementation
  struct Impl;

  ASTContext(const ASTContext &) = delete;
  ASTContext &operator=(const ASTContext &) = delete;

  ASTContext(const SourceManager &srcMgr, DiagnosticEngine &diagEngine);

  llvm::BumpPtrAllocator &getArena(ArenaKind kind = ArenaKind::Permanent);

  void buildBuiltinTypesLookupMap();

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
                 ArenaKind allocator = ArenaKind::Permanent) {
    return size ? getArena(allocator).Allocate(size, align) : nullptr;
  }

  /// Allocates enough memory for an object \p Ty.
  /// This simply calls allocate using sizeof/alignof Ty.
  /// This does not construct the object. You'll need to use placement
  /// new for that.
  template <typename Ty>
  void *allocate(ArenaKind allocator = ArenaKind::Permanent) {
    return allocate(sizeof(Ty), alignof(Ty), allocator);
  }

  /// \returns true if the ArenaKind::ConstraintSystem allocator is active.
  /// It is only active if there's one active RAIIConstraintSystemArena.
  bool hasConstraintSystemArena() const;

  /// Creates a new ConstraintSystem arena with a lifetime tied to the returned
  /// object's.
  /// \c hasConstraintSystemArena() must return false for this to be used!
  RAIIConstraintSystemArena createConstraintSystemArena();

  /// Frees (deallocates) all UnresolvedExprs allocated within this ASTContext.
  void freeUnresolvedExprs();

  /// \returns the total memory used (in bytes) by this ASTContext and all of
  /// its arenas.
  size_t getTotalMemoryUsed() const;
  /// \returns the memory used (in bytes) by \p arena.
  size_t getMemoryUsed(ArenaKind arena) const;

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

  /// \returns the builtin type with name \p ident, or nullptr if nothing was
  /// found.
  CanType lookupBuiltinType(Identifier ident) const;

  /// Puts a list of every available builtin type in \p results.
  void getAllBuiltinTypes(SmallVectorImpl<Type> &results) const;

  /// Puts a list of every available builtin type in \p results.
  void getAllBuiltinTypes(SmallVectorImpl<CanType> &results) const;

  //===- Common Singletons ------------------------------------------------===//

  /// The SourceManager that owns the source buffers that created this AST.
  const SourceManager &srcMgr;

  /// The DiagnosticEngine used to diagnose errors related to this AST.
  DiagnosticEngine &diagEngine;

  //===- Frequently Used Types --------------------------------------------===//

  const CanType i8Type;    /// "i8"
  const CanType i16Type;   /// "i16"
  const CanType i32Type;   /// "i32"
  const CanType i64Type;   /// "i64"
  const CanType isizeType; /// "isize"

  const CanType u8Type;    /// "u8"
  const CanType u16Type;   /// "u16"
  const CanType u32Type;   /// "u32"
  const CanType u64Type;   /// "u64"
  const CanType usizeType; /// "usize"

  const CanType f32Type; /// "f32"
  const CanType f64Type; /// "f64"

  const CanType voidType; /// "void"
  const CanType boolType; /// "bool"

  const CanType errorType; // The "error" type.

  //===--------------------------------------------------------------------===//
};
} // namespace sora