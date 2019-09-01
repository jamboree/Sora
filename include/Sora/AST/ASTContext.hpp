//===--- ASTContext.hpp - AST Root and Memory Management --------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Identifier.hpp"
#include "llvm/Support/Allocator.h"
#include <memory>
#include <stdint.h>

namespace sora {
class SourceManager;
class DiagnosticEngine;

enum class ASTAllocatorKind : uint8_t {
  /// The "permanent" AST allocator that holds long-lived objects such as types,
  /// resolved AST nodes, etc.
  Permanent,
  /// The AST allocator for unresolved AST nodes.
  /// This is usually freed right after semantic analysis.
  UnresolvedNodes
};

/// The ASTContext is a large object designed as the core of the AST.
/// It contains the allocator for the AST, many type singletons and references
/// to other important structures like the DiagnosticEngine & SourceManager.
class ASTContext final {
  /// The ASTContext's implementation
  struct Impl;

  ASTContext(const ASTContext &) = delete;
  ASTContext &operator=(const ASTContext &) = delete;

  ASTContext(SourceManager &srcMgr, DiagnosticEngine &diagEngine);

public:
  /// Members for ASTContext.cpp
  Impl &getImpl();
  const Impl &getImpl() const {
    return const_cast<ASTContext *>(this)->getImpl();
  }

  /// Creates a new ASTContext.
  /// This is a separate method because the ASTContext needs to trail-allocate
  /// its implementation object.
  static std::unique_ptr<ASTContext> create(SourceManager &srcMgr,
                                            DiagnosticEngine &diagEngine);
  ~ASTContext();

  /// Allocates memory using \p allocator.
  /// \returns a pointer to the allocated memory (aligned to \p align) or
  /// nullptr if \p size == 0
  void *allocate(size_t size, size_t align,
                 ASTAllocatorKind allocator = ASTAllocatorKind::Permanent) {
    return size ? getAllocator(allocator).Allocate(size, align) : nullptr;
  }

  /// Allocates enough memory for an object \p Ty.
  /// This simply calls allocate using sizeof/alignof Ty.
  /// This does not construct the object. You'll need to use placement
  /// new for that.
  template <typename Ty>
  void *allocate(ASTAllocatorKind allocator = ASTAllocatorKind::Permanent) {
    return allocate(sizeof(Ty), alignof(Ty), allocator);
  }

  /// Fetch an allocator. This can be used to allocate or free memory.
  /// ASTContext allocators are "bump pointer allocators", or "arena"
  /// allocators if you will. You can only free *everything* at once
  /// (deallocate is useless).
  ///
  /// \param kind the kind of allocator desired (default = Permanent)
  /// \returns a reference the allocator corresponding to \p kind
  llvm::BumpPtrAllocator &
  getAllocator(ASTAllocatorKind kind = ASTAllocatorKind::Permanent);

  /// Interns an identifier string
  /// \returns an identifier object for \p str
  Identifier getIdentifier(StringRef str);

  /// The SourceManager that owns the source buffers that created this AST.
  SourceManager &srcMgr;

  /// The DiagnosticEngine used to diagnose errors related to the AST.
  DiagnosticEngine &diagEngine;
};
} // namespace sora