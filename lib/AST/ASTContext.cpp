//===--- ASTContext.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include <algorithm>

using namespace sora;

/// the ASTContext's implementation
struct ASTContext::Impl {
  /// for ASTArenaKind::Permanent
  llvm::BumpPtrAllocator permanentAllocator;
  /// for ASTArenaKind::Unresolved
  llvm::BumpPtrAllocator unresolvedAllocator;
};

ASTContext::ASTContext(SourceManager &srcMgr, DiagnosticEngine &diagEngine)
    : srcMgr(srcMgr), diagEngine(diagEngine) {}

ASTContext::Impl &ASTContext::getImpl() {
  return *reinterpret_cast<Impl *>(llvm::alignAddr(this + 1, alignof(Impl)));
}

std::unique_ptr<ASTContext> ASTContext::create(SourceManager &srcMgr,
                                               DiagnosticEngine &diagEngine) {
  // FIXME: This could be simplified with a aligned_alloc if we had access to
  // it.

  // We need to allocate enough memory to support both the ASTContext and its
  // implementation *plus* some padding to align the addresses correctly.
  size_t sizeToAlloc = sizeof(ASTContext) + (alignof(ASTContext) - 1);
  sizeToAlloc += sizeof(Impl) + (alignof(Impl) - 1);

  void *memory = llvm::safe_malloc(sizeToAlloc);
  // The ASTContext's memory begins at the first correctly aligned address
  // of the memory
  void *astContextMemory =
      reinterpret_cast<void *>(llvm::alignAddr(memory, alignof(ASTContext)));
  // The Impl's memory begins at the first correctly aligned addres after the
  // ASTContext's memory.
  void *implMemory = (char *)astContextMemory + sizeof(ASTContext);
  implMemory =
      reinterpret_cast<void *>(llvm::alignAddr(implMemory, alignof(Impl)));

  // Do some checking because I'm kinda paranoïd.
  //  Check that we aren't going out of bounds and going to segfault later.
  assert(((char *)implMemory + sizeof(Impl)) < ((char *)memory + sizeToAlloc) &&
         "Going out-of-bounds of the allocated memory");
  //  Check that the ASTContext's memory doesn't overlap the Implementation's.
  assert((((char *)astContextMemory + sizeof(ASTContext)) <= implMemory) &&
         "ASTContext's memory overlaps the Impl's memory");

  // Call the placement news
  ASTContext *astContext =
      new (astContextMemory) ASTContext(srcMgr, diagEngine);
  new (implMemory) Impl();

  // And return a managed pointer.
  return std::unique_ptr<ASTContext>(astContext);
}

ASTContext::~ASTContext() { getImpl().~Impl(); }

llvm::BumpPtrAllocator &ASTContext::getAllocator(ASTAllocatorKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown allocator kind");
  case ASTAllocatorKind::Permanent:
    return getImpl().permanentAllocator;
  case ASTAllocatorKind::Unresolved:
    return getImpl().unresolvedAllocator;
  }
}
