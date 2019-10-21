﻿//===--- ASTContext.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"
#include <algorithm>
#include <tuple>

using namespace sora;

/// the ASTContext's implementation
struct ASTContext::Impl {
  /// for AllocatorKind::Permanent
  llvm::BumpPtrAllocator permanentAllocator;
  /// for AllocatorKind::UnresolvedExpr
  llvm::BumpPtrAllocator unresolvedExprAllocator;

  /// The Identifier Table
  /// FIXME: Is using MallocAllocator the right thing to do here?
  llvm::StringSet<> identifierTable;

  /// The set of cleanups that must be ran when the ASTContext is destroyed.
  SmallVector<std::function<void()>, 4> cleanups;

  /// Signed Integer Types
  llvm::DenseMap<IntegerWidth, IntegerType *> signedIntegerTypes;
  /// Unsigned Integer Types
  llvm::DenseMap<IntegerWidth, IntegerType *> unsignedIntegerTypes;
  /// Floating-point types

  /// The target triple
  llvm::Triple targetTriple;

  Impl() {
    // Init with a default target triple
    targetTriple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
  }

  /// Custom destructor that runs the cleanups.
  ~Impl() {
    for (auto &cleanup : cleanups)
      cleanup();
  }
};

static IntegerWidth getPointerWidth(ASTContext &ctxt) {
  return IntegerWidth::pointer(ctxt.getTargetTriple());
}

ASTContext::ASTContext(const SourceManager &srcMgr,
                       DiagnosticEngine &diagEngine)
    : srcMgr(srcMgr), diagEngine(diagEngine),
      i8Type(IntegerType::getSigned(*this, IntegerWidth::fixed(8))),
      i16Type(IntegerType::getSigned(*this, IntegerWidth::fixed(16))),
      i32Type(IntegerType::getSigned(*this, IntegerWidth::fixed(32))),
      i64Type(IntegerType::getSigned(*this, IntegerWidth::fixed(64))),
      isizeType(IntegerType::getSigned(*this, getPointerWidth(*this))),
      u8Type(IntegerType::getUnsigned(*this, IntegerWidth::fixed(8))),
      u16Type(IntegerType::getUnsigned(*this, IntegerWidth::fixed(16))),
      u32Type(IntegerType::getUnsigned(*this, IntegerWidth::fixed(32))),
      u64Type(IntegerType::getUnsigned(*this, IntegerWidth::fixed(64))),
      usizeType(IntegerType::getUnsigned(*this, getPointerWidth(*this))),
      f32Type(new (*this) FloatType(FloatKind::IEEE32)),
      f64Type(new (*this) FloatType(FloatKind::IEEE64)),
      voidType(new (*this) VoidType()) {}

ASTContext::Impl &ASTContext::getImpl() {
  return *reinterpret_cast<Impl *>(llvm::alignAddr(this + 1, alignof(Impl)));
}

std::unique_ptr<ASTContext> ASTContext::create(const SourceManager &srcMgr,
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

  // Use placement new to call the constructors.
  // Note: it is very important that the implementation is initialized first.
  new (implMemory) Impl();
  ASTContext *astContext =
      new (astContextMemory) ASTContext(srcMgr, diagEngine);

  // And return a managed pointer.
  return std::unique_ptr<ASTContext>(astContext);
}

ASTContext::~ASTContext() { getImpl().~Impl(); }

llvm::BumpPtrAllocator &ASTContext::getAllocator(AllocatorKind kind) {
  switch (kind) {
  case AllocatorKind::Permanent:
    return getImpl().permanentAllocator;
  case AllocatorKind::UnresolvedExpr:
    return getImpl().unresolvedExprAllocator;
  }
  llvm_unreachable("unknown allocator kind");
}

void ASTContext::freeUnresolvedExprs() {
  getImpl().unresolvedExprAllocator.Reset();
}

void ASTContext::addCleanup(std::function<void()> cleanup) {
  getImpl().cleanups.push_back(cleanup);
}

Identifier ASTContext::getIdentifier(StringRef str) {
  // Don't intern null & empty strings (StringRef::size() returns 0 for null
  // strings)
  return str.size() ? getImpl().identifierTable.insert(str).first->getKeyData()
                    : Identifier();
}

void ASTContext::overrideTargetTriple(const llvm::Triple &triple) {
  getImpl().targetTriple = triple;
}

llvm::Triple ASTContext::getTargetTriple() const {
  return getImpl().targetTriple;
}

//===- Types --------------------------------------------------------------===//

IntegerType *IntegerType::getSigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl().signedIntegerTypes[width];
  if (ty)
    return ty;
  return ty = (new (ctxt) IntegerType(width, /*isSigned*/ true));
}

IntegerType *IntegerType::getUnsigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl().unsignedIntegerTypes[width];
  if (ty)
    return ty;
  return ty = (new (ctxt) IntegerType(width, /*isSigned*/ false));
}