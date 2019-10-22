//===--- ASTContext.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
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
  /// Reference types
  llvm::DenseMap<size_t, ReferenceType *> referenceTypes;
  /// Maybe types
  llvm::DenseMap<TypeBase *, MaybeType *> maybeTypes;
  /// Tuple types
  llvm::FoldingSet<TupleType> tupleTypes;
  /// LValue types
  llvm::DenseMap<TypeBase *, LValueType *> lvalueTypes;

  TupleType *emptyTupleType = nullptr;

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
      f32Type(new (*this) FloatType(*this, FloatKind::IEEE32)),
      f64Type(new (*this) FloatType(*this, FloatKind::IEEE64)),
      voidType(new (*this) VoidType(*this)),
      errorType(new (*this) ErrorType(*this)) {}

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

Type ASTContext::getBuiltinType(StringRef str) {
  // All builtin types currently begin with 'i', 'u'  or 'f'.
  char first = str[0];
  if (first != 'i' && first != 'u' && first != 'f')
    return nullptr;

  // They also all have a length of 2 to 3 characters.
  if (str.size() < 2 || str.size() > 3)
    return nullptr;

  // Signed integers begin with 'i'
  if (first == 'i') {
    return llvm::StringSwitch<Type>(str)
        .Case("i8", i8Type)
        .Case("i16", i16Type)
        .Case("i32", i32Type)
        .Case("i64", i64Type)
        .Default(nullptr);
  }
  // Unsigned integers begin with 'u'
  if (first == 'u') {
    return llvm::StringSwitch<Type>(str)
        .Case("u8", u8Type)
        .Case("u16", u16Type)
        .Case("u32", u32Type)
        .Case("u64", u64Type)
        .Default(nullptr);
  }
  // Floats begin with 'f'
  if (first == 'f') {
    return llvm::StringSwitch<Type>(str)
        .Case("f32", f32Type)
        .Case("f64", f64Type)
        .Default(nullptr);
  }
  return nullptr;
}

//===- Types --------------------------------------------------------------===//

IntegerType *IntegerType::getSigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl().signedIntegerTypes[width];
  if (ty)
    return ty;
  return ty = (new (ctxt) IntegerType(ctxt, width, /*isSigned*/ true));
}

IntegerType *IntegerType::getUnsigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl().unsignedIntegerTypes[width];
  if (ty)
    return ty;
  return ty = (new (ctxt) IntegerType(ctxt, width, /*isSigned*/ false));
}

ReferenceType *ReferenceType::get(ASTContext &ctxt, Type pointee, bool isMut) {
  assert(pointee && "pointee can't be null!");
  size_t hash = llvm::hash_combine(pointee.getPtr(), isMut);
  ReferenceType *&type = ctxt.getImpl().referenceTypes[hash];
  if (type)
    return type;
  ASTContext *canTypeCtxt = pointee->isCanonical() ? &ctxt : nullptr;
  return type = new (ctxt) ReferenceType(canTypeCtxt, pointee, isMut);
}

MaybeType *MaybeType::get(ASTContext &ctxt, Type valueType) {
  MaybeType *&type = ctxt.getImpl().maybeTypes[valueType.getPtr()];
  if (type)
    return type;
  ASTContext *canTypeCtxt = valueType->isCanonical() ? &ctxt : nullptr;
  return type = new (ctxt) MaybeType(canTypeCtxt, valueType);
}

Type TupleType::get(ASTContext &ctxt, ArrayRef<Type> elems) {
  if (elems.empty())
    return getEmpty(ctxt);
  void *insertPos = nullptr;
  llvm::FoldingSetNodeID id;
  Profile(id, elems);
  auto &set = ctxt.getImpl().tupleTypes;

  if (TupleType *type = set.FindNodeOrInsertPos(id, insertPos))
    return type;

  // Only canonical if all elements are.
  bool isCanonical = false;
  for (Type elem : elems)
    isCanonical &= elem->isCanonical();
  ASTContext *canTypeCtxt = isCanonical ? &ctxt : nullptr;

  void *mem =
      ctxt.allocate(totalSizeToAlloc<Type>(elems.size()), alignof(TupleType));
  TupleType *type = new (mem) TupleType(canTypeCtxt, elems);
  set.InsertNode(type, insertPos);
  return type;
}

TupleType *TupleType::getEmpty(ASTContext &ctxt) {
  TupleType *&type = ctxt.getImpl().emptyTupleType;
  if (type)
    return type;
  return type = new (ctxt) TupleType(&ctxt, {});
}

LValueType *LValueType::get(ASTContext &ctxt, Type objectType) {
  LValueType *&type = ctxt.getImpl().lvalueTypes[objectType.getPtr()];
  if (type)
    return type;
  ASTContext *canTypeCtxt = objectType->isCanonical() ? &ctxt : nullptr;
  return type = new (ctxt) LValueType(canTypeCtxt, objectType);
}