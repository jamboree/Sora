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
#include "llvm/ADT/Optional.h"
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
  /// The Identifier Table
  /// FIXME: Is using MallocAllocator the right thing to do here?
  llvm::StringSet<> identifierTable;

  /// The set of cleanups that must be ran when the ASTContext is destroyed.
  SmallVector<std::function<void()>, 4> cleanups;

  /// An arena that doesn't support type allocation.
  using Arena = llvm::BumpPtrAllocator;
  /// An Arena that supports type allocation
  struct TypeArena : public Arena {
    TypeArena() = default;

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
  };

  /// for ArenaKind::Permanent
  TypeArena permanentArena;
  ///   The empty TupleType is only allocated once inside the Permanent Arena.
  TupleType *emptyTupleType = nullptr;
  /// for ArenaKind::UnresolvedExpr
  llvm::BumpPtrAllocator unresolvedExprArena;
  /// for ArenaKind::TypeChecker
  Optional<TypeArena> typeCheckerArena;

  void initTypeCheckerArena() {
    assert(!hasTypeCheckerArena() && "TypeChecker arena already active");
    typeCheckerArena.emplace();
  }

  void destroyTypeCheckerArena() {
    assert(hasTypeCheckerArena() && "TypeChecker arena not active");
    typeCheckerArena.reset();
  }

  bool hasTypeCheckerArena() const { return typeCheckerArena.hasValue(); }

  /// \returns the TypeArena for \p kind. \p kind can't be UnresolvedExpr!
  TypeArena &getTypeArena(ArenaKind kind) {
    switch (kind) {
    case ArenaKind::Permanent:
      return permanentArena;
    case ArenaKind::TypeChecker:
      assert(hasTypeCheckerArena() && "TypeChecker allocator isn't active!");
      return *typeCheckerArena;
    case ArenaKind::UnresolvedExpr:
      llvm_unreachable(
          "Can't allocate types inside the UnresolvedExpr allocator!");
    }
    llvm_unreachable("Unknown ArenaKind");
  }

  /// The target triple
  llvm::Triple targetTriple;

  Impl(const Impl &) = delete;
  Impl &operator=(const Impl &) = delete;

  Impl() {
    // Init with a default target triple
    targetTriple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
  }

  /// Custom destructor that runs the cleanups.
  ~Impl() {
    assert(
        !hasTypeCheckerArena() &&
        "Destroying the ASTContext while the TypeChecker allocator is active!");
    for (auto &cleanup : cleanups)
      cleanup();
  }
};

TypeCheckerArenaRAII::TypeCheckerArenaRAII(ASTContext &ctxt) : ctxt(ctxt) {
  ctxt.getImpl().initTypeCheckerArena();
}

TypeCheckerArenaRAII::~TypeCheckerArenaRAII() {
  ctxt.getImpl().destroyTypeCheckerArena();
}

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
      f32Type(new (*this, ArenaKind::Permanent)
                  FloatType(*this, FloatKind::IEEE32)),
      f64Type(new (*this, ArenaKind::Permanent)
                  FloatType(*this, FloatKind::IEEE64)),
      voidType(new (*this, ArenaKind::Permanent) VoidType(*this)),
      errorType(new (*this, ArenaKind::Permanent) ErrorType(*this)) {}

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

llvm::BumpPtrAllocator &ASTContext::getArena(ArenaKind kind) {
  switch (kind) {
  case ArenaKind::Permanent:
    return getImpl().permanentArena;
  case ArenaKind::UnresolvedExpr:
    return getImpl().unresolvedExprArena;
  case ArenaKind::TypeChecker:
    assert(getImpl().hasTypeCheckerArena() && "TypeChecker arena not active!");
    return *getImpl().typeCheckerArena;
  }
  llvm_unreachable("unknown allocator kind");
}

bool ASTContext::hasTypeCheckerArena() const {
  return getImpl().hasTypeCheckerArena();
}

void ASTContext::freeUnresolvedExprs() {
  getImpl().unresolvedExprArena.Reset();
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

/// \returns The ArenaKind to use for a type using \p properties.
static ArenaKind getArena(TypeProperties properties) {
  return (properties & TypeProperties::hasTypeVariable) ? ArenaKind::TypeChecker
                                                        : ArenaKind::Permanent;
}

IntegerType *IntegerType::getSigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl()
                         .getTypeArena(ArenaKind::Permanent)
                         .signedIntegerTypes[width];
  if (ty)
    return ty;
  return ty = (new (ctxt, ArenaKind::Permanent)
                   IntegerType(ctxt, width, /*isSigned*/ true));
}

IntegerType *IntegerType::getUnsigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl()
                         .getTypeArena(ArenaKind::Permanent)
                         .unsignedIntegerTypes[width];
  if (ty)
    return ty;
  return ty = (new (ctxt, ArenaKind::Permanent)
                   IntegerType(ctxt, width, /*isSigned*/ false));
}

ReferenceType *ReferenceType::get(ASTContext &ctxt, Type pointee, bool isMut) {
  assert(pointee && "pointee can't be null!");
  size_t typeID = llvm::hash_combine(pointee.getPtr(), isMut);

  auto props = pointee->getTypeProperties();
  auto arena = getArena(props);

  ReferenceType *&type =
      ctxt.getImpl().getTypeArena(arena).referenceTypes[typeID];
  if (type)
    return type;
  ASTContext *canTypeCtxt = pointee->isCanonical() ? &ctxt : nullptr;
  return type = new (ctxt, arena)
             ReferenceType(props, canTypeCtxt, pointee, isMut);
}

MaybeType *MaybeType::get(ASTContext &ctxt, Type valueType) {
  auto props = valueType->getTypeProperties();
  auto arena = getArena(props);

  MaybeType *&type =
      ctxt.getImpl().getTypeArena(arena).maybeTypes[valueType.getPtr()];

  if (type)
    return type;
  ASTContext *canTypeCtxt = valueType->isCanonical() ? &ctxt : nullptr;
  return type = new (ctxt, arena) MaybeType(props, canTypeCtxt, valueType);
}

Type TupleType::get(ASTContext &ctxt, ArrayRef<Type> elems) {
  if (elems.empty())
    return getEmpty(ctxt);

  // Determine the properties of this type
  bool isCanonical = false;
  TypeProperties props;
  for (Type elem : elems) {
    // Only canonical if all elements are
    isCanonical &= elem->isCanonical();
    // Properties are or'd together
    props |= elem->getTypeProperties();
  }

  auto &typeArena = ctxt.getImpl().getTypeArena(getArena(props));
  void *insertPos = nullptr;
  llvm::FoldingSetNodeID id;
  Profile(id, elems);
  auto &set = typeArena.tupleTypes;

  if (TupleType *type = set.FindNodeOrInsertPos(id, insertPos))
    return type;

  ASTContext *canTypeCtxt = isCanonical ? &ctxt : nullptr;

  void *mem = typeArena.Allocate(totalSizeToAlloc<Type>(elems.size()),
                                 alignof(TupleType));
  TupleType *type = new (mem) TupleType(props, canTypeCtxt, elems);
  set.InsertNode(type, insertPos);
  return type;
}

TupleType *TupleType::getEmpty(ASTContext &ctxt) {
  TupleType *&type = ctxt.getImpl().emptyTupleType;
  if (type)
    return type;
  return type = new (ctxt, ArenaKind::Permanent)
             TupleType(TypeProperties(), &ctxt, {});
}

LValueType *LValueType::get(ASTContext &ctxt, Type objectType) {
  auto props = objectType->getTypeProperties();
  auto arena = getArena(props);

  LValueType *&type =
      ctxt.getImpl().getTypeArena(arena).lvalueTypes[objectType.getPtr()];
  if (type)
    return type;
  ASTContext *canTypeCtxt = objectType->isCanonical() ? &ctxt : nullptr;
  return type = new (ctxt, arena) LValueType(props, canTypeCtxt, objectType);
}

TypeVariableType *TypeVariableType::create(ASTContext &ctxt, unsigned id) {
  return new (ctxt, ArenaKind::TypeChecker) TypeVariableType(ctxt, id);
}