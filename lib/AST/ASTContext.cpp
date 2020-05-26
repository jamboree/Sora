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

//===- ASTContext::Impl ---------------------------------------------------===//

struct ASTContext::Impl {
  /// The Identifier Table
  /// FIXME: Ideally, this should use the permanent arena.
  llvm::StringSet<llvm::BumpPtrAllocator> identifierTable;

  /// The set of cleanups that must be ran when the ASTContext is destroyed.
  SmallVector<std::function<void()>, 4> cleanups;

  /// An arena that doesn't support type allocation.
  using Arena = llvm::BumpPtrAllocator;
  /// An Arena that supports type allocation
  struct TypeArena : public Arena {
    TypeArena() = default;

    /// \returns the memory used (in bytes) by this Arena's contents.
    size_t getTotalMemory() const {
      size_t value = 0;
      value += Arena::getTotalMemory();
      value += llvm::capacity_in_bytes(signedIntegerTypes);
      value += llvm::capacity_in_bytes(unsignedIntegerTypes);
      value += llvm::capacity_in_bytes(referenceTypes);
      value += llvm::capacity_in_bytes(maybeTypes);
      // tupleTypes? FoldingSet doesn't provide a function to calculate the
      // memory used.
      value += llvm::capacity_in_bytes(lvalueTypes);
      return value;
    }

    /// Signed Integer Types
    llvm::DenseMap<IntegerWidth::opaque_t, IntegerType *> signedIntegerTypes;
    /// Unsigned Integer Types
    llvm::DenseMap<IntegerWidth::opaque_t, IntegerType *> unsignedIntegerTypes;
    /// Reference types
    llvm::DenseMap<size_t, ReferenceType *> referenceTypes;
    /// Maybe types
    llvm::DenseMap<TypeBase *, MaybeType *> maybeTypes;
    /// Tuple types
    llvm::FoldingSet<TupleType> tupleTypes;
    /// Function types
    llvm::FoldingSet<FunctionType> functionTypes;
    /// LValue types
    llvm::DenseMap<TypeBase *, LValueType *> lvalueTypes;
  };

  /// for ArenaKind::Permanent
  TypeArena permanentArena;
  ///   The empty TupleType is only allocated once inside the Permanent Arena.
  TupleType *emptyTupleType = nullptr;
  /// for ArenaKind::UnresolvedExpr
  llvm::BumpPtrAllocator unresolvedExprArena;
  /// for ArenaKind::ConstraintSystem
  Optional<TypeArena> constraintSystemArena;

  // Built-in types lookup map
  llvm::DenseMap<Identifier, CanType> builtinTypesLookupMap;

  void initConstraintSystemArena() {
    assert(!hasConstraintSystemArena() &&
           "ConstraintSystem arena already active");
    constraintSystemArena.emplace();
  }

  void destroyConstraintSystemArena() {
    assert(hasConstraintSystemArena() && "ConstraintSystem arena not active");
    constraintSystemArena.reset();
  }

  bool hasConstraintSystemArena() const {
    return constraintSystemArena.hasValue();
  }

  /// \returns the TypeArena for \p kind. \p kind can't be UnresolvedExpr!
  TypeArena &getTypeArena(ArenaKind kind) {
    switch (kind) {
    case ArenaKind::Permanent:
      return permanentArena;
    case ArenaKind::ConstraintSystem:
      assert(hasConstraintSystemArena() &&
             "ConstraintSystem allocator isn't active!");
      return *constraintSystemArena;
    case ArenaKind::UnresolvedExpr:
      llvm_unreachable(
          "Can't allocate types inside the UnresolvedExpr allocator!");
    }
    llvm_unreachable("Unknown ArenaKind");
  }

  /// The target triple
  llvm::Triple targetTriple;

  /// \returns the total memory used (in bytes) by this ASTContext::Impl and all
  /// of its arenas.
  size_t getTotalMemoryUsed() const;
  /// \returns the memory used (in bytes) by \p arena.
  size_t getMemoryUsed(ArenaKind arena) const;

  Impl(const Impl &) = delete;
  Impl &operator=(const Impl &) = delete;

  Impl() {
    // Init with a default target triple
    targetTriple = llvm::Triple(llvm::sys::getDefaultTargetTriple());
  }

  /// Custom destructor that runs the cleanups.
  ~Impl() {
    assert(!hasConstraintSystemArena() &&
           "Destroying the ASTContext while the ConstraintSystem allocator is "
           "active!");
    for (auto &cleanup : cleanups)
      cleanup();
  }
};

size_t ASTContext::Impl::getTotalMemoryUsed() const {
  size_t value = sizeof(Impl);
  value += identifierTable.getAllocator().getTotalMemory();
  value += llvm::capacity_in_bytes(cleanups);
  value += getMemoryUsed(ArenaKind::Permanent);
  value += getMemoryUsed(ArenaKind::UnresolvedExpr);
  value += getMemoryUsed(ArenaKind::ConstraintSystem);
  return value;
}

size_t ASTContext::Impl::getMemoryUsed(ArenaKind arena) const {
  switch (arena) {
  case ArenaKind::Permanent:
    return permanentArena.getTotalMemory();
  case ArenaKind::UnresolvedExpr:
    return unresolvedExprArena.getTotalMemory();
  case ArenaKind::ConstraintSystem:
    if (!constraintSystemArena)
      return 0;
    return constraintSystemArena->getTotalMemory();
  }
  llvm_unreachable("Unknown ArenaKind");
}

//===- RAIIConstraintSystemArena ------------------------------------------===//

RAIIConstraintSystemArena::RAIIConstraintSystemArena(ASTContext &ctxt)
    : ctxt(ctxt) {
  ctxt.getImpl().initConstraintSystemArena();
}

RAIIConstraintSystemArena::~RAIIConstraintSystemArena() {
  ctxt.getImpl().destroyConstraintSystemArena();
}

//===- Utilities ----------------------------------------------------------===//

static IntegerWidth getPointerWidth(ASTContext &ctxt) {
  return IntegerWidth::pointer(ctxt.getTargetTriple());
}

//===- ASTContext ---------------------------------------------------------===//

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
      nullType(new (*this, ArenaKind::Permanent) NullType(*this)),
      boolType(new (*this, ArenaKind::Permanent) BoolType(*this)),
      errorType(new (*this, ArenaKind::Permanent) ErrorType(*this)) {
  // Build the BuiltinTypes lookup map
  // FIXME: Can't this be built lazily?
  buildBuiltinTypesLookupMap();
}

void ASTContext::buildBuiltinTypesLookupMap() {
  auto &map = getImpl().builtinTypesLookupMap;
  assert(map.empty() && "BuiltinTypes lookup map already created");
#define BUILTIN_TYPE(T) map.insert({getIdentifier(#T), T##Type})
  BUILTIN_TYPE(i8);
  BUILTIN_TYPE(i16);
  BUILTIN_TYPE(i32);
  BUILTIN_TYPE(i64);
  BUILTIN_TYPE(isize);

  BUILTIN_TYPE(u8);
  BUILTIN_TYPE(u16);
  BUILTIN_TYPE(u32);
  BUILTIN_TYPE(u64);
  BUILTIN_TYPE(usize);

  BUILTIN_TYPE(f32);
  BUILTIN_TYPE(f64);

  BUILTIN_TYPE(void);
  BUILTIN_TYPE(bool);
#undef BUILTIN_TYPE
}

ASTContext::Impl &ASTContext::getImpl() {
  return *reinterpret_cast<Impl *>(
      llvm::alignAddr(this + 1, llvm::Align(alignof(Impl))));
}

std::unique_ptr<ASTContext> ASTContext::create(const SourceManager &srcMgr,
                                               DiagnosticEngine &diagEngine) {
  // We need to allocate enough memory to support both the ASTContext and
  // its implementation *plus* some padding to align the addresses
  // correctly.
  size_t sizeToAlloc = sizeof(ASTContext) + (alignof(ASTContext) - 1);
  sizeToAlloc += sizeof(Impl) + (alignof(Impl) - 1);

  // FIXME: Replace this with aligned_alloc or similar
  void *memory = llvm::safe_malloc(sizeToAlloc);

  // The ASTContext's memory begins at the first correctly aligned address
  // of the memory.
  void *astContextMemory = reinterpret_cast<void *>(
      llvm::alignAddr(memory, llvm::Align(alignof(ASTContext))));

  // The Impl's memory begins at the first correctly aligned address after
  // the ASTContext's memory.
  void *implMemory = (char *)astContextMemory + sizeof(ASTContext);
  implMemory = reinterpret_cast<void *>(
      llvm::alignAddr(implMemory, llvm::Align(alignof(Impl))));

  // Do some checking because I'm kinda paranoïd.
  //  Check that we aren't going out of bounds and going to segfault later.
  assert(((char *)implMemory + sizeof(Impl)) < ((char *)memory + sizeToAlloc) &&
         "Going out-of-bounds of the allocated memory");
  //  Check that the ASTContext's memory doesn't overlap the
  //  Implementation's.
  assert((((char *)astContextMemory + sizeof(ASTContext)) <= implMemory) &&
         "ASTContext's memory overlaps the Impl's memory");

  // Use placement new to call the constructors.
  // It is very important that the implementation is initialized first.
  new (implMemory) Impl();
  ASTContext *astContext =
      new (astContextMemory) ASTContext(srcMgr, diagEngine);

  // Put the pointer in an unique_ptr to avoid leaks.
  return std::unique_ptr<ASTContext>(astContext);
}

ASTContext::~ASTContext() { getImpl().~Impl(); }

llvm::BumpPtrAllocator &ASTContext::getArena(ArenaKind kind) {
  switch (kind) {
  case ArenaKind::Permanent:
    return getImpl().permanentArena;
  case ArenaKind::UnresolvedExpr:
    return getImpl().unresolvedExprArena;
  case ArenaKind::ConstraintSystem:
    assert(getImpl().hasConstraintSystemArena() &&
           "ConstraintSystem arena not active!");
    return *getImpl().constraintSystemArena;
  }
  llvm_unreachable("unknown allocator kind");
}

bool ASTContext::hasConstraintSystemArena() const {
  return getImpl().hasConstraintSystemArena();
}

RAIIConstraintSystemArena ASTContext::createConstraintSystemArena() {
  if (hasConstraintSystemArena())
    llvm_unreachable("Only one ConstraintSystem Arena can exist at a time!");
  return {*this};
}

void ASTContext::freeUnresolvedExprs() {
  getImpl().unresolvedExprArena.Reset();
}

size_t ASTContext::getTotalMemoryUsed() const {
  size_t value = getImpl().getTotalMemoryUsed();
  value += sizeof(ASTContext);
  return value;
}

size_t ASTContext::getMemoryUsed(ArenaKind arena) const {
  return getImpl().getMemoryUsed(arena);
}

void ASTContext::addCleanup(std::function<void()> cleanup) {
  getImpl().cleanups.push_back(cleanup);
}

Identifier ASTContext::getIdentifier(StringRef str) {
  // Don't intern null & empty strings
  return str.size() ? getImpl().identifierTable.insert(str).first->getKeyData()
                    : Identifier();
}

void ASTContext::overrideTargetTriple(const llvm::Triple &triple) {
  getImpl().targetTriple = triple;
}

llvm::Triple ASTContext::getTargetTriple() const {
  return getImpl().targetTriple;
}

CanType ASTContext::lookupBuiltinType(Identifier ident) const {
  if (!ident)
    return CanType(nullptr);
  auto &map = getImpl().builtinTypesLookupMap;
  auto it = map.find(ident);
  if (it == map.end())
    return CanType(nullptr);
  return it->second;
}

void ASTContext::getAllBuiltinTypes(SmallVectorImpl<Type> &results) const {
  auto &map = getImpl().builtinTypesLookupMap;
  results.reserve(map.size());
  for (auto entry : map)
    results.push_back(entry.second);
}

void ASTContext::getAllBuiltinTypes(SmallVectorImpl<CanType> &results) const {
  auto &map = getImpl().builtinTypesLookupMap;
  results.reserve(map.size());
  for (auto entry : map)
    results.push_back(entry.second);
}

//===- Types --------------------------------------------------------------===//

/// \returns The ArenaKind to use for a type using \p properties.
static ArenaKind getArena(TypeProperties properties) {
  return (properties & TypeProperties::hasTypeVariable)
             ? ArenaKind::ConstraintSystem
             : ArenaKind::Permanent;
}

IntegerType *IntegerType::getSigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl()
                         .getTypeArena(ArenaKind::Permanent)
                         .signedIntegerTypes[width.getOpaqueValue()];
  if (ty)
    return ty;
  return ty = (new (ctxt, ArenaKind::Permanent)
                   IntegerType(ctxt, width, /*isSigned*/ true));
}

IntegerType *IntegerType::getUnsigned(ASTContext &ctxt, IntegerWidth width) {
  IntegerType *&ty = ctxt.getImpl()
                         .getTypeArena(ArenaKind::Permanent)
                         .unsignedIntegerTypes[width.getOpaqueValue()];
  if (ty)
    return ty;
  return ty = (new (ctxt, ArenaKind::Permanent)
                   IntegerType(ctxt, width, /*isSigned*/ false));
}

ReferenceType *ReferenceType::get(Type pointee, bool isMut) {
  assert(pointee && "pointee type can't be null!");
  ASTContext &ctxt = pointee->getASTContext();

  size_t typeID = llvm::hash_combine(pointee.getPtr(), isMut);
  auto props = pointee->getTypeProperties();
  auto arena = getArena(props);

  ReferenceType *&type =
      ctxt.getImpl().getTypeArena(arena).referenceTypes[typeID];
  if (type)
    return type;
  return type = new (ctxt, arena) ReferenceType(props, ctxt, pointee, isMut);
}

MaybeType *MaybeType::get(Type valueType) {
  assert(valueType && "value type is null");
  ASTContext &ctxt = valueType->getASTContext();

  auto props = valueType->getTypeProperties();
  auto arena = getArena(props);

  MaybeType *&type =
      ctxt.getImpl().getTypeArena(arena).maybeTypes[valueType.getPtr()];

  if (type)
    return type;
  return type = new (ctxt, arena) MaybeType(props, ctxt, valueType);
}

Type TupleType::get(ASTContext &ctxt, ArrayRef<Type> elems) {
  // For empty tuples, don't bother doing any research, just return the
  // singleton stored in the impl.
  if (elems.empty())
    return getEmpty(ctxt);
  // For single-element tuples, we don't create a tuple. Just return the
  // element.
  // FIXME: If, in the future, something like ParenType is added, return that
  // instead.
  if (elems.size() == 1)
    return elems[0];

  // Determine the properties of this type
  bool isCanonical = true;
  TypeProperties props;
  for (Type elem : elems) {
    assert(elem && "elem is null");
    isCanonical &= elem->isCanonical();
    props |= elem->getTypeProperties();
  }

  auto &typeArena = ctxt.getImpl().getTypeArena(getArena(props));
  void *insertPos = nullptr;
  llvm::FoldingSetNodeID id;
  Profile(id, elems);
  auto &set = typeArena.tupleTypes;

  if (TupleType *type = set.FindNodeOrInsertPos(id, insertPos))
    return type;

  void *mem = typeArena.Allocate(totalSizeToAlloc<Type>(elems.size()),
                                 alignof(TupleType));
  TupleType *type = new (mem) TupleType(props, ctxt, isCanonical, elems);
  set.InsertNode(type, insertPos);
  return type;
}

TupleType *TupleType::getEmpty(ASTContext &ctxt) {
  TupleType *&type = ctxt.getImpl().emptyTupleType;
  if (type)
    return type;
  return type = new (ctxt, ArenaKind::Permanent)
             TupleType(TypeProperties(), ctxt, /*isCanonical*/ false, {});
}

LValueType *LValueType::get(Type objectType) {
  assert(objectType && "object type is null");
  ASTContext &ctxt = objectType->getASTContext();

  auto props = objectType->getTypeProperties() | TypeProperties::hasLValue;
  auto arena = getArena(props);

  LValueType *&type =
      ctxt.getImpl().getTypeArena(arena).lvalueTypes[objectType.getPtr()];
  if (type)
    return type;
  return type = new (ctxt, arena) LValueType(props, ctxt, objectType);
}

FunctionType *FunctionType::get(ArrayRef<Type> args, Type rtr) {
  assert(rtr && "return type is null");
  ASTContext &ctxt = rtr->getASTContext();

  TypeProperties props = rtr->getTypeProperties();
  bool isCanonical = rtr->isCanonical();
  for (Type arg : args) {
    assert(arg && "arg type is null");
    isCanonical &= arg->isCanonical();
    props |= arg->getTypeProperties();
  }

  auto &typeArena = ctxt.getImpl().getTypeArena(getArena(props));

  void *insertPos = nullptr;
  llvm::FoldingSetNodeID id;
  Profile(id, args, rtr);
  auto &set = typeArena.functionTypes;

  if (FunctionType *type = set.FindNodeOrInsertPos(id, insertPos))
    return type;

  void *mem = typeArena.Allocate(totalSizeToAlloc<Type>(args.size()),
                                 alignof(FunctionType));
  FunctionType *type =
      new (mem) FunctionType(props, ctxt, isCanonical, args, rtr);
  set.InsertNode(type, insertPos);
  return type;
}

void *TypeVariableType::operator new(size_t size, ASTContext &ctxt,
                                     unsigned align) {
  return TypeBase::operator new(size, ctxt, ArenaKind::ConstraintSystem, align);
}
