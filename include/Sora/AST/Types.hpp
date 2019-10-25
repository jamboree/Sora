//===--- Types.hpp - Types ASTs ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the whole "type" hierarchy
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/IntegerWidth.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TrailingObjects.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace sora {
class ASTContext;
enum class ArenaKind : uint8_t;

/// Kinds of Types
enum class TypeKind : uint8_t {
#define TYPE(KIND, PARENT) KIND,
#define TYPE_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#define LAST_TYPE(KIND) Last_Type = KIND
#include "Sora/AST/TypeNodes.def"
};

/// Small (byte-sized) class to represent & manipulate type properties.
class TypeProperties {
  static TypeProperties merge() { return {}; }

  using value_t = uint8_t;
  value_t value;

public:
  enum Property : value_t { hasErrorType = 0x01, hasTypeVariable = 0x02 };

  TypeProperties() : TypeProperties(0) {}
  /*implicit*/ TypeProperties(value_t value) : value(value) {}

  TypeProperties operator&(TypeProperties other) const {
    return value_t(value & other.value);
  }

  TypeProperties operator&(value_t other) const {
    return value_t(value & other);
  }

  TypeProperties operator|(TypeProperties other) const {
    return value_t(value | other.value);
  }

  TypeProperties operator|(value_t other) const {
    return value_t(value | other);
  }

  TypeProperties &operator|=(TypeProperties other) {
    value |= other.value;
    return *this;
  }

  TypeProperties &operator|=(value_t other) {
    value |= other;
    return *this;
  }

  /// \returns true if at least one property of this type is activated.
  /// Useful in conjunction with & to check for the presence of properties
  /// \verbatim
  ///   if(type->getTypeProperties() & TypeProperties::foo) { /* ... */ }
  /// \endverbatim
  operator bool() const { return value; }
};

static_assert(sizeof(TypeProperties) == 1, "TypeProperties is too large!");

/// Common base class for Types.
class alignas(TypeBaseAlignement) TypeBase {
  // Disable vanilla new/delete for types
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  /// Make use of the padding bits by allowing derived class to store data here.
  /// NOTE: Derived classes are expected to initialize the bitfields.
  LLVM_PACKED(union Bits {
    Bits() : raw() {}
    // Raw bits (to zero-init the union)
    char raw[6];
    // IntegerType bits
    struct {
      bool isSigned;
      IntegerWidth width;
    } integerType;
    /// FloatType bits
    struct {
      uint8_t floatKind;
    } floatType;
    /// TupleType bits
    struct {
      unsigned numElems;
    } tupleType;
    /// TypeVariableType bits
    struct {
      unsigned id;
    } typeVariableType;
    /// FunctionType bits
    struct {
      unsigned numArgs;
    } functionType;
  });
  static_assert(sizeof(Bits) == 6, "Bits is too large!");

  //===--- 8 Bits ---===//
  const TypeKind kind : 7;
  const bool canonical : 1;
  static_assert(
      unsigned(TypeKind::Last_Type) <= (1 << 7),
      "Not enough bits allocated to the TypeKind to represent every Type Kind");
  //===--------------===//

  TypeProperties properties;

protected:
  Bits bits;

private:
  /// This union always contains the ASTContext for canonical types.
  /// For non-canonical types, it contains the ASTContext if the canonical type
  /// hasn't been calculated yet, else it contains a pointer to the canonical
  /// type.
  mutable llvm::PointerUnion<ASTContext *, TypeBase *> ctxtOrCanType;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  // Also allow allocation of Types using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt, ArenaKind allocator,
                     unsigned align = alignof(TypeBase));

  friend ASTContext; // The ASTContext should be able to allocate types as well

  /// \param kind the kind of type this
  /// \param properties the properties of this type
  /// \param ctxt the ASTContext&
  /// \param canonical whether this type is canonical
  TypeBase(TypeKind kind, TypeProperties properties, ASTContext &ctxt,
           bool canonical)
      : kind(kind), canonical(canonical), properties(properties),
        ctxtOrCanType(&ctxt) {}

public:
  TypeBase(const TypeBase &) = delete;
  void operator=(const TypeBase &) = delete;

  /// \returns whether this type contains an ErrorType
  bool hasErrorType() const {
    return properties & TypeProperties::hasErrorType;
  }

  /// \returns whether this type contains a TypeVariable
  bool hasTypeVariable() const {
    return properties & TypeProperties::hasTypeVariable;
  }

  /// \returns the ASTContext in which this type is allocated
  ASTContext &getASTContext() const {
    if (ASTContext *ctxt = ctxtOrCanType.dyn_cast<ASTContext *>()) {
      assert(ctxt && "ASTContext pointer is null!");
      return *ctxt;
    }
    return ctxtOrCanType.get<TypeBase *>()->getASTContext();
  }

  /// \returns the canonical version of this type
  CanType getCanonicalType() const;

  /// \returns the TypeProperties of this type
  TypeProperties getTypeProperties() const { return properties; }

  /// \returns whether this type is canonical
  bool isCanonical() const { return canonical; }

  /// Prints this type
  void print(raw_ostream &out,
             const TypePrintOptions &printOptions = TypePrintOptions()) const;

  /// Prints this type to a string
  std::string
  getString(const TypePrintOptions &printOptions = TypePrintOptions()) const;

  /// \returns the kind of type this is
  TypeKind getKind() const { return kind; }

  template <typename Ty> Ty *getAs() { return dyn_cast<Ty>(this); }
  template <typename Ty> bool is() { return isa<Ty>(this); }
  template <typename Ty> Ty *castTo() { return cast<Ty>(this); }
};

/// TypeBase should only be 2 pointers in size (kind + padding bits +
/// ctxt/canonical type ptr)
static_assert(sizeof(TypeBase) <= 16, "TypeBase is too large!");

/// Common base class for builtin primitive types.
class BuiltinType : public TypeBase {
protected:
  /// BuiltinTypes have no particular properties.
  BuiltinType(TypeKind kind, ASTContext &ctxt)
      : TypeBase(kind, TypeProperties(), ctxt, /*isCanonical*/ true) {}

public:
  static bool classof(const TypeBase *type) {
    return (type->getKind() >= TypeKind::First_Builtin) &&
           (type->getKind() <= TypeKind::Last_Builtin);
  }
};

/// Integer types
///
/// Used for i8, u8, i16, u16, i32, u32, i64, u64, isize and usize.
///
/// This type is always canonical.
class IntegerType final : public BuiltinType {
  IntegerType(ASTContext &ctxt, IntegerWidth width, bool isSigned)
      : BuiltinType(TypeKind::Integer, ctxt) {
    assert((width.isFixedWidth() || width.isPointerSized()) &&
           "Can only create fixed or pointer-sized integer types!");
    bits.integerType.width = width;
    bits.integerType.isSigned = isSigned;
  }

public:
  /// \returns a signed integer of width \p width
  static IntegerType *getSigned(ASTContext &ctxt, IntegerWidth width);
  /// \returns an unsigned integer of width \p width
  static IntegerType *getUnsigned(ASTContext &ctxt, IntegerWidth width);

  IntegerWidth getWidth() const { return bits.integerType.width; }

  bool isSigned() const { return bits.integerType.isSigned; }
  bool isUnsigned() const { return !isSigned(); }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Integer;
  }
};

/// Kinds of floating point types
enum class FloatKind : uint8_t { IEEE32, IEEE64 };

/// Floating-point types
///
/// Used for f32 and f64.
///
/// This type is always canonical.
class FloatType final : public BuiltinType {
  FloatType(ASTContext &ctxt, FloatKind kind)
      : BuiltinType(TypeKind::Float, ctxt) {
    bits.floatType.floatKind = uint8_t(kind);
    assert(FloatKind(bits.floatType.floatKind) == kind && "bits dropped?");
  }

  friend ASTContext;

public:
  static FloatType *get(ASTContext &ctxt, FloatKind kind);

  unsigned getWidth() {
    switch (getFloatKind()) {
    case FloatKind::IEEE32:
      return 32;
    case FloatKind::IEEE64:
      return 64;
    }
    llvm_unreachable("unknown FloatKind!");
  }

  const llvm::fltSemantics &getAPFloatSemantics() const;

  FloatKind getFloatKind() const { return FloatKind(bits.floatType.floatKind); }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Float;
  }
};

/// Void type
///
/// Used for 'void', and also the canonical form of '()' (empty tuple type)
///
/// This type is always canonical.
class VoidType final : public BuiltinType {
  VoidType(ASTContext &ctxt) : BuiltinType(TypeKind::Void, ctxt) {}
  friend ASTContext;

public:
  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Void;
  }
};

/// Reference Type
///
/// Non-nullable pointer type, spelled '&T' or '&mut T'.
///
/// This type is canonical only if T is.
class ReferenceType final : public TypeBase {
  llvm::PointerIntPair<TypeBase *, 1> pointeeAndIsMut;

  ReferenceType(TypeProperties props, ASTContext &ctxt, Type pointee,
                bool isMut)
      : TypeBase(TypeKind::Reference, props, ctxt, pointee->isCanonical()),
        pointeeAndIsMut(pointee.getPtr(), isMut) {}

public:
  static ReferenceType *get(Type pointee, bool isMut);

  Type getPointeeType() const { return pointeeAndIsMut.getPointer(); }
  bool isMut() const { return pointeeAndIsMut.getInt(); }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Reference;
  }
};

/// Maybe type
///
/// A type representing an optional value, spelled 'maybe T'
///
/// This type is canonical only if T is.
class MaybeType final : public TypeBase {
  Type valueType;

  MaybeType(TypeProperties prop, ASTContext &ctxt, Type valueType)
      : TypeBase(TypeKind::Maybe, prop, ctxt, valueType->isCanonical()),
        valueType(valueType) {}

public:
  static MaybeType *get(Type valueType);

  Type getValueType() const { return valueType; }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Maybe;
  }
};

/// Tuple Type
///
/// Tuple of 0 or 2+ elements.
///
/// This type isn't canonical if it has 0 elements, else it's canonical only if
/// every element inside it is. The canonical version of the empty tuple type is
/// 'void'
class TupleType final : public TypeBase,
                        public llvm::FoldingSetNode,
                        private llvm::TrailingObjects<TupleType, Type> {
  friend llvm::TrailingObjects<TupleType, Type>;

  TupleType(TypeProperties prop, ASTContext &ctxt, bool canonical,
            ArrayRef<Type> elements)
      : TypeBase(TypeKind::Tuple, prop, ctxt, canonical) {
    unsigned numElems = elements.size();
    assert(numElems >= 2 || (numElems == 0 && canonical));
    bits.tupleType.numElems = numElems;
    std::uninitialized_copy(elements.begin(), elements.end(),
                            getTrailingObjects<Type>());
#ifndef NDEBUG
    bool shouldBeCanonical = true;
    for (auto elem : elements)
      shouldBeCanonical &= elem->isCanonical();
    assert(shouldBeCanonical == canonical && "Type canonicality is incorrect");
#endif
  }

public:
  /// \returns the uniqued tuple type type for \p elems. If \p elems has a
  /// single element, returns it instead. This is done to avoid creation of a
  /// single-element tuple type, as it's not a type you can write in Sora.
  static Type get(ASTContext &ctxt, ArrayRef<Type> elems);

  /// \returns the empty tuple type.
  static TupleType *getEmpty(ASTContext &ctxt);

  bool isEmpty() const { return getNumElements() == 0; }
  unsigned getNumElements() const { return bits.tupleType.numElems; }
  ArrayRef<Type> getElements() const {
    return {getTrailingObjects<Type>(), getNumElements()};
  }

  void Profile(llvm::FoldingSetNodeID &id) { Profile(id, getElements()); }
  static void Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> elements);

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Tuple;
  }
};

/// Function Type
///
/// The type of a function when referenced inside an expression.
///
/// This class directly stores the argument types as trailing objects instead of
/// relying on something like TupleType because it facilitates canonicalization;
///
/// This type is canonical only if the arguments and return type are.
class FunctionType final : public TypeBase,
                           private llvm::TrailingObjects<FunctionType, Type>,
                           public llvm::FoldingSetNode {
  friend llvm::TrailingObjects<FunctionType, Type>;

  FunctionType(TypeProperties properties, ASTContext &ctxt, bool canonical,
               ArrayRef<Type> args, Type rtr)
      : TypeBase(TypeKind::Function, properties, ctxt, canonical), rtr(rtr) {
    bits.functionType.numArgs = args.size();
    std::uninitialized_copy(args.begin(), args.end(),
                            getTrailingObjects<Type>());
#ifndef NDEBUG
    bool shouldBeCanonical = rtr->isCanonical();
    for (auto arg : args)
      shouldBeCanonical &= arg->isCanonical();
    assert(shouldBeCanonical == canonical && "Type canonicality is incorrect");
#endif
  }

  Type rtr;

public:
  static FunctionType *get(ArrayRef<Type> args, Type rtr);

  unsigned getNumArgs() const { return bits.functionType.numArgs; }
  ArrayRef<Type> getArgs() const {
    return {getTrailingObjects<Type>(), getNumArgs()};
  }

  Type getReturnType() const { return rtr; }

  void Profile(llvm::FoldingSetNodeID &id) { Profile(id, getArgs(), rtr); }
  static void Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> args,
                      Type rtr);

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Function;
  }
};

/// LValue type
///
/// The LValue or "left-value" (as in something that can appear to the left
/// of an assignement) represents an handle to a physical object (something that
/// can be written to).
///
/// LValues aren't really first-class types. They only exist when you access a
/// mutable member through a mutable reference, or when you refer to a mutable
/// variable. Example:
///
/// \verbatim
///   let x = 0
///   let mut y = 0
///   x // has type i32
///   y // has type LValue(i32)
/// \endverbatim
///
/// This type is canonical only if T is.
class LValueType final : public TypeBase {
  Type objectType;

  LValueType(TypeProperties prop, ASTContext &ctxt, Type objectType)
      : TypeBase(TypeKind::LValue, prop, ctxt, objectType->isCanonical()),
        objectType(objectType) {}

public:
  static LValueType *get(Type objectType);

  Type getObjectType() const { return objectType; }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::LValue;
  }
};

/// Error Type
///
/// The type of an expression whose type couldn't be inferred, and the type of
/// ErrorExprs.
///
/// This type is always canonical.
class ErrorType final : public TypeBase {
  ErrorType(ASTContext &ctxt)
      : TypeBase(TypeKind::Error, TypeProperties::hasErrorType, ctxt,
                 /*isCanonical*/ true) {}
  friend ASTContext;

public:
  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Error;
  }
};

/// Type Variable Type
///
/// Used by the typechecker. This type is never unique, and is always allocated
/// in the ASTContext's TypeChecker allocator.
///
/// This type is always canonical.
class TypeVariableType final : public TypeBase {
  TypeVariableType(ASTContext &ctxt, unsigned id)
      : TypeBase(TypeKind::TypeVariable, TypeProperties::hasTypeVariable, ctxt,
                 /*isCanonical*/ true) {
    bits.typeVariableType.id = id;
  }

  friend ASTContext;

public:
  static TypeVariableType *create(ASTContext &ctxt, unsigned id);

  unsigned getID() const { return bits.typeVariableType.id; }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::TypeVariable;
  }
};
} // namespace sora