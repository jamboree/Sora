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
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Error.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace sora {
class ASTContext;

/// Kinds of Types
enum class TypeKind : uint8_t {
#define TYPE(KIND, PARENT) KIND,
#define TYPE_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/TypeNodes.def"
};

/// Common base class for Types.
class alignas(TypeBaseAlignement) TypeBase {
  // Disable vanilla new/delete for types
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  TypeKind kind;
  bool canonical = false;
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
  });
  static_assert(sizeof(Bits) == 6, "Bits is too large!");

protected:
  Bits bits;

private:
  /// This union contains the ASTContext for canonical types, and a (potentially
  /// null (if not computed yet)) pointer to the canonical type for
  /// non-canonical types.
  union {
    ASTContext *ctxt;
    TypeBase *ptr;
  };

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  // Also allow allocation of Types using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(TypeBase));

  friend ASTContext; // The ASTContext should be able to allocate types as well

  /// \param kind the kind of type this is
  /// \param canTypeCtxt for canonical types, the ASTContext
  TypeBase(TypeKind kind, ASTContext *canTypeCtxt) : kind(kind), ctxt(nullptr) {
    if (ctxt) {
      canonical = true;
      ctxt = canTypeCtxt;
    }
  }

public:
  TypeBase(const TypeBase &) = delete;
  void operator=(const TypeBase &) = delete;

  /// \returns true if this type is canonical
  bool isCanonical() const { return canonical; }

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
  BuiltinType(TypeKind kind, ASTContext *canTypeCtxt)
      : TypeBase(kind, canTypeCtxt) {}

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
      : BuiltinType(TypeKind::Integer, &ctxt) {
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
      : BuiltinType(TypeKind::Float, &ctxt) {
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
  VoidType(ASTContext &ctxt) : BuiltinType(TypeKind::Void, &ctxt) {}
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

  ReferenceType(ASTContext *canTypeCtxt, Type pointee, bool isMut)
      : TypeBase(TypeKind::Reference, canTypeCtxt),
        pointeeAndIsMut(pointee.getPtr(), isMut) {
    assert(bool(canTypeCtxt) == pointee->isCanonical() &&
           "if the type is canonical, the ASTContext* must not be null, else "
           "it must be null.");
  }

public:
  static ReferenceType *get(ASTContext &ctxt, Type pointee, bool isMut);

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

  MaybeType(ASTContext *canTypeCtxt, Type valueType)
      : TypeBase(TypeKind::Maybe, canTypeCtxt), valueType(valueType) {
    assert(bool(canTypeCtxt) == valueType->isCanonical() &&
           "if the type is canonical, the ASTContext* must not be null, else "
           "it must be null.");
  }

public:
  static MaybeType *get(ASTContext &ctxt, Type valueType);

  Type getValueType() const { return valueType; }

  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Maybe;
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

  LValueType(ASTContext *canTypeCtxt, Type objectType)
      : TypeBase(TypeKind::LValue, canTypeCtxt), objectType(objectType) {
    assert(bool(canTypeCtxt) == objectType->isCanonical() &&
           "if the type is canonical, the ASTContext* must not be null, else "
           "it must be null.");
  }

public:
  static LValueType *get(ASTContext &ctxt, Type objectType);

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
  ErrorType(ASTContext &ctxt) : TypeBase(TypeKind::Error, &ctxt) {}
  friend ASTContext;

public:
  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Error;
  }
};
} // namespace sora