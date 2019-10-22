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
  /// Make use of the padding bits by allowing derived class to store data here.
  /// NOTE: Derived classes are expected to initialize the bitfields.
  LLVM_PACKED(union Bits {
    Bits() : raw() {}
    // Raw bits (to zero-init the union)
    char raw[7];
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
  static_assert(sizeof(Bits) == 7, "Bits is too large!");

protected:
  Bits bits;

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

  TypeBase(TypeKind kind) : kind(kind) {}

public:
  /// \returns the kind of type this is
  TypeKind getKind() const { return kind; }

  template <typename Ty> Ty *getAs() { return dyn_cast<Ty>(this); }
  template <typename Ty> bool is() { return isa<Ty>(this); }
  template <typename Ty> Ty *castTo() { return cast<Ty>(this); }
};

/// TypeBase should only be one pointer in size (kind + padding bits)
static_assert(sizeof(TypeBase) <= 8, "TypeBase is too large!");

/// Common base class for builtin primitive types.
class BuiltinType : public TypeBase {
protected:
  BuiltinType(TypeKind kind) : TypeBase(kind) {}

public:
  static bool classof(const TypeBase *type) {
    return (type->getKind() >= TypeKind::First_Builtin) &&
           (type->getKind() <= TypeKind::Last_Builtin);
  }
};

/// Integer types
///
/// Used for i8, u8, i16, u16, i32, u32, i64, u64, isize and usize.
class IntegerType final : public BuiltinType {
  IntegerType(IntegerWidth width, bool isSigned)
      : BuiltinType(TypeKind::Integer) {
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
class FloatType final : public BuiltinType {
  FloatType(FloatKind kind) : BuiltinType(TypeKind::Float) {
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
class VoidType final : public BuiltinType {
  VoidType() : BuiltinType(TypeKind::Void) {}

  friend ASTContext;

public:
  static bool classof(const TypeBase *type) {
    return type->getKind() == TypeKind::Void;
  }
};
} // namespace sora