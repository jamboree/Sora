//===--- Types.hpp - Sora IR Types Declaration ------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace sora {
namespace ir {
namespace detail {
/// Common storage class for types that only contain another type.
struct SingleTypeStorage;
} // namespace detail

enum class SoraTypeKind {
  First_Type = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,

  /// Sora Maybe Type. (maybe T)
  Maybe,
  /// Sora References. (&T and &mut T)
  Reference,
  /// A non-nullable pointer type.
  Pointer,
  /// Sora 'Void' type (canonical form of '()' as well)
  Void,

  Last_Type = Maybe
};

/// Common base for all Sora IR Types.
class SoraType : public mlir::Type {
public:
  using Type::Type;

  static bool classof(Type type) {
    return type.getKind() >= (unsigned)SoraTypeKind::First_Type &&
           type.getKind() <= (unsigned)SoraTypeKind::Last_Type;
  }
};

/// The IR Representation of "maybe" type.
///
/// This type is written "!sora.maybe<T>"
class MaybeType : public mlir::Type::TypeBase<MaybeType, SoraType,
                                              detail::SingleTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == (unsigned)SoraTypeKind::Maybe;
  }

  static MaybeType get(mlir::Type valueType);

  mlir::Type getValueType() const;
};

/// The IR Representation of references types.
///
/// This type is written "!sora.reference<T>"
class ReferenceType : public mlir::Type::TypeBase<ReferenceType, SoraType,
                                                  detail::SingleTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == (unsigned)SoraTypeKind::Reference;
  }

  static ReferenceType get(mlir::Type pointeeType);

  mlir::Type getPointeeType() const;
};

/// A non-nullable pointer type.
/// This is different from the reference type, which is a "user type", a type
/// that can be written by the user. This cannot be written by the user and is
/// used by the compiler to manipulate memory, for instance, stack allocation.
/// It's also important that there is a distinction between both types as we
/// want to enforce some reference-only semantics in the IR.
///
/// This type is written "!sora.pointer<T>"
class PointerType : public mlir::Type::TypeBase<PointerType, SoraType,
                                                detail::SingleTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == (unsigned)SoraTypeKind::Pointer;
  }

  static PointerType get(mlir::Type objectType);

  mlir::Type getObjectType() const;
};

/// The IR Representation of void types.
///
/// This type is written "!sora.void"
class VoidType : public mlir::Type::TypeBase<VoidType, SoraType> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == (unsigned)SoraTypeKind::Void;
  }

  static VoidType get(mlir::MLIRContext *ctxt) {
    return Base::get(ctxt, (unsigned)SoraTypeKind::Void);
  }
};
} // namespace ir
} // namespace sora