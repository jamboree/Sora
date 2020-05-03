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
  /// Sora LValues.
  LValue,

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
/// This type is written "sora.maybe<T>"
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
/// This type is written "sora.reference<T>"
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

/// The IR Representation of lvalues types.
///
/// This type is written "sora.lvalue<T>"
class LValueType : public mlir::Type::TypeBase<LValueType, SoraType,
                                               detail::SingleTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == (unsigned)SoraTypeKind::LValue;
  }

  static LValueType get(mlir::Type objectType);

  mlir::Type getObjectType() const;
};
} // namespace ir
} // namespace sora