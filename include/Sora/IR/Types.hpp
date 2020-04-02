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
struct MaybeTypeStorage;
} // namespace detail

enum class SoraTypeKind {
  First_Type = mlir::Type::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,

  /// The Sora "Maybe" Type.
  Maybe,

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
                                              detail::MaybeTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) {
    return kind == (unsigned)SoraTypeKind::Maybe;
  }

  static MaybeType get(mlir::Type valueType);

  mlir::Type getValueType() const;
};
} // namespace ir
} // namespace sora