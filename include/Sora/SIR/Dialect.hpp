//===--- Dialect.hpp - Sora IR MLIR Dialect Declaration ---------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
//#include "mlir/IR/Function.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace sora::sir {
using mlir::Block;
using mlir::FuncOp;
using mlir::Type;
using mlir::Value;
} // namespace sora::sir

namespace sora::sir {
namespace detail {
/// Common storage class for types that only contain another type.
struct SingleTypeStorage;
} // namespace detail

/// Common base for all Sora IR Types.
class SIRType : public mlir::Type {
public:
  using Type::Type;

  static bool classof(Type type);
};

/// Sora "maybe" types. Equivalent to a AST MaybeType.
///
/// This type is written "!sir.maybe<T>"
class MaybeType : public mlir::Type::TypeBase<MaybeType, SIRType,
                                              detail::SingleTypeStorage> {
public:
  using Base::Base;

  static MaybeType get(mlir::Type valueType);

  mlir::Type getValueType() const;
};

/// Sora reference types. Equivalent to a AST ReferenceType, but without the
/// "mut" qualifier - it is irrelevant in the IR.
///
/// This type is written "!sir.reference<T>"
class ReferenceType : public mlir::Type::TypeBase<ReferenceType, SIRType,
                                                  detail::SingleTypeStorage> {
public:
  using Base::Base;

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
/// This type is written "!sir.pointer<T>"
class PointerType : public mlir::Type::TypeBase<PointerType, SIRType,
                                                detail::SingleTypeStorage> {
public:
  using Base::Base;

  static PointerType get(mlir::Type objectType);

  mlir::Type getPointeeType() const;
};

/// Sora 'void' type. Equivalent to the AST VoidType;
///
/// This type is written "!sir.void"
class VoidType
    : public mlir::Type::TypeBase<VoidType, SIRType, mlir::TypeStorage> {
public:
  using Base::Base;

  // static VoidType get(mlir::MLIRContext *ctxt) {
  //  return Base::get(ctxt, (unsigned)SIRTypeKind::Void);
  //}
};

inline bool SIRType::classof(Type type) {
  return type.isa<MaybeType, ReferenceType, PointerType, VoidType>();
}
} // namespace sora::sir

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "Sora/SIR/Dialect.h.inc"

// Include the TableGen'd file containing the declarations of the Sora IR
// Operations.
#define GET_OP_CLASSES
#include "Sora/SIR/Ops.h.inc"