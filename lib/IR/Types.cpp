//===--- Types.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/IR/Types.hpp"
#include "Sora/IR/Dialect.hpp"
#include "mlir/IR/DialectImplementation.h"

using namespace sora;
using namespace sora::ir;

//===- Type Printing ------------------------------------------------------===//

static void print(MaybeType type, mlir::DialectAsmPrinter &os) {
  os << "maybe<" << type.getValueType() << ">";
}

static void print(ReferenceType type, mlir::DialectAsmPrinter &os) {
  os << "reference<" << type.getPointeeType() << ">";
}

static void print(LValueType type, mlir::DialectAsmPrinter &os) {
  os << "lvalue<" << type.getObjectType() << ">";
}

void SoraDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &os) const {
  switch (SoraTypeKind(type.getKind())) {
#define HANDLE(T)                                                              \
  case SoraTypeKind::T:                                                        \
    return print(type.cast<T##Type>(), os)
    HANDLE(Maybe);
    HANDLE(Reference);
    HANDLE(LValue);
#undef HANDLE
  default:
    llvm_unreachable("Unknown Sora Type!");
  }
}

//===- Type Storage -------------------------------------------------------===//

namespace sora {
namespace ir {
namespace detail {
/// Common storage class for types that only contain another type.
struct SingleTypeStorage : public mlir::TypeStorage {
  SingleTypeStorage(mlir::Type type) : type(type) {}

  using KeyTy = mlir::Type;
  bool operator==(const KeyTy &key) const { return key == type; }

  static SingleTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      mlir::Type type) {
    return new (allocator.allocate<SingleTypeStorage>())
        SingleTypeStorage(type);
  }

  mlir::Type type;
};
} // namespace detail
} // namespace ir
} // namespace sora

//===- MaybeType ----------------------------------------------------------===//

MaybeType MaybeType::get(mlir::Type valueType) {
  assert(valueType && "value type cannot be null!");
  return Base::get(valueType.getContext(), (unsigned)SoraTypeKind::Maybe,
                   valueType);
}

mlir::Type MaybeType::getValueType() const { return getImpl()->type; }

//===- ReferenceType ------------------------------------------------------===//

ReferenceType ReferenceType::get(mlir::Type pointeeType) {
  assert(pointeeType && "pointee type cannot be null!");
  return Base::get(pointeeType.getContext(), (unsigned)SoraTypeKind::Reference,
                   pointeeType);
}

mlir::Type ReferenceType::getPointeeType() const { return getImpl()->type; }

//===- LValueType ---------------------------------------------------------===//

LValueType LValueType::get(mlir::Type objectType) {
  assert(objectType && "object type cannot be null!");
  return Base::get(objectType.getContext(), (unsigned)SoraTypeKind::LValue,
                   objectType);
}

mlir::Type LValueType::getObjectType() const { return getImpl()->type; }