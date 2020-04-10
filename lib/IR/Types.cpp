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

void SoraDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &os) const {
  switch (SoraTypeKind(type.getKind())) {
  case SoraTypeKind::Maybe:
    return print(type.cast<MaybeType>(), os);
  case SoraTypeKind::Reference:
    return print(type.cast<ReferenceType>(), os);
  default:
    llvm_unreachable("Unknown Sora Type!");
  }
}

//===- Type Storage -------------------------------------------------------===//

namespace sora {
namespace ir {
namespace detail {
struct MaybeTypeStorage : public mlir::TypeStorage {
  MaybeTypeStorage(mlir::Type valueType) : valueType(valueType) {}

  // MaybeTypes are pointer-unique.
  using KeyTy = mlir::Type;
  bool operator==(const KeyTy &key) const { return key == valueType; }

  static MaybeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     mlir::Type valueType) {
    return new (allocator.allocate<MaybeTypeStorage>())
        MaybeTypeStorage(valueType);
  }

  mlir::Type valueType;
};

struct ReferenceTypeStorage : public mlir::TypeStorage {
  ReferenceTypeStorage(mlir::Type pointeeType) : pointeeType(pointeeType) {}

  // MaybeTypes are pointer-unique.
  using KeyTy = mlir::Type;
  bool operator==(const KeyTy &key) const { return key == pointeeType; }

  static ReferenceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         mlir::Type pointeeType) {
    return new (allocator.allocate<ReferenceTypeStorage>())
        ReferenceTypeStorage(pointeeType);
  }

  mlir::Type pointeeType;
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

mlir::Type MaybeType::getValueType() const { return getImpl()->valueType; }

//===- ReferenceType ------------------------------------------------------===//

ReferenceType ReferenceType::get(mlir::Type pointeeType) {
  assert(pointeeType && "pointee type cannot be null!");
  return Base::get(pointeeType.getContext(), (unsigned)SoraTypeKind::Reference,
                   pointeeType);
}

mlir::Type ReferenceType::getPointeeType() const {
  return getImpl()->pointeeType;
}