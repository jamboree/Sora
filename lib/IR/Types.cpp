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

static void print(PointerType type, mlir::DialectAsmPrinter &os) {
  os << "pointer<" << type.getObjectType() << ">";
}

static void print(VoidType type, mlir::DialectAsmPrinter &os) { os << "void"; }

void SoraDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &os) const {
  switch (SoraTypeKind(type.getKind())) {
#define HANDLE(T)                                                              \
  case SoraTypeKind::T:                                                        \
    return print(type.cast<T##Type>(), os)
    HANDLE(Maybe);
    HANDLE(Reference);
    HANDLE(Pointer);
    HANDLE(Void);
#undef HANDLE
  default:
    llvm_unreachable("Unknown Sora Type!");
  }
}

//===- Type Parsing -------------------------------------------------------===//

mlir::Type SoraDialect::parseType(mlir::DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  // Parses '<' type '>'
  auto parseTypeArgument = [&] {
    mlir::Type result;
    if (parser.parseLess() || parser.parseType(result) || parser.parseGreater())
      return mlir::Type();
    return result;
  };

  // sora-type = 'maybe' '<' type '>'
  //           | 'reference' '<' type '>'
  //           | 'pointer' '<' type '>'
  //           | 'void'
  if (keyword == "maybe")
    return MaybeType::get(parseTypeArgument());
  if (keyword == "reference")
    return ReferenceType::get(parseTypeArgument());
  if (keyword == "pointer")
    return PointerType::get(parseTypeArgument());
  if (keyword == "void")
    return VoidType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown Sora type: ") << keyword;
  return Type();
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

//===- PointerType --------------------------------------------------------===//

PointerType PointerType::get(mlir::Type objectType) {
  assert(objectType && "object type cannot be null!");
  return Base::get(objectType.getContext(), (unsigned)SoraTypeKind::Pointer,
                   objectType);
}

mlir::Type PointerType::getObjectType() const { return getImpl()->type; }