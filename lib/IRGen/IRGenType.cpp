//===--- IRGenType.cpp - Type IR Generation ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/TypeVisitor.hpp"
#include "Sora/AST/Types.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

using namespace sora;

//===- TypeConverter ------------------------------------------------------===//

namespace {
/// Converts Sora Types to MLIR Types.
class TypeConverter : public IRGeneratorBase,
                      public TypeVisitor<TypeConverter, mlir::Type> {
public:
  TypeConverter(IRGen &irGen)
      : IRGeneratorBase(irGen) {}

  mlir::Type visitIntegerType(IntegerType *type) {
    IntegerWidth integerWidth = type->getWidth();
    assert(!integerWidth.isArbitraryPrecision() &&
           "arbitrary-precision integer are not supported in IRGen");
    unsigned width = integerWidth.getWidth();
    auto signedness = (type->isSigned() ? mlir::IntegerType::Signed
                                        : mlir::IntegerType::Unsigned);
    return mlir::IntegerType::get(width, signedness, &mlirCtxt);
  }

  mlir::Type visitFloatType(FloatType *type) {
    switch (type->getFloatKind()) {
    case FloatKind::IEEE32:
      return mlir::FloatType::getF32(&mlirCtxt);
    case FloatKind::IEEE64:
      return mlir::FloatType::getF64(&mlirCtxt);
    default:
      llvm_unreachable("Unknown FloatKind");
    }
  }

  mlir::Type visitVoidType(VoidType *type) {
    return mlir::NoneType::get(&mlirCtxt);
  }

  mlir::Type visitBoolType(BoolType *type) {
    // FIXME: Needs a special bool type?
    return mlir::IntegerType::get(1, &mlirCtxt);
  }

  mlir::Type visitNullType(NullType *type) {
    llvm_unreachable("Cannot lower a Null Type");
  }

  mlir::Type visitReferenceType(ReferenceType *type) {
    llvm_unreachable("TODO - visitReferenceType");
  }

  mlir::Type visitMaybeType(MaybeType *type) {
    llvm_unreachable("TODO - visitMaybeType");
  }

  mlir::Type visitTupleType(TupleType *type) {
    SmallVector<mlir::Type, 4> elts;
    elts.reserve(type->getNumElements());
    for (Type elt : type->getElements())
      elts.push_back(visit(elt));
    return mlir::TupleType::get(elts, &mlirCtxt);
  }

  mlir::Type visitFunctionType(FunctionType *type) {
    mlir::Type rtr = visit(type->getReturnType());
    SmallVector<mlir::Type, 4> args;
    args.reserve(type->getNumArgs());
    for (Type arg : type->getArgs())
      args.push_back(visit(arg));
    return mlir::FunctionType::get(args, rtr, &mlirCtxt);
  }

  mlir::Type visitLValueType(LValueType *type) {
    // FIXME: Do LValues need a IR representation?
    return visit(type->getObjectType());
  }

  mlir::Type visitErrorType(ErrorType *) {
    llvm_unreachable("ErrorType found after Sema");
  }

  mlir::Type visitTypeVariableType(TypeVariableType *) {
    llvm_unreachable("TypeVariable found after Sema");
  }
};
} // namespace

//===- IRGen --------------------------------------------------------------===//

mlir::Type IRGen::getMLIRType(Type type) {
  assert(!type->hasNullType() && "Cannot lower Null Types");
  auto iter = typeCache.find(type.getPtr());
  if (iter != typeCache.end())
    return iter->second;
  mlir::Type mlirType = TypeConverter(*this).visit(type);
  typeCache.insert({type.getPtr(), mlirType});
  return mlirType;
}