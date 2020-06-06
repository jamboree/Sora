//===--- SIRGenType.cpp - Type SIR Generation -------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/TypeVisitor.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/SIR/Types.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"

using namespace sora;

//===- TypeConverter ------------------------------------------------------===//

namespace {
/// Converts Sora AST Types to SIR Types.
class TypeConverter : public SIRGeneratorBase,
                      public TypeVisitor<TypeConverter, mlir::Type> {
  mlir::Type visit(Type type) { return Base::visit(type); }

public:
  using SIRGeneratorBase::SIRGeneratorBase;
  using Base = TypeVisitor<TypeConverter, mlir::Type>;

  mlir::Type visit(CanType type) { return Base::visit(type); }

  mlir::Type visitIntegerType(IntegerType *type) {
    IntegerWidth integerWidth = type->getWidth();
    assert(!integerWidth.isArbitraryPrecision() &&
           "arbitrary-precision integer are not supported in SIRGen");
    unsigned width = integerWidth.getWidth();
    return mlir::IntegerType::get(width, mlir::IntegerType::Signless,
                                  &mlirCtxt);
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
    return sir::VoidType::get(&mlirCtxt);
  }

  mlir::Type visitBoolType(BoolType *type) {
    return mlir::IntegerType::get(1, &mlirCtxt);
  }

  mlir::Type visitNullType(NullType *type) {
    llvm_unreachable("Cannot lower a Null Type");
  }

  mlir::Type visitReferenceType(ReferenceType *type) {
    mlir::Type pointeeType = visit(type->getPointeeType());
    return sir::ReferenceType::get(pointeeType);
  }

  mlir::Type visitMaybeType(MaybeType *type) {
    mlir::Type valueType = visit(type->getValueType());
    return sir::MaybeType::get(valueType);
  }

  mlir::Type visitTupleType(TupleType *type) {
    SmallVector<mlir::Type, 4> elts;
    elts.reserve(type->getNumElements());
    for (Type elt : type->getElements())
      elts.push_back(visit(elt));
    return mlir::TupleType::get(elts, &mlirCtxt);
  }

  mlir::Type visitFunctionType(FunctionType *type) {
    SmallVector<mlir::Type, 4> args;
    args.reserve(type->getNumArgs());
    for (Type arg : type->getArgs())
      args.push_back(visit(arg));

    // If the return type is void, or void-like, don't generate a return type
    // and leave the result set empty.
    Type returnType = type->getReturnType();
    if (sirGen.isVoidOrVoidLikeType(returnType->getCanonicalType()))
      return mlir::FunctionType::get(args, {}, &mlirCtxt);
    return mlir::FunctionType::get(args, visit(returnType), &mlirCtxt);
  }

  mlir::Type visitLValueType(LValueType *type) {
    mlir::Type objectType = visit(type->getObjectType());
    return sir::PointerType::get(objectType);
  }

  mlir::Type visitErrorType(ErrorType *) {
    llvm_unreachable("ErrorType found after Sema");
  }

  mlir::Type visitTypeVariableType(TypeVariableType *) {
    llvm_unreachable("TypeVariable found after Sema");
  }
};
} // namespace

//===- SIRGen -------------------------------------------------------------===//

bool SIRGen::isVoidOrVoidLikeType(CanType type) {
  // void is obviously void-like.
  if (type->isVoidType())
    return true;
  // tuples are void-like if their elements are.
  if (TupleType *tuple = type->getAs<TupleType>()) {
    for (Type tupleElt : tuple->getElements())
      if (!isVoidOrVoidLikeType(CanType(tupleElt)))
        return false;
    return true;
  }
  // everything else isn't void-like.
  return false;
}

mlir::Type SIRGen::getType(Type type) {
  assert(!type->hasNullType() && "Cannot lower Null Types");
  auto iter = typeCache.find(type.getPtr());
  if (iter != typeCache.end())
    return iter->second;
  mlir::Type mlirType = TypeConverter(*this).visit(type->getCanonicalType());
  typeCache.insert({type.getPtr(), mlirType});
  return mlirType;
}