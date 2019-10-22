//===--- Types.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Types.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "llvm/ADT/APFloat.h"

using namespace sora;

bool CanType::isValid() const {
  if (const TypeBase *ptr = getPtr())
    return ptr->isCanonical();
  return true;
}

SourceRange TypeLoc::getSourceRange() const {
  return tyRepr ? tyRepr->getSourceRange() : SourceRange();
}

SourceLoc TypeLoc::getBegLoc() const {
  return tyRepr ? tyRepr->getBegLoc() : SourceLoc();
}

SourceLoc TypeLoc::getLoc() const {
  return tyRepr ? tyRepr->getLoc() : SourceLoc();
}

SourceLoc TypeLoc::getEndLoc() const {
  return tyRepr ? tyRepr->getEndLoc() : SourceLoc();
}

void *TypeBase::operator new(size_t size, ASTContext &ctxt,
                             AllocatorKind allocator, unsigned align) {
  return ctxt.allocate(size, align, allocator);
}

FloatType *FloatType::get(ASTContext &ctxt, FloatKind kind) {
  switch (kind) {
  case FloatKind::IEEE32:
    return ctxt.f32Type->castTo<FloatType>();
  case FloatKind::IEEE64:
    return ctxt.f64Type->castTo<FloatType>();
  }
  llvm_unreachable("Unknown FloatKind!");
}

const llvm::fltSemantics &FloatType::getAPFloatSemantics() const {
  switch (getFloatKind()) {
  case FloatKind::IEEE32:
    return APFloat::IEEEsingle();
  case FloatKind::IEEE64:
    return APFloat::IEEEdouble();
  }
  llvm_unreachable("Unknown FloatKind!");
}

void TupleType::Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> elements) {
  id.AddInteger(elements.size());
  for (Type type : elements)
    id.AddPointer(type.getPtr());
}