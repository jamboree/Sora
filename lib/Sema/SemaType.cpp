//===--- SemaType.cpp -------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Type Semantic Analysis
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

void Sema::resolveTypeLoc(TypeLoc &tyLoc) {
  assert(tyLoc.hasLocation() && "Must have a TypeRepr");
  assert(!tyLoc.hasType() && "TypeLoc already resolved!");
  // TODO
  tyLoc.setType(ctxt.errorType);
}