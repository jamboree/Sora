//===--- Expr.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Expr.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

void *Expr::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align);
}
