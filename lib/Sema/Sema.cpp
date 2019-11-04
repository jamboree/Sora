//===--- Sema.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

Sema::Sema(ASTContext &ctxt) : ctxt(ctxt), diagEngine(ctxt.diagEngine) {}
