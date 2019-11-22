//===--- ConstraintSystem.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "ConstraintSystem.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

TypeVariableType *ConstraintSystem::createTypeVariable() {
  return TypeVariableType::create(ctxt, nextTypeVariableID++);
}