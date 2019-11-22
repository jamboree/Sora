//===--- ConstraintSystem.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "ConstraintSystem.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

TypeVariableType *ConstraintSystem::createGeneralTypeVariable() {
  return TypeVariableType::createGeneral(ctxt, nextTypeVariableID++);
}

TypeVariableType *ConstraintSystem::createIntegerTypeVariable() {
  return TypeVariableType::createInteger(ctxt, nextTypeVariableID++);
}

TypeVariableType *ConstraintSystem::createFloatTypeVariable() {
  return TypeVariableType::createFloat(ctxt, nextTypeVariableID++);
}
