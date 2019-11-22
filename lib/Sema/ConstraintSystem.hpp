//===--- ConstraintSystem.hpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTContext.hpp"
#include "Sora/Common/LLVM.hpp"
#include "TypeChecker.hpp"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class TypeVariableType;

/// The ConstraintSystem serves a context for constraint generation, solving
/// and diagnosing.
///
/// It creates TypeVariables, keeps track of their substitutions, of the
/// current constraints, etc.
///
/// It also possesses a RAIIConstraintSystemArena object, so while this object
/// is alive, the ConstraintSystem arena is available.
class ConstraintSystem final {
  ConstraintSystem(TypeChecker &tc)
      : ctxt(tc.ctxt), raiiCSArena(ctxt.createConstraintSystemArena()) {}

  ASTContext &ctxt;

private:
  /// The Constraint System Arena RAII Object.
  RAIIConstraintSystemArena raiiCSArena;

  /// The ID of the next TypeVariable that'll be allocated by the constraint
  /// system.
  unsigned nextTypeVariableID = 0;

public:
  /// Creates a new general type variable, which can be unified with anything.
  TypeVariableType *createGeneralTypeVariable();
  /// Creates a new integer type variable, which can only be unified with
  /// integers.
  TypeVariableType *createIntegerTypeVariable();
  /// Creates a new float type variable, which can only be unified with floats.
  TypeVariableType *createFloatTypeVariable();
};
} // namespace sora