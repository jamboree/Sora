//===--- ConstraintSystem.hpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/TypeVariableEnvironment.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/Common/LLVM.hpp"
#include "TypeChecker.hpp"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace sora {
class ConstraintSystem;
class TypeVariableType;

/// Options for type unification
struct UnificationOptions {
  /// Whether mutability is ignored when checking equality of reference types.
  bool ignoreReferenceMutability = false;
};

/// The Constraint System serves as a context for type-checking of expressions.
///
/// It also possesses a RAIIConstraintSystemArena object, so while this object
/// is alive, the ConstraintSystem arena is available and type variables can be
/// allocated.
///
/// Only one ConstraintSystem can be alive at once.
///
/// Type variables should be created through the constraint system, it'll keep
/// track of them and take care of giving them a unique id.
class ConstraintSystem final : public TypeVariableEnvironment {
public:
  /// Default constructor that uses i32 and f32 for the default int/float type
  /// variable bindings.
  /// \param tc the TypeChecker instance
  ConstraintSystem(TypeChecker &tc)
      : ConstraintSystem(tc, tc.ctxt.i32Type, tc.ctxt.f32Type) {}

  /// \param tc the TypeChecker instance
  /// \param intTVDefault The default binding for Integer type variables.
  /// \param floatTVDefault The default binding for Float type variables.
  ConstraintSystem(TypeChecker &tc, Type intTVDefault, Type floatTVDefault)
      : TypeVariableEnvironment(tc.ctxt), typeChecker(tc) {

    setIntegerTypeVariableDefaultType(intTVDefault);
    setFloatTypeVariableDefaultType(floatTVDefault);

    assert(getIntegerTypeVariableDefaultType() &&
           "No default binding for integer type variables");
    assert(getFloatTypeVariableDefaultType() &&
           "No default binding for float type variables");
  }

  ConstraintSystem(const ConstraintSystem &) = delete;
  ConstraintSystem &operator=(const ConstraintSystem &) = delete;

  TypeChecker &typeChecker;

private:
  /// The list of type variables created by this ConstraintSystem. Mostly used
  /// by dumpTypeVariables().
  SmallVector<TypeVariableType *, 8> typeVariables;

  /// Creates a new type variable of kind \p kind
  TypeVariableType *createTypeVariable(TypeVariableKind kind);

public:
  /// Creates a new General type variable inside this ConstraintSystem.
  TypeVariableType *createGeneralTypeVariable() {
    return createTypeVariable(TypeVariableKind::General);
  }

  /// Create a new Integer type variable inside this ConstraintSystem.
  TypeVariableType *createIntegerTypeVariable() {
    return createTypeVariable(TypeVariableKind::Integer);
  }

  /// Create a new Float type variable inside this ConstraintSystem.
  TypeVariableType *createFloatTypeVariable() {
    return createTypeVariable(TypeVariableKind::Float);
  }

  /// \returns true if \p tv is a integer type variable or any int type.
  /// This only looks through LValues, nothing else.
  bool isIntegerTypeOrTypeVariable(Type type) const {
    assert(type);
    if (auto *tv = type->getRValueType()->getAs<TypeVariableType>())
      if (tv->isIntegerTypeVariable())
        return true;
    return type->isAnyIntegerType();
  }

  /// \returns true if \p tv is a float type variable or any float type.
  /// This only looks through LValues, nothing else.
  bool isFloatTypeOrTypeVariable(Type type) const {
    assert(type);
    if (auto *tv = type->getRValueType()->getAs<TypeVariableType>())
      if (tv->isFloatTypeVariable())
        return true;
    return type->isAnyFloatType();
  }

  /// Unifies \p a with \p b using \p options.
  /// \returns true if unification was successful, false otherwise.
  bool unify(Type a, Type b,
             const UnificationOptions &options = UnificationOptions());

  /// Checks if \p a can unify with \p b using \p options.
  /// \returns true if unification is possible, false otherwise.
  bool canUnify(Type a, Type b,
                const UnificationOptions &options = UnificationOptions()) const;

  /// Binds every type variable in \p type to the error type.
  void bindAllToErrorType(Type type) {
    if (!type->hasTypeVariable())
      return;
    // FIXME: it'd be great if there was a "walk" method.
    type->rebuildType([&](Type type) -> Type {
      if (TypeVariableType *tyVar = type->getAs<TypeVariableType>())
        if (!isBound(tyVar))
          bind(tyVar, ctxt.errorType);
      return {};
    });
  }

  void dumpTypeVariables(
      raw_ostream &out,
      TypePrintOptions printOptions = TypePrintOptions::forDebug()) const;
};
} // namespace sora