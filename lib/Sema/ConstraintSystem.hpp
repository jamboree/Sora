//===--- ConstraintSystem.hpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/Common/LLVM.hpp"
#include "TypeChecker.hpp"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace sora {
class ConstraintSystem;
class TypeVariableType;

/// Kinds of Type Variables
enum class TypeVariableKind : uint8_t {
  /// Can unify w/ any type
  General,
  /// Can unify w/ integer types
  Integer,
  /// Can unify w/ float types
  Float
};

/// Options for type unification
struct UnificationOptions {
  /// Whether LValues are ignored.
  /// e.g. if true, unification will succeed for "@lvalue i32" and"i32", if set
  /// to false, it'll fail.
  bool ignoreLValues = true;
};

/// The ConstraintSystem serves a context for constraint generation, solving
/// and diagnosing.
///
/// It creates TypeVariables, keeps track of their substitutions, of the
/// current constraints, etc.
///
/// It also possesses a RAIIConstraintSystemArena object, so while this object
/// is alive, the ConstraintSystem arena is available.
///
/// Note that only one of those can be active at once, and that this is a
/// relatively large object that can't be copied.
class ConstraintSystem final {
public:
  /// \param tc the TypeChecker instance
  /// Uses i32 and f32 for the default int/float type variable substitutions.
  ConstraintSystem(TypeChecker &tc)
      : ConstraintSystem(tc, tc.ctxt.i32Type, tc.ctxt.f32Type) {}

  /// \param tc the TypeChecker instance
  /// \param intTVDefault The default type for Integer type variables
  /// \param floatTVDefault The default type for Float type variables
  ConstraintSystem(TypeChecker &tc, Type intTVDefault, Type floatTVDefault)
      : ctxt(tc.ctxt), typeChecker(tc),
        raiiCSArena(ctxt.createConstraintSystemArena()),
        intTVDefault(intTVDefault), floatTVDefault(floatTVDefault) {
    assert(intTVDefault &&
           "No default substitution for integer type variables");
    assert(floatTVDefault &&
           "No default substitution for float type variables");
  }

  ConstraintSystem(const ConstraintSystem &) = delete;
  ConstraintSystem &operator=(const ConstraintSystem &) = delete;

  ASTContext &ctxt;
  TypeChecker &typeChecker;

private:
  /// The Constraint System Arena RAII Object.
  RAIIConstraintSystemArena raiiCSArena;

  /// The list of type variables created by this ConstraintSystem. Mostly used
  /// by dumpTypeVariables().
  SmallVector<TypeVariableType *, 8> typeVariables;

  /// The default type of int type variables
  const Type intTVDefault;
  /// The default type of float type variables
  const Type floatTVDefault;

  /// Creates a new type variable of kind \p kind
  TypeVariableType *createTypeVariable(TypeVariableKind kind);

public:
  /// Creates a new General type variable.
  TypeVariableType *createGeneralTypeVariable() {
    return createTypeVariable(TypeVariableKind::General);
  }
  /// Create a new Integer type variable
  TypeVariableType *createIntegerTypeVariable() {
    return createTypeVariable(TypeVariableKind::Integer);
  }
  /// Create a new Float type variable
  TypeVariableType *createFloatTypeVariable() {
    return createTypeVariable(TypeVariableKind::Float);
  }

  /// \returns the default substitution for integer type variables
  Type getIntegerTypeVariableDefaultType() const { return intTVDefault; }
  /// \returns the default substitution for float type variables
  Type getFloatTypeVariableDefaultType() const { return floatTVDefault; }

  /// \returns the kind of TypeVariable that \p tv is
  TypeVariableKind getTypeVariableKind(TypeVariableType *tv) const;

  /// Sets \p tv's substitution.
  /// \returns false if the substitution was rejected (because there's already a
  /// substitution, or because it's not compatible), true if the substitution
  /// was accepted.
  bool setSubstitution(TypeVariableType *tv, Type subst) const;

  /// \returns whether \p tv has a substitution
  bool hasSubstitution(TypeVariableType *tv) const;

  /// \returns \p tv's substitution
  Type getSubstitution(TypeVariableType *tv) const;

  /// \returns true if \p tv is a general type variable
  bool isGeneralTypeVariable(TypeVariableType *tv) const {
    return getTypeVariableKind(tv) == TypeVariableKind::General;
  }

  /// \returns true if \p tv is a integer type variable
  bool isIntegerTypeVariable(TypeVariableType *tv) const {
    return getTypeVariableKind(tv) == TypeVariableKind::Integer;
  }

  /// \returns true if \p tv is a float type variable
  bool isFloatTypeVariable(TypeVariableType *tv) const {
    return getTypeVariableKind(tv) == TypeVariableKind::Float;
  }

  /// Simplifies \p type, replacing type variables with their substitutions.
  /// If a general type variable has no substitution, an ErrorType is used, if
  /// an Integer or Float type variable has no substitution, the default type
  /// for those type variables is used (see \c getIntegerTypeVariableDefaultType
  /// and \c getFloatTypeVariableDefaultType)
  ///
  /// \param type the type to simplify
  /// \param hadUnboundTypeVariable whether the type contained an unbound
  /// general type variable.
  /// \returns the simplified type, or the ErrorType on error (never nullptr)
  Type simplifyType(Type type, bool *hadUnboundTypeVariable = nullptr);

  /// Unifies \p a with \p b using \p options.
  /// \returns true if unification was successful, false otherwise.
  bool unify(Type a, Type b,
             const UnificationOptions &options = UnificationOptions());

  /// Checks if \p a can unify with \p b using \p options.
  /// \returns true if unification is possible, false otherwise.
  bool canUnify(Type a, Type b,
                const UnificationOptions &options = UnificationOptions()) const;

  /// Prints \p type and its TypeVariableInfo to \p out
  void print(raw_ostream &out, const TypeVariableType *type,
             const TypePrintOptions &printOptions = TypePrintOptions()) const;

  void dumpTypeVariables(
      raw_ostream &out,
      const TypePrintOptions &printOptions = TypePrintOptions()) const;

  /// Same as \c print, but returns a string instead of printing to a stream.
  std::string
  getString(const TypeVariableType *type,
            const TypePrintOptions &printOptions = TypePrintOptions()) const;
};
} // namespace sora