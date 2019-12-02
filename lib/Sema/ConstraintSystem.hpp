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

/// TypeVariable extra information
class alignas(alignof(TypeVariableType)) TypeVariableInfo final {
  friend class ConstraintSystem;

  // Disable vanilla new/delete for TypeVariableInfo
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  TypeVariableInfo(TypeVariableKind tvKind) : tvKind(tvKind) {}

  /// The kind of TypeVariable this is
  TypeVariableKind tvKind;
  /// This TypeVariable's substitution.
  /// Note that this can be another TypeVariable in case of an equivalence.
  Type substitution;

public:
  /// Make TypeVariableInfo noncopyable, so we don't copy it by mistake.
  TypeVariableInfo(const TypeVariableInfo &) = delete;
  TypeVariableInfo &operator=(const TypeVariableInfo &) = delete;

  /// \returns the TypeVariableInfo object for \p type
  static TypeVariableInfo &get(const TypeVariableType *type) {
    return *reinterpret_cast<TypeVariableInfo *>(
        const_cast<TypeVariableType *>(type) + 1);
  }

  TypeVariableKind getTypeVariableKind() const { return tvKind; }

  bool isGeneralTypeVariable() const {
    return getTypeVariableKind() == TypeVariableKind::General;
  }

  bool isIntegerTypeVariable() const {
    return getTypeVariableKind() == TypeVariableKind::Integer;
  }

  bool isFloatTypeVariable() const {
    return getTypeVariableKind() == TypeVariableKind::Float;
  }

  /// Sets this TypeVariable's substitution. This can only be done once. (\c
  /// hasSubstitution() must return false)
  void setSubstitution(Type type) {
    assert(!hasSubstitution() && "Already has a substitution");
    substitution = type;
  }
  /// \returns true if this TypeVariable has a substitution
  bool hasSubstitution() const { return (bool)substitution; }
  /// \returns this TypeVariable's substitution (it can be a TypeVariable in
  /// case of an equivalence class, see \c getBoundSubstitution() to get a
  /// non TypeVariable substitution)
  Type getSubstitution() const { return substitution; }
};

/// Options for type unification
struct UnificationOptions {
  std::function<bool(CanType, CanType)> typeComparator =
      [](CanType a, CanType b) { return a == b; };

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
  ConstraintSystem(TypeChecker &tc)
      : ctxt(tc.ctxt), typeChecker(tc),
        raiiCSArena(ctxt.createConstraintSystemArena()) {}

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

  /// Simplifies \p type, replacing type variables with their substitution (or
  /// ErrorType if there's no substitution)
  /// \param type the type to simplify
  /// \param defaultInt If non-null, overrides the default substitution of
  /// Integer type variables. (i32 by default)
  /// \param defaultFloat If non-null, overrides the default substitution of
  /// Float type variables. (f32 by default)
  /// \\\returns the simplified type, ErrorType on error (never nullptr)
  Type simplifyType(Type type, Type defaultInt = Type(),
                    Type defaultFloat = Type());

  /// Unifies \p a with \p b with \p options.
  /// \returns true if unification was successful, false otherwise.
  bool unify(Type a, Type b,
             const UnificationOptions &options = UnificationOptions());

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