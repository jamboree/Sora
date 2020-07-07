//===--- TypeVariableEnvironment.hpp ----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2020 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Type.hpp"

namespace sora {
class ASTContext;
class IntegerType;
class FloatType;
enum class TypeVariableKind : uint8_t;
class TypeVariableType;

/// This class provides an environment for TypeVariables and can:
///   - Set a TypeVariable's binding
///   - Simplify types to remove all of the type variables inside it
///   - Allocate TypeVariables and types containing them.
class TypeVariableEnvironment {
  Type intTypeVarDefault = nullptr;
  Type floatTypeVarDefault = nullptr;

  friend ASTContext;

  void adjustTypeVariableKind(TypeVariableType *typeVar);

protected:
  ASTContext &ctxt;

public:
  TypeVariableEnvironment(ASTContext &ctxt);
  ~TypeVariableEnvironment();

  /// Allocates memory using the ASTContext's TypeVariableEnvironment allocator.
  /// \returns a pointer to the allocated memory (aligned to \p align) or
  /// nullptr if \p size == 0
  void *allocate(size_t size, size_t align);

  /// Allocates enough memory for an object \p Ty using the ASTContext's
  /// TypeVariableEnvironment allocator. This simply calls allocate using
  /// sizeof/alignof Ty. This does not construct the object. You'll need to use
  /// placement new for that.
  template <typename Ty> void *allocate() {
    return allocate(sizeof(Ty), alignof(Ty));
  }

  ASTContext &getASTContext() const { return ctxt; }

  void setIntegerTypeVariableDefaultType(Type type) {
    intTypeVarDefault = type;
  }

  Type getIntegerTypeVariableDefaultType() const { return intTypeVarDefault; }

  void setFloatTypeVariableDefaultType(Type type) {
    floatTypeVarDefault = type;
  }

  Type getFloatTypeVariableDefaultType() const { return floatTypeVarDefault; }

  /// \returns the default type for a TypeVariable of kind \p kind, or null if
  /// there is none.
  Type getDefaultType(TypeVariableKind kind) const;

  /// \returns whether \p typeVar can bind to \p type.
  bool canBind(TypeVariableType *typeVar, Type binding) const;

  /// Binds \p typeVar to \p binding.
  /// Bindings are definitive and cannot be changed later.
  /// This asserts that the binding is possible (see \p canBind).
  ///
  /// \param typeVar the unbound type variable.
  /// \param binding the desired binding.
  /// \param canAdjustKind whether the kind of \p typeVar can change depending
  /// on the binding. For instance, if this is true and \p typeVar is a General
  /// TV while binding is an Int/Int TV, the kind of \p typeVar will be changed
  /// to Integer.
  void bind(TypeVariableType *typeVar, Type binding, bool canAdjustKind = true);

  /// \returns whether \p typeVar is bound.
  bool isBound(TypeVariableType *typeVar) const;

  /// Simplifies \p type, replacing all type variables with their
  /// binding/default type, or an error type if no default type exists for this
  /// TV.
  /// \param hadUnboundTypeVar is set to true if at least one TypeVariable had
  /// no binding _and_ no default type (and thus was replaced with an
  /// ErrorType).
  Type simplify(Type type, bool *hadUnboundTypeVar = nullptr) const;

  /// \returns \p typeVar's binding, or null if it's unbound.
  Type getBinding(TypeVariableType *typeVar) const;

  /// \returns \p typeVar's binding if it's bound, else returns the default type
  /// for Int/Float type variables or null for General type variables.
  Type getBindingOrDefault(TypeVariableType *typeVar) const;
};
} // namespace sora