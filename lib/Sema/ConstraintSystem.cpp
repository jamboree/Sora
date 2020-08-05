//===--- ConstraintSystem.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "ConstraintSystem.hpp"
#include "Sora/AST/TypeVisitor.hpp"
#include "Sora/AST/Types.hpp"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

#define DEBUG_TYPE "Constraint System "

STATISTIC(numSuccessfulUnifications, "# of successful type unifications");
STATISTIC(numFailedUnifications, "# of failed type unifications");
STATISTIC(numCanUnifyCalls, "# of calls to canUnify");
STATISTIC(numTypeVariablesBound, "# of type variables bound");
STATISTIC(numTypeVariablesCreated, "# of type variables created");

//===--- TypeUnifier ------------------------------------------------------===//

/// Implements ConstraintSystem::unify()
///
/// visit() methods return true on success, false on failure.
namespace {
class TypeUnifier {

public:
  TypeUnifier(ConstraintSystem &cs, const UnificationOptions &options,
              bool canBindTypeVariables)
      : cs(cs), options(options), canBindTypeVariables(canBindTypeVariables) {}

  ConstraintSystem &cs;
  const UnificationOptions &options;
  const bool canBindTypeVariables;

  /// Attempts to bind \p tv's to \p proposedBinding.
  bool tryBindTypeVariable(TypeVariableType *tv, Type proposedBinding) {
    proposedBinding = proposedBinding->rebuildTypeWithoutLValues();
    if (!cs.canBind(tv, proposedBinding))
      return false;
    if (canBindTypeVariables)
      cs.bind(tv, proposedBinding);
    return true;
  }

  bool unify(Type type, Type other) {
    type = type->getRValueType();
    other = other->getRValueType();

    // If both types are equal, unification succeeds.
    if (type->getCanonicalType() == other->getCanonicalType())
      return true;

    auto setBindingOrUnify = [&](TypeVariableType *tv, Type proposedBinding) {
      if (cs.isBound(tv))
        return unify(cs.getBinding(tv), proposedBinding);
      return tryBindTypeVariable(tv, proposedBinding);
    };

    // If one is a type variable, and the other isn't, just bind, or unify the
    // current binding with 'other'.
    if (TypeVariableType *tv = type->getAs<TypeVariableType>()) {
      if (!other->is<TypeVariableType>())
        return setBindingOrUnify(tv, other);
    }
    else if (TypeVariableType *otherTV = other->getAs<TypeVariableType>())
      return setBindingOrUnify(otherTV, type);

    type = type->getDesugaredType();
    other = other->getDesugaredType();

    // Else, we must visit the types to check that their structure matches.
    if (type->getKind() != other->getKind())
      return false;

    switch (type->getKind()) {
#define TYPE(ID, PARENT)                                                       \
  case TypeKind::ID:                                                           \
    return visit##ID##Type(static_cast<ID##Type *>(type.getPtr()),             \
                           static_cast<ID##Type *>(other.getPtr()));
#include "Sora/AST/TypeNodes.def"
    }
    llvm_unreachable("Unknown node");
  }

  /// Builtin types must be strictly equal!
#define BUILTIN_TYPE(TYPE)                                                     \
  bool visit##TYPE(TYPE *type, TYPE *other) { return false; }

  BUILTIN_TYPE(IntegerType)
  BUILTIN_TYPE(FloatType)
  BUILTIN_TYPE(VoidType)
  BUILTIN_TYPE(BoolType)

#undef BUILTIN_TYPE

  bool visitReferenceType(ReferenceType *type, ReferenceType *other) {
    if (!options.ignoreReferenceMutability)
      if (type->isMut() != other->isMut())
        return false;
    return unify(type->getPointeeType(), other->getPointeeType());
  }

  bool visitMaybeType(MaybeType *type, MaybeType *other) {
    return unify(type->getValueType(), other->getValueType());
  }

  bool visitTupleType(TupleType *type, TupleType *other) {
    // Tuples must have the same number of elements
    if (type->getNumElements() != other->getNumElements())
      return false;

    bool success = true;
    // Unify the elements
    for (size_t k = 0, size = type->getNumElements(); k < size; ++k)
      success &= unify(type->getElement(k), other->getElement(k));
    return success;
  }

  bool visitFunctionType(FunctionType *type, FunctionType *other) {
    // Functions must have the same number of arguments
    if (type->getNumArgs() != other->getNumArgs())
      return false;

    bool success = true;
    // Unify return types
    success &= unify(type->getReturnType(), other->getReturnType());
    // Unify arg types
    for (size_t k = 0, size = type->getNumArgs(); k < size; ++k)
      success &= unify(type->getArg(k), other->getArg(k));
    return success;
  }

  bool visitLValueType(LValueType *type, LValueType *other) {
    return unify(type->getObjectType(), other->getObjectType());
  }

  bool visitErrorType(ErrorType *type, ErrorType *other) {
    // FIXME: Return true or false? The types are indeed equal, but they're
    // error types. Should <error_type> = <error_type> really return true?
    return true;
  }

  bool visitTypeVariableType(TypeVariableType *type, TypeVariableType *other) {
    // If they're both bound, unify their respective bindings.
    if (cs.isBound(type) && cs.isBound(other))
      return unify(cs.getBinding(type), cs.getBinding(other));

    // if only 'type' is bound, bind 'other' to 'type', unless 'type' is already
    // bound to 'other'.
    if (Type binding = cs.getBinding(type)) {
      if (binding->getCanonicalType()->getAs<TypeVariableType>() == other)
        return true;
      return tryBindTypeVariable(other, type);
    }

    // if only 'other' is bound, bind 'type' to 'other', unless 'other' is
    // already bound to 'type'.
    if (Type binding = cs.getBinding(other)) {
      if (binding->getAs<TypeVariableType>() == type)
        return true;
      return tryBindTypeVariable(type, other);
    }

    // Else, if they're both unbound, just bind 'type' to 'other', or 'other' to
    // 'type if the first one doesn't work.
    return tryBindTypeVariable(type, other) || tryBindTypeVariable(other, type);
  }
};
} // namespace

//===--- ConstraintSystem -------------------------------------------------===//

TypeVariableType *ConstraintSystem::createTypeVariable(TypeVariableKind kind) {
  auto *tyVar = new (*this) TypeVariableType(*this, kind, typeVariables.size());
  typeVariables.push_back(tyVar);
  return tyVar;
}

bool ConstraintSystem::unify(Type a, Type b,
                             const UnificationOptions &options) {
  assert(a && "a is null");
  assert(b && "b is null");
  TypeUnifier unifier(*this, options, /*canBindTypeVariables*/ true);
  bool result = unifier.unify(a, b);
  result ? ++numSuccessfulUnifications : ++numFailedUnifications;
  return result;
}

bool ConstraintSystem::canUnify(Type a, Type b,
                                const UnificationOptions &options) const {
  assert(a && "a is null");
  assert(b && "b is null");
  ++numCanUnifyCalls;
  return TypeUnifier(*const_cast<ConstraintSystem *>(this), options,
                     /*canBindTypeVariables*/ false)
      .unify(a, b);
}

void ConstraintSystem::dumpTypeVariables(raw_ostream &out,
                                         TypePrintOptions printOptions) const {
  if (typeVariables.empty())
    out << "    <no type variables>\n";
  else
    for (TypeVariableType *tyVar : typeVariables) {
      out << "    " << *tyVar;
      if (isBound(tyVar))
        if (Type simplified = simplify(tyVar))
          if (simplified->hasErrorType())
            out << " AKA '" << simplified << "'";
      out << "\n";
    }
}
