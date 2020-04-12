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

//===--- TypeSimplifier ---------------------------------------------------===//

/// The TypeVariableType simplifier, which replaces type variables with their
/// bindings.
///
/// Note that visit methods return null when no change is needed, so  visit()
/// returns null for types with no type variables.
namespace {
class TypeSimplifier : public TypeVisitor<TypeSimplifier, Type> {
public:
  using Parent = TypeVisitor<TypeSimplifier, Type>;
  friend Parent;

  TypeSimplifier(const ConstraintSystem &cs, Type defaultInt, Type defaultFloat)
      : cs(cs), defaultInt(defaultInt), defaultFloat(defaultFloat) {
    assert(defaultInt && "No Default Integer Type!");
    assert(!defaultInt->hasErrorType() ||
           !defaultInt->hasTypeVariable() && "Invalid Default Integer Type!");
    assert(defaultFloat && "No Default Integer Type!");
    assert(!defaultFloat->hasErrorType() ||
           !defaultFloat->hasTypeVariable() && "Invalid Default Float Type!");
  }

  const ConstraintSystem &cs;
  Type defaultInt, defaultFloat;

  /// Whether we found an unbound general type variable.
  bool hadUnboundGeneralTypeVariable = false;

  Type visit(Type type) {
    if (type->hasTypeVariable())
      return Parent::visit(type);
    return type;
  }

private:
  Type visitBuiltinType(BuiltinType *type) { return nullptr; }

  Type visitReferenceType(ReferenceType *type) {
    if (Type simplified = visit(type->getPointeeType()))
      return ReferenceType::get(simplified, type->isMut());
    return nullptr;
  }

  Type visitMaybeType(MaybeType *type) {
    if (Type simplified = visit(type->getValueType()))
      return MaybeType::get(simplified);
    return nullptr;
  }

  Type visitTupleType(TupleType *type) {
    bool needsRebuilding = false;

    SmallVector<Type, 4> elems;
    elems.reserve(type->getNumElements());
    for (Type elem : type->getElements()) {
      if (Type simplified = visit(elem)) {
        needsRebuilding = true;
        elems.push_back(simplified);
      }
      else
        elems.push_back(elem);
    }

    if (!needsRebuilding)
      return nullptr;
    return TupleType::get(cs.ctxt, elems);
  }

  Type visitFunctionType(FunctionType *type) {
    bool needsRebuilding = false;

    Type rtr = type->getReturnType();
    if (Type simplified = visit(rtr)) {
      needsRebuilding = true;
      rtr = simplified;
    }

    SmallVector<Type, 4> args;
    args.reserve(type->getNumArgs());
    for (Type arg : type->getArgs()) {
      if (Type simplified = visit(arg)) {
        needsRebuilding = true;
        args.push_back(simplified);
      }
      else
        args.push_back(arg);
    }

    if (!needsRebuilding)
      return nullptr;
    return FunctionType::get(args, rtr);
  }

  Type visitLValueType(LValueType *type) {
    if (Type simplified = visit(type->getObjectType()))
      return LValueType::get(simplified);
    return nullptr;
  }

  Type visitErrorType(ErrorType *type) { return nullptr; }

  Type visitTypeVariableType(TypeVariableType *type) {
    Type binding = type->getBinding();
    if (binding)
      return visit(binding);
    if (type->isFloatTypeVariable())
      return defaultFloat;
    if (type->isIntegerTypeVariable())
      return defaultInt;
    assert(type->isGeneralTypeVariable() && "Unknown Type Variable Kind!");
    hadUnboundGeneralTypeVariable = true;
    return cs.ctxt.errorType;
  }
};
} // namespace

//===--- TypeUnifier ------------------------------------------------------===//

/// Implements ConstraintSystem::unify()
///
/// visit() methods return true on success, false on failure.
namespace {
class TypeUnifier {

public:
  TypeUnifier(const UnificationOptions &options, bool canBindTypeVariables)
      : options(options), canBindTypeVariables(canBindTypeVariables) {}

  const UnificationOptions &options;
  const bool canBindTypeVariables;

  /// Attempts to bind \p tv's to \p proposedBinding.
  bool tryBindTypeVariable(TypeVariableType *tv, Type proposedBinding) {
    proposedBinding = proposedBinding->getRValueType();
    if (!tv->canBindTo(proposedBinding))
      return false;
    if (canBindTypeVariables)
      tv->bindTo(proposedBinding);
    return true;
  }

  bool unify(Type type, Type other) {
    type = type->getRValueType();
    other = other->getRValueType();

    // If both types are equal, unification succeeds.
    if (type->getCanonicalType() == other->getCanonicalType())
      return true;

    auto setBindingOrUnify = [&](TypeVariableType *tv, Type proposedBinding) {
      if (tv->isBound())
        return unify(tv->getBinding(), proposedBinding);
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
  BUILTIN_TYPE(NullType)

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
    if (type->isBound() && other->isBound())
      return unify(type->getBinding(), other->getBinding());

    // if only 'type' is bound, bind 'other' to 'type', unless 'type' is already
    // bound to 'other'.
    if (type->isBound()) {
      if (type->getBinding()->getCanonicalType()->getAs<TypeVariableType>() ==
          other)
        return true;
      return tryBindTypeVariable(other, type);
    }

    // if only 'other' is bound, bind 'type' to 'other', unless 'other' is
    // already bound to 'type'.
    if (other->isBound()) {
      if (other->getBinding()->getAs<TypeVariableType>() == type)
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
  auto *tyVar = new (ctxt) TypeVariableType(ctxt, kind, typeVariables.size());
  typeVariables.push_back(tyVar);
  return tyVar;
}

Type ConstraintSystem::simplifyType(Type type,
                                    bool *hadUnboundGeneralTypeVariable) const {
  if (!type->hasTypeVariable())
    return type;
  auto typeSimplifier = TypeSimplifier(*this, intTVDefault, floatTVDefault);
  Type simplified = typeSimplifier.visit(type);
  assert(simplified && !simplified->hasTypeVariable() && "Not simplified!");
  if (hadUnboundGeneralTypeVariable)
    *hadUnboundGeneralTypeVariable =
        typeSimplifier.hadUnboundGeneralTypeVariable;
  return simplified;
}

bool ConstraintSystem::unify(Type a, Type b,
                             const UnificationOptions &options) {
  assert(a && "a is null");
  assert(b && "b is null");
  TypeUnifier unifier(options, /*canBindTypeVariables*/ true);
  bool result = unifier.unify(a, b);
  result ? ++numSuccessfulUnifications : ++numFailedUnifications;
  return result;
}

bool ConstraintSystem::canUnify(Type a, Type b,
                                const UnificationOptions &options) const {
  assert(a && "a is null");
  assert(b && "b is null");
  ++numCanUnifyCalls;
  return TypeUnifier(options, /*canBindTypeVariables*/ false).unify(a, b);
}

void ConstraintSystem::dumpTypeVariables(
    raw_ostream &out, const TypePrintOptions &printOptions) const {
  if (typeVariables.empty())
    out << "    <no type variables>\n";
  else
    for (TypeVariableType *tyVar : typeVariables) {
      out << "    ";
      tyVar->print(out, TypePrintOptions::forDebug());
      if (tyVar->isBound())
        if (Type simplified = simplifyType(tyVar->getBinding()))
          if (simplified->hasErrorType())
            out << " AKA '" << simplified << "'";
      out << "\n";
    }
}
