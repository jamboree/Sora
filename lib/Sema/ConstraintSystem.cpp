//===--- ConstraintSystem.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "ConstraintSystem.hpp"
#include "Sora/AST/TypeVisitor.hpp"
#include "Sora/AST/Types.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

//===--- TypeSimplifier ---------------------------------------------------===//

namespace {
/// The TypeVariableType simplifier, which replaces type variables with their
/// substitutions.
///
/// Note that visit methods return null when no substitutions are needed, so
/// visit() returns null for type with no type variables.
class TypeSimplifier : public TypeVisitor<TypeSimplifier, Type> {
public:
  using Parent = TypeVisitor<TypeSimplifier, Type>;
  friend Parent;

  TypeSimplifier(ConstraintSystem &cs, Type defaultInt, Type defaultFloat)
      : cs(cs), defaultInt(defaultInt), defaultFloat(defaultFloat) {
    // Default types can't have type variables
    assert(defaultInt && !defaultInt->hasTypeVariable());
    assert(defaultFloat && !defaultFloat->hasTypeVariable());
  }

  ConstraintSystem &cs;
  Type defaultInt, defaultFloat;
  bool hadUnboundTypeVariable = false;

  using Parent::visit;

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
    TypeVariableInfo &info = TypeVariableInfo::get(type);
    Type subst = info.getSubstitution();
    // Return a default type or ErrorType when there are no substitutions
    if (!subst) {
      if (info.isFloatTypeVariable())
        return defaultFloat;
      if (info.isIntegerTypeVariable())
        return defaultInt;
      hadUnboundTypeVariable = true;
      return cs.ctxt.errorType;
    }
    // If the substitution is also a type variable, simplify it as well.
    if (Type simplified = visit(subst))
      return simplified;
    return subst;
  }
};
//===--- TypeUnifier ------------------------------------------------------===//

/// Implements ConstraintSystem::unify()
///
/// visit() methods return true on success, false on failure.
class TypeUnifier {

  /// Checks for equality between a and b using options.typeComparator
  bool checkEquality(CanType a, CanType b) {
    return options.typeComparator(a, b);
  }

public:
  TypeUnifier(ConstraintSystem &cs, const UnificationOptions &options)
      : cs(cs), options(options) {}

  ConstraintSystem &cs;
  const UnificationOptions &options;

  bool unify(Type type, Type other) {
    // Ignore LValues if allowed to
    if (options.ignoreLValues) {
      type = type->getRValue();
      other = other->getRValue();
    }

    // If the types don't contain type variables, just compare their canonical
    // types.
    if (!type->hasTypeVariable() && !other->hasTypeVariable())
      return options.typeComparator(type->getCanonicalType(),
                                    other->getCanonicalType());

    auto setOrUnifySubstitution = [&](TypeVariableType *tv, Type subst) {
      TypeVariableInfo &info = TypeVariableInfo::get(tv);
      if (info.hasSubstitution())
        return unify(info.getSubstitution(), subst);
      return info.setSubstitution(subst);
    };

    // If one is a type variable, and the other isn't, just set the
    // substitution, or unify the current substitution with other.
    if (TypeVariableType *tv = type->getAs<TypeVariableType>()) {
      if (!other->is<TypeVariableType>()) {
        return setOrUnifySubstitution(tv, other);
      }
    }
    else if (TypeVariableType *otherTV = other->getAs<TypeVariableType>())
      return setOrUnifySubstitution(otherTV, type);

    // FIXME: After this step, desugared types should be used.

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

#define BUILTIN_TYPE(TYPE)                                                     \
  bool visit##TYPE(TYPE *type, TYPE *other) {                                  \
    return checkEquality(CanType(type), CanType(other));                       \
  }

  BUILTIN_TYPE(IntegerType)
  BUILTIN_TYPE(FloatType)
  BUILTIN_TYPE(VoidType)
  BUILTIN_TYPE(BoolType)
  BUILTIN_TYPE(NullType)

#undef BUILTIN_TYPE

  bool visitReferenceType(ReferenceType *type, ReferenceType *other) {
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
    TypeVariableInfo &typeInfo = TypeVariableInfo::get(type);
    TypeVariableInfo &otherInfo = TypeVariableInfo::get(other);

    // if both have a substitution, unify their substitution
    if (typeInfo.hasSubstitution() && otherInfo.hasSubstitution())
      return unify(typeInfo.getSubstitution(), otherInfo.getSubstitution());

    // if only 'type' has a substitution, set the substitution of 'other' to
    // 'type', unless type's substitution *is* other (equivalence class).
    if (typeInfo.hasSubstitution()) {
      if (typeInfo.getSubstitution()->getAs<TypeVariableType>() == other)
        return true;
      return otherInfo.setSubstitution(type);
    }

    // if only 'other' has a substitution, set the substitution of 'type' to
    // 'other', unless other's substitution *is* type (equivalence class).
    if (otherInfo.hasSubstitution()) {
      if (otherInfo.getSubstitution()->getAs<TypeVariableType>() == type)
        return true;
      return typeInfo.setSubstitution(type);
    }

    // Else, if both don't have a substitution, just set the substitution of
    // 'type' to 'other', or 'other' to 'type if the first one doesn't work out.
    return typeInfo.setSubstitution(other) || otherInfo.setSubstitution(type);
  }
};

} // namespace

//===--- ConstraintSystem -------------------------------------------------===//

TypeVariableType *ConstraintSystem::createTypeVariable(TypeVariableKind kind) {
  // Calculate the total size we need
  size_t size = sizeof(TypeVariableType) + sizeof(TypeVariableInfo);
  void *mem = ctxt.allocate(size, alignof(TypeVariableType),
                            ArenaKind::ConstraintSystem);
  // Create the TypeVariableType
  TypeVariableType *tyVar =
      new (mem) TypeVariableType(ctxt, typeVariables.size());
  // Create the info
  new (reinterpret_cast<void *>(tyVar + 1)) TypeVariableInfo(kind);
  // Store the TypeVariable
  typeVariables.push_back(tyVar);
  return tyVar;
}

Type ConstraintSystem::simplifyType(Type type, bool *hadUnboundTypeVariable) {
  if (!type->hasTypeVariable())
    return type;
  auto typeSimplifier = TypeSimplifier(*this, intTVDefault, floatTVDefault);
  Type simplified = typeSimplifier.visit(type);
  assert(simplified && !simplified->hasTypeVariable() && "Not simplified!");
  if (hadUnboundTypeVariable)
    *hadUnboundTypeVariable = typeSimplifier.hadUnboundTypeVariable;
  return simplified;
}

bool ConstraintSystem::unify(Type a, Type b,
                             const UnificationOptions &options) {
  return TypeUnifier(*this, options).unify(a, b);
}

void ConstraintSystem::print(raw_ostream &out, const TypeVariableType *type,
                             const TypePrintOptions &printOptions) const {
  // Print the TypeVariable itself
  type->print(out, printOptions);
  // Print the kind
  auto &info = TypeVariableInfo::get(type);
  switch (info.getTypeVariableKind()) {
  case TypeVariableKind::General:
    out << " [general type variable]";
    break;
  case TypeVariableKind::Integer:
    out << " [integer type variable]";
    break;
  case TypeVariableKind::Float:
    out << " [float type variable]";
    break;
  }
  // Print the substitution
  if (Type type = info.getSubstitution()) {
    out << " bound to '";
    type.print(out, printOptions);
    out << "'";
  }
  else
    out << " unbound";
}

void ConstraintSystem::dumpTypeVariables(
    raw_ostream &out, const TypePrintOptions &printOptions) const {
  out << "Type Variables Created By ConstraintSystem 0x" << (void *)this
      << "\n";
  for (TypeVariableType *tv : typeVariables) {
    out << "    ";
    print(out, tv, printOptions);
    out << "\n";
  }
  if (typeVariables.empty())
    out << "    <empty>\n";
}

std::string
ConstraintSystem::getString(const TypeVariableType *type,
                            const TypePrintOptions &printOptions) const {
  std::string str;
  llvm::raw_string_ostream out(str);
  print(out, type, printOptions);
  return out.str();
}