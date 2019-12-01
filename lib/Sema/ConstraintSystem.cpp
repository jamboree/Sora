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

  /// Attempts to set the substitution of \p tv to \p subst.
  /// \returns true on success, false on failure.
  bool setSubstitution(TypeVariableType *tv, Type subst) {
    TypeVariableInfo &info = TypeVariableInfo::get(tv);
    if (info.hasSubstitution())
      return false;
    // Check that the substitution is legal
    switch (info.getTypeVariableKind()) {
    case TypeVariableKind::General:
      break;
    case TypeVariableKind::Integer:
      if (!subst->is<IntegerType>())
        return false;
      break;
    case TypeVariableKind::Float:
      if (!subst->is<FloatType>())
        return false;
      break;
    }
    info.setSubstitution(subst);
    return true;
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

    // If the other is a type variable, and 'type' isn't, just set the
    // substitution
    if (!type->is<TypeVariableType>())
      if (TypeVariableType *otherTV = other->getAs<TypeVariableType>())
        return setSubstitution(otherTV, type);

    // Else, we must visit the type to check that their structure matches.
    if (type->getKind() != other->getKind())
      return false;

    switch (type->getKind()) {
#define TYPE(ID, PARENT)                                                       \
  case TypeKind::ID:                                                           \
    return visit##ID##Type(static_cast<ID##Type *>(type.getPtr()),             \
                           static_cast<ID##Type *>(other.getPtr()));
#include "Sora/AST/TypeNodes.def"
    default:
      llvm_unreachable("Unknown node");
    }
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
    for (unsigned k = 0, size = type->getNumElements(); k < size; ++k)
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
    for (unsigned k = 0, size = type->getNumArgs(); k < size; ++k)
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
    // Equivalence (T0 = T1, so T0 should have the same substitution as T1)
    return setSubstitution(type, other);
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

Type ConstraintSystem::simplifyType(Type type, Type defaultInt,
                                    Type defaultFloat) {
  if (!type->hasTypeVariable())
    return type;
  if (defaultInt.isNull())
    defaultInt = ctxt.i32Type;
  if (defaultFloat.isNull())
    defaultFloat = ctxt.f32Type;
  Type simplified = TypeSimplifier(*this, defaultInt, defaultFloat).visit(type);
  assert(simplified && !simplified->hasTypeVariable() && "Not simplified!");
  return simplified;
}

bool ConstraintSystem::unify(Type a, Type b,
                             const UnificationOptions &options) {
  // If both types don't contain any type variables, just compare them.
  if (!a->hasTypeVariable() && !b->hasTypeVariable())
    return options.typeComparator(a->getCanonicalType(), b->getCanonicalType());
  // Else, use the TypeUnifier.
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