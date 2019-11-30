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

  TypeSimplifier(ConstraintSystem &cs, bool &success)
      : cs(cs), success(success) {}

  ConstraintSystem &cs;
  bool &success;

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
    Type subst = TypeVariableInfo::get(type).getSubstitution();
    // Return ErrorType when there are no substitutions
    if (!subst) {
      success = false;
      return cs.ctxt.errorType;
    }
    success &= true;
    // Recursively simplify TypeVariableTypes
    if (Type simplified = visit(subst))
      return simplified;
    return subst;
  }
};
} // namespace

//===--- ConstraintSystem -------------------------------------------------===//

TypeVariableType *ConstraintSystem::createTypeVariable(TypeVariableKind kind) {
  // Calculate the total size we need
  size_t size = sizeof(TypeVariableType) + sizeof(TypeVariableInfo);
  void *mem = ctxt.allocate(size, alignof(TypeVariableType),
                            ArenaKind::ConstraintSystem);
  /// Create the TypeVariableType
  TypeVariableType *tyVar =
      new (mem) TypeVariableType(ctxt, nextTypeVariableID++);
  /// Create the info
  new (reinterpret_cast<void *>(tyVar + 1)) TypeVariableInfo(kind);
  return tyVar;
}

Type ConstraintSystem::simplifyType(Type type, bool &success) {
  success = true;
  if (!type->hasTypeVariable())
    return type;
  Type simplified = TypeSimplifier(*this, success).visit(type);
  assert(simplified && !type->hasTypeVariable() && "Not simplified!");
  return simplified;
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

std::string
ConstraintSystem::getString(const TypeVariableType *type,
                            const TypePrintOptions &printOptions) const {
  std::string str;
  llvm::raw_string_ostream out(str);
  print(out, type, printOptions);
  return out.str();
}