//===--- TypeCheckType.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Type Semantic Analysis (Type, TypeRepr, etc.)
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "ConstraintSystem.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/AST/TypeVisitor.hpp"
#include "Sora/AST/Types.hpp"

using namespace sora;

//===- TypeReprResolver ---------------------------------------------------===//

namespace {
/// Resolves TypeReprs into types.
class TypeReprResolver : public ASTCheckerBase,
                         public TypeReprVisitor<TypeReprResolver, Type> {
public:
  SourceFile &file;

  TypeReprResolver(TypeChecker &tc, SourceFile &file)
      : ASTCheckerBase(tc), file(file) {}

  Type visitIdentifierTypeRepr(IdentifierTypeRepr *tyRepr) {
    UnqualifiedTypeLookup utl(file);
    utl.performLookup(tyRepr->getIdentifierLoc(), tyRepr->getIdentifier());
    // Handle the lookup result

    // Currently, as there's only built-in types, we can only have 1 or 0
    // results.
    auto &results = utl.results;
    if (results.empty()) {
      diagnose(tyRepr->getLoc(), diag::cannot_find_type_in_scope,
               tyRepr->getIdentifier());
      return tc.ctxt.errorType;
    }

    if (results.size() > 1)
      llvm_unreachable("Multiple lookup results are not supported");
    return results[0];
  }

  Type visitParenTypeRepr(ParenTypeRepr *tyRepr) {
    // No corresponding type.
    return visit(tyRepr->getSubTypeRepr());
  }

  Type visitTupleTypeRepr(TupleTypeRepr *tyRepr) {
    std::vector<Type> tupleElts;
    tupleElts.reserve(tyRepr->getNumElements());
    for (TypeRepr *elt : tyRepr->getElements())
      tupleElts.push_back(visit(elt));
    return TupleType::get(tc.ctxt, tupleElts);
  }

  Type visitReferenceTypeRepr(ReferenceTypeRepr *tyRepr) {
    return ReferenceType::get(visit(tyRepr->getSubTypeRepr()),
                              tyRepr->hasMut());
  }

  Type visitMaybeTypeRepr(MaybeTypeRepr *tyRepr) {
    return MaybeType::get(visit(tyRepr->getSubTypeRepr()));
  }
};
} // namespace

//===- ExplicitTypeCastChecker --------------------------------------------===//

/// The TypeCastCheckers checks that a cast from a given type to another type is
/// valid.
class ExplicitTypeCastChecker
    : public TypeVisitor<ExplicitTypeCastChecker, bool, Type> {
  using Parent = TypeVisitor<ExplicitTypeCastChecker, bool, Type>;

public:
  ExplicitTypeCastChecker(TypeChecker &tc, const ConstraintSystem &cs)
      : tc(tc), cs(cs) {}

  TypeChecker &tc;
  const ConstraintSystem &cs;

  bool visit(Type from, Type to) {
    // Ignore sugar & RValues on both sides.
    from = from->getRValue()->getDesugaredType();
    to = to->getRValue()->getDesugaredType();

    // If the types are canonically equal, it's valid
    if (from->getCanonicalType() == to->getCanonicalType())
      return true;

    // Else we must visit the 'from' type
    return Parent::visit(from, to);
  }

  // Note: visit methods are only called when the types are already known to not
  // be strictly equal. They should return true if the conversion can happen,
  // false otherwise.

  /// \returns true if ANY integer type can be explicitly converted to the type
  /// \p to.
  bool canConvertIntegerTypeTo(Type to) {
    return to->isAnyIntegerType() || to->isAnyFloatType() || to->isBoolType();
  }

  /// \returns true if ANY floating-point type can be explicitly converted to
  /// the type \p to.
  bool canConvertFloatTypeTo(Type to) {
    return to->isAnyIntegerType() || to->isAnyFloatType() || to->isBoolType();
  }

  bool visitBuiltinType(BuiltinType *from, Type to) {
    // Check bools.
    if (from->isAnyIntegerType())
      return canConvertIntegerTypeTo(to);
    // Check floats.
    if (from->isAnyFloatType())
      return canConvertFloatTypeTo(to);
    // Booleans can be converted to int/float.
    if (from->isBoolType())
      return to->isAnyIntegerType() || to->isAnyFloatType();
    // Else the types must be strictly equal.
    return false;
  }

  bool visitReferenceType(ReferenceType *from, Type to) {
    // We can't convert reference types between each other. If they're not
    // strictly equal, conversion can't happen.
    return false;
  }

  bool visitMaybeType(MaybeType *from, Type to) {
    // We can only convert a maybe type to another maybe type, and only if the
    // object types can also be converted.
    if (MaybeType *toMaybe = to->getAs<MaybeType>())
      return visit(from->getValueType(), toMaybe->getValueType());
    return false;
  }

  bool visitTupleType(TupleType *from, Type to) {
    // We can only convert a tuple type to another tuple type, and only they
    // have the same number of elements and elements types can also be
    // converted.
    TupleType *toTuple = to->getAs<TupleType>();
    if (!toTuple)
      return false;
    if (from->getNumElements() != toTuple->getNumElements())
      return false;

    size_t numElems = from->getNumElements();
    for (size_t k = 0; k < numElems; ++k)
      if (!visit(from->getElement(k), toTuple->getElement(k)))
        return false;
    return true;
  }

  bool visitFunctionType(FunctionType *from, Type to) {
    // We can't convert function types between each other. If they're not
    // strictly equal, conversion can't happen.
    return false;
  }

  bool visitLValueType(LValueType *from, Type to) {
    llvm_unreachable("LValues should be ignored!");
  }

  bool visitErrorType(ErrorType *from, Type to) {
    llvm_unreachable("ErrorType are not allowed");
  }

  bool visitTypeVariableType(TypeVariableType *from, Type to) {
    // Dereference bound type variables
    if (Type fromSubst = cs.getSubstitution(from))
      return visit(fromSubst, to);
    // For Integer Type Variables, use canConvertIntegerTypeTo
    if (cs.isIntegerTypeVariable(from) && canConvertIntegerTypeTo(to))
      return true;
    // For Float Type Variables, use canConvertFloatTypeTo
    if (cs.isFloatTypeVariable(from) && canConvertFloatTypeTo(to))
      return true;
    // Else, the conversion is only possible if unification can happen.
    return cs.canUnify(from, to);
  }
};

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::resolveTypeLoc(TypeLoc &tyLoc, SourceFile &file) {
  assert(tyLoc.hasTypeRepr() && "Must have a TypeRepr");
  assert(!tyLoc.hasType() && "TypeLoc already resolved!");
  tyLoc.setType(resolveTypeRepr(tyLoc.getTypeRepr(), file));
  assert(tyLoc.hasType() && "Type not set?");
}

Type TypeChecker::resolveTypeRepr(TypeRepr *tyRepr, SourceFile &file) {
  assert(tyRepr);
  TypeReprResolver resolver(*this, file);
  Type result = resolver.visit(tyRepr);
  assert(result);
  return result;
}

bool TypeChecker::canExplicitlyCast(const ConstraintSystem &cs, Type from,
                                    Type to) {
  assert(!to->hasTypeVariable() &&
         "the 'to' type cannot contain type variables");
  assert(!to->hasErrorType() && "the 'to' type cannot contain error types");
  assert(!from->hasErrorType() && "the 'from' type cannot contain error types");
  return ExplicitTypeCastChecker(*this, cs).visit(from, to);
}

//===- ASTCheckerBase -----------------------------------------------------===//

bool TypeChecker::canImplicitlyCast(const ConstraintSystem &cs, Type from,
                                    Type to) {
  // FIXME: Ideally this should follow the same pattern as canExplicitlyCast
  assert(!to->hasErrorType() && "the 'to' type cannot contain error types");
  assert(!from->hasErrorType() && "the 'from' type cannot contain error types");

  // FIXME: Calling canUnify here isn't ideal, especially since recursion is
  // involved. This can have a huge performance cost for complex types.
  if (cs.canUnify(from, to))
    return true;

  to = to->getRValue()->getDesugaredType();

  // T to maybe T conversions, or null to maybe T conversion
  if (MaybeType *toMaybe = to->getAs<MaybeType>())
    return canImplicitlyCast(cs, from, toMaybe->getValueType()) ||
           from->isNullType();

  // &mut T to &T conversions
  if (ReferenceType *toRef = to->getAs<ReferenceType>()) {
    // "from" must also be a reference type
    ReferenceType *fromRef = from->getAs<ReferenceType>();
    if (!fromRef)
      return false;
    // toRef must be immutable and fromRef must be mutable
    if (fromRef->isMut() && !toRef->isMut()) {
      // The pointee types must unify
      return cs.canUnify(fromRef->getPointeeType(), toRef->getPointeeType());
    }
    return false;
  }

  // Tuple to Tuple conversions
  if (TupleType *toTuple = to->getAs<TupleType>()) {
    // "from" must also be a tuple type
    TupleType *fromTuple = from->getAs<TupleType>();

    // Both must have the same number of elts
    if (toTuple->getNumElements() != fromTuple->getNumElements())
      return false;

    // Every element in 'fromTuple' must be convertible to its counterpart in
    // 'toTuple'
    size_t numElts = toTuple->getNumElements();
    for (size_t k = 0; k < numElts; ++k) {
      if (!canImplicitlyCast(cs, fromTuple->getElement(k),
                             toTuple->getElement(k)))
        return false;
    }

    return true;
  }

  return false;
}

bool TypeChecker::canDiagnose(Type type) {
  return type && !type->hasErrorType();
}
