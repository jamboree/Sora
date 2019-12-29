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
class TypeReprResolver : public ASTChecker,
                         public TypeReprVisitor<TypeReprResolver, Type> {
public:
  SourceFile &file;

  TypeReprResolver(TypeChecker &tc, SourceFile &file)
      : ASTChecker(tc), file(file) {}

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
    : public ASTChecker,
      public TypeVisitor<ExplicitTypeCastChecker, bool, Type> {
  using Parent = TypeVisitor<ExplicitTypeCastChecker, bool, Type>;

public:
  ExplicitTypeCastChecker(TypeChecker &tc, ConstraintSystem &cs)
      : ASTChecker(tc), cs(cs) {}

  ConstraintSystem &cs;

  bool visit(Type from, Type to) {
    assert(!to->hasTypeVariable() && !to->hasErrorType() &&
           "the 'to' type should be a bound, error-free type!");
    assert(!from->hasErrorType() &&
           "the 'from' types shouldn't contain ErrorTypes!");

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

  bool visitBuiltinType(BuiltinType *from, Type to) {
    // We allow integer <-> float conversions, no matter the width/signedness.
    if (isa<IntegerType>(from) || isa<FloatType>(from))
      return to->is<IntegerType>() || to->is<FloatType>();
    // Else the types must be strictly equal
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
    // If the TypeVariable has a substitution, check if we can convert to it
    TypeVariableInfo &fromInfo = TypeVariableInfo::get(from);
    if (fromInfo.hasSubstitution())
      return visit(fromInfo.getSubstitution(), to);
    // Else just unify
    return cs.unify(from, to);
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

bool TypeChecker::canExplicitlyCast(ConstraintSystem &cs, Type from, Type to) {
  return ExplicitTypeCastChecker(*this, cs).visit(from, to);
}

//===- ASTChecker ---------------------------------------------------------===//

bool TypeChecker::canDiagnose(Type type) {
  return type && !type->hasErrorType();
}
