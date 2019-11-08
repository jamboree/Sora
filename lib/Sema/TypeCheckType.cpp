//===--- TypeCheckType.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Type Semantic Analysis (Type, TypeRepr, etc.)
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/AST/Types.hpp"
#include <vector>

using namespace sora;

//===- TypeReprResolver ---------------------------------------------------===//

namespace {
/// Resolves TypeReprs into types.
class TypeReprResolver : public TypeReprVisitor<TypeReprResolver, Type> {
public:
  TypeChecker &tc;
  SourceFile &file;

  TypeReprResolver(TypeChecker &tc, SourceFile &file) : tc(tc), file(file) {}

  Type visitIdentifierTypeRepr(IdentifierTypeRepr *tyRepr) {
    UnqualifiedTypeLookup utl(file);
    utl.performLookup(tyRepr->getIdentifierLoc(), tyRepr->getIdentifier());
    // Handle the lookup result

    // Currently, as there's only built-in types, we can only have 1 or 0
    // results.
    auto &results = utl.results;
    if (results.empty()) {
      tc.diagnose(tyRepr->getLoc(), diag::cannot_find_type_in_scope,
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

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::resolveTypeLoc(TypeLoc &tyLoc, SourceFile &file) {
  assert(tyLoc.hasTypeRepr() && "Must have a TypeRepr");
  assert(!tyLoc.hasType() && "TypeLoc already resolved!");
  TypeReprResolver resolver(*this, file);
  tyLoc.setType(resolver.visit(tyLoc.getTypeRepr()));
}