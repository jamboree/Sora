//===--- TypeCheckExpr.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Expression Semantic Analysis
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Diagnostics/DiagnosticsSema.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

//===- ExprTypeCheckingPrologue -------------------------------------------===//

namespace {
/// The Prologue of Expression Type-Checking, which prepares the expression for
/// constraint solving.
///
/// Resolves UnresolvedDeclRefExprs and asExpr types.
class ExprPrologue : public ASTWalker {
public:
  TypeChecker &tc;
  ASTContext &ctxt;
  DeclContext *dc;

  ExprPrologue(TypeChecker &tc, DeclContext *dc)
      : tc(tc), ctxt(tc.ctxt), dc(dc) {}

  SourceFile &getSourceFile() const {
    assert(dc && "no DeclContext?");
    SourceFile *sf = dc->getParentSourceFile();
    assert(sf && "no source file");
    return *sf;
  }

  DeclRefExpr *resolve(UnresolvedDeclRefExpr *udre, ValueDecl *resolved) {
    DeclRefExpr *expr = new (ctxt) DeclRefExpr(udre, resolved);
    expr->setType(resolved->getValueType());
    llvm::outs() << "\tresolved\n";
    return expr;
  }

  Expr *tryResolveUDRE(UnresolvedDeclRefExpr *udre) {
    llvm::outs() << "trying for '" << udre->getIdentifier() << "'\n";
    // Lookup
    UnqualifiedValueLookup uvl(getSourceFile());
    uvl.performLookup(udre->getLoc(), udre->getIdentifier());
    // Perfect scenario: only one result
    if (ValueDecl *decl = uvl.getUniqueResult())
      return resolve(udre, decl);
    // Else, we got an error: emit a diagnostic depending on the situation
    if (uvl.isEmpty()) {
      tc.diagnose(udre->getLoc(), diag::cannot_find_value_in_scope,
                  udre->getIdentifier());
    }
    else {
      assert(uvl.results.size() >= 2);
      tc.diagnose(udre->getLoc(), diag::reference_to_value_is_ambiguous,
                  udre->getIdentifier());
      for (ValueDecl *candidate : uvl.results)
        tc.diagnose(candidate->getLoc(), diag::potential_candidate_found_here);
    }
    // and just return an ErrorExpr.
    return new (ctxt) ErrorExpr(udre);
  }

  Expr *handleCast(CastExpr *cast) {
    tc.resolveTypeLoc(cast->getTypeLoc(), getSourceFile());
    return cast;
  }

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    if (auto *udre = dyn_cast<UnresolvedDeclRefExpr>(expr))
      return {true, tryResolveUDRE(udre)};
    if (auto *cast = dyn_cast<CastExpr>(expr))
      return {true, handleCast(cast)};
    return {true, expr};
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *TypeChecker::typecheckExpr(Expr *expr, DeclContext *dc, Type ofType) {
  assert(expr && dc);
  expr = expr->walk(ExprPrologue(*this, dc)).second;
  assert(expr && "walk returns null?");
  return expr;
}