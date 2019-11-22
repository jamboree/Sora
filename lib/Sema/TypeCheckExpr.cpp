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
  DeclContext *dc;

  ExprPrologue(TypeChecker &tc, DeclContext *dc) : tc(tc), dc(dc) {}

  SourceFile &getSourceFile() const {
    assert(dc && "no DeclContext?");
    SourceFile *sf = dc->getParentSourceFile();
    assert(sf && "no source file");
    return *sf;
  }

  std::pair<bool, Expr *> resolveUDRE(UnresolvedDeclRefExpr *udre) {
    UnqualifiedValueLookup uvl(getSourceFile());
    llvm::outs() << "-----------------------------------\n";
    llvm::outs() << "Expression:\n";
    udre->dump(llvm::outs(), tc.ctxt.srcMgr);
    llvm::outs() << "Performing lookup...\n";
    uvl.performLookup(udre->getLoc(), udre->getIdentifier());
    llvm::outs() << "Results found: " << uvl.results.size() << "\n";
    unsigned k = 0;
    for (ValueDecl *result : uvl.results) {
      llvm::outs() << "Result #" << k++ << ":\n";
      result->dump(llvm::outs());
      llvm::outs() << "\n";
    }
    return {true, udre};
  }

  std::pair<bool, Expr *> resolveCast(CastExpr *cast) {
    // Just resolve the cast's typeloc.
    tc.resolveTypeLoc(cast->getTypeLoc(), getSourceFile());
    return {true, cast};
  }

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    if (auto *udre = dyn_cast<UnresolvedDeclRefExpr>(expr))
      return resolveUDRE(udre);
    if (auto *cast = dyn_cast<CastExpr>(expr))
      return resolveCast(cast);
    return {true, expr};
  }
};
} // namespace

//===- TypeChecker --------------------------------------------------------===//

Expr *TypeChecker::typecheckExpr(Expr *expr, DeclContext *dc, Type ofType) {
  assert(expr && dc);
  expr = expr->walk(ExprPrologue(*this, dc)).second;
  return expr;
}