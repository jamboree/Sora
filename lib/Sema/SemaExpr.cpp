//===--- SemaExpr.cpp -------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Expression Semantic Analysis
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Diagnostics/DiagnosticsSema.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

Expr *Sema::typecheckExpr(Expr *expr, DeclContext *dc) {
  struct Impl : ASTWalker {
    Sema &sema;
    DeclContext *dc;

    Impl(Sema &sema, DeclContext *dc) : sema(sema), dc(dc) {}

    SourceFile &getSourceFile() const {
      assert(dc && "no DeclContext?");
      SourceFile *sf = dc->getParentSourceFile();
      assert(sf && "no source file");
      return *sf;
    }

    std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
      auto *udre = dyn_cast<UnresolvedDeclRefExpr>(expr);
      if (!udre)
        return {true, expr};
      UnqualifiedValueLookup uvl(getSourceFile());
      llvm::outs() << "-----------------------------------\n";
      llvm::outs() << "Expression:\n";
      udre->dump(llvm::outs(), sema.ctxt.srcMgr);
      llvm::outs() << "Performing lookup...\n";
      uvl.performLookup(udre->getLoc(), udre->getIdentifier());
      llvm::outs() << "Results found: " << uvl.results.size() << "\n";
      unsigned k = 0;
      for (ValueDecl *result : uvl.results) {
        llvm::outs() << "Result #" << k++ << ":\n";
        result->dump(llvm::outs());
        llvm::outs() << "\n";
      }
      return {true, expr};
    }
  };
  Expr *replacement = expr->walk(Impl(*this, dc)).second;
  return replacement ? replacement : expr;
}