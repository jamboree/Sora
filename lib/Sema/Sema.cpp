//===--- Sema.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"
#include "Sora/AST/ASTContext.hpp"

using namespace sora;

Sema::Sema(ASTContext &ctxt) : ctxt(ctxt), diagEngine(ctxt.diagEngine) {}

//===----------------------------------------------------------------------===//
// This is currently just for testing purposes and will be refactored later.
// We're just going to perform name binding on every UnresolvedDeclRef in
// the source file.
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/NameLookup.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Diagnostics/DiagnosticsSema.hpp"
#include "llvm/Support/raw_ostream.h"

void Sema::performSema(SourceFile &file) {
  struct Impl : ASTWalker {
    Sema &sema;
    SourceFile &file;

    Impl(Sema &sema, SourceFile &file) : sema(sema), file(file) {}

    std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
      auto *udre = dyn_cast<UnresolvedDeclRefExpr>(expr);
      if (!udre)
        return {true, expr};
      UnqualifiedLookup ul(file);
      ul.performLookup(udre->getLoc(), udre->getIdentifier());
      llvm::outs() << "-----------------------------------\n";
      llvm::outs() << "Expression:\n";
      udre->dump(llvm::outs(), sema.ctxt.srcMgr);
      llvm::outs() << "Results found: " << ul.results.size() << "\n";
      unsigned k = 0;
      for (ValueDecl *result : ul.results) {
        llvm::outs() << "Result #" << k++ << ":";
        result->dump(llvm::outs());
        llvm::outs() << "\n";
      }
      return {true, expr};
    }
  };
  llvm::outs() << "Attempting to find candidates\n";
  file.walk(Impl(*this, file));
}