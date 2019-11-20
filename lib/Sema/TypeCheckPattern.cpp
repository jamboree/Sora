//===--- TypeCheckPattern.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Pattern Semantic Analysis
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Pattern.hpp"
#include "TypeChecker.hpp"

using namespace sora;

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::typecheckPattern(Pattern *pat) {
  assert(pat);
  class Impl : public ASTWalker {
  public:
    TypeChecker &tc;

    Impl(TypeChecker &tc) : tc(tc) {}

    bool walkToPatternPost(Pattern *pattern) override {
      if (VarPattern *var = dyn_cast<VarPattern>(pattern))
        tc.typecheckDecl(var->getVarDecl());
      return true;
    }
  };
  pat->walk(Impl(*this));
}