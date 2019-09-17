//===--- DeclTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class DeclTest : public ::testing::Test {
protected:
  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};
};
} // namespace

TEST_F(DeclTest, rtti) {
  // VarDecl
  {
    Decl *decl = new (*ctxt) VarDecl(SourceLoc(), Identifier());
    EXPECT_TRUE(isa<VarDecl>(decl));
    EXPECT_TRUE(isa<ValueDecl>(decl));
  }

  // ParamDecl
  {
    Decl *decl = new (*ctxt)
        ParamDecl(SourceLoc(), Identifier(), SourceLoc(), TypeLoc());
    EXPECT_TRUE(isa<ParamDecl>(decl));
    EXPECT_TRUE(isa<ValueDecl>(decl));
  }

  // FuncDecl
  {
    Decl *decl = new (*ctxt) FuncDecl(SourceLoc(), SourceLoc(), Identifier());
    EXPECT_TRUE(isa<FuncDecl>(decl));
    EXPECT_TRUE(isa<ValueDecl>(decl));
  }
}