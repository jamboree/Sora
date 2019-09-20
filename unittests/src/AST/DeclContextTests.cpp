//===--- DeclContextTests.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/DeclContext.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class DeclContextTest : public ::testing::Test {
protected:
  DeclContextTest() { func = new (*ctxt) FuncDecl(&sf, {}, {}, {}); }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  FuncDecl *func = nullptr;
  SourceFile sf{*ctxt, nullptr, {}};
};
} // namespace

TEST_F(DeclContextTest, isLocalContext) {
  DeclContext *dc = func;
  EXPECT_TRUE(dc->isLocalContext());
  dc = &sf;
  EXPECT_FALSE(dc->isLocalContext());
}

TEST_F(DeclContextTest, getAsDecl) {
  DeclContext *dc = func;
  EXPECT_EQ(dc->getAsDecl(), (Decl*)func);
  dc = &sf;
  EXPECT_EQ(dc->getAsDecl(), nullptr);
}