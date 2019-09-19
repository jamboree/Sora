//===--- DeclTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/TypeRepr.hpp"
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
    Decl *decl = new (*ctxt) VarDecl({}, {});
    EXPECT_TRUE(isa<VarDecl>(decl));
    EXPECT_TRUE(isa<ValueDecl>(decl));
  }

  // ParamDecl
  {
    Decl *decl = new (*ctxt) ParamDecl({}, {}, {}, TypeLoc());
    EXPECT_TRUE(isa<ParamDecl>(decl));
    EXPECT_TRUE(isa<ValueDecl>(decl));
  }

  // FuncDecl
  {
    Decl *decl = new (*ctxt) FuncDecl({}, {}, {});
    EXPECT_TRUE(isa<FuncDecl>(decl));
    EXPECT_TRUE(isa<ValueDecl>(decl));
  }

  // LetDecl
  {
    Decl *decl = LetDecl::create(*ctxt, {}, nullptr);
    EXPECT_TRUE(isa<LetDecl>(decl));
    EXPECT_TRUE(isa<PatternBindingDecl>(decl));
  }
}

TEST_F(DeclTest, getSourceRange) {
  const char *str = "Hello, World!";
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc mid = SourceLoc::fromPointer(str + 5);
  SourceLoc end = SourceLoc::fromPointer(str + 10);

  // VarDecl
  {
    Decl *decl = new (*ctxt) VarDecl(beg, {});
    EXPECT_EQ(beg, decl->getBegLoc());
    EXPECT_EQ(beg, decl->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), decl->getSourceRange());
  }

  // ParamDecl
  {
    Decl *decl = new (*ctxt) ParamDecl(
        beg, {}, {}, TypeLoc(Type(), new (*ctxt) IdentifierTypeRepr(end, {})));
    EXPECT_EQ(beg, decl->getBegLoc());
    EXPECT_EQ(end, decl->getEndLoc());
    EXPECT_EQ(SourceRange(beg, end), decl->getSourceRange());
  }

  // FuncDecl
  {
    Decl *decl = new (*ctxt) FuncDecl(beg, {}, {});
    cast<FuncDecl>(decl)->setBody(BlockStmt::createEmpty(*ctxt, mid, end));
    EXPECT_EQ(beg, decl->getBegLoc());
    EXPECT_EQ(end, decl->getEndLoc());
    EXPECT_EQ(SourceRange(beg, end), decl->getSourceRange());
  }

  // LetDecl
  {
    Decl *decl = LetDecl::create(*ctxt, beg, new (*ctxt) DiscardPattern(end));
    EXPECT_EQ(beg, decl->getBegLoc());
    EXPECT_EQ(end, decl->getEndLoc());
    EXPECT_EQ(SourceRange(beg, end), decl->getSourceRange());
    decl =
        LetDecl::create(*ctxt, beg, nullptr, {}, new (*ctxt) DiscardExpr(end));
    EXPECT_EQ(beg, decl->getBegLoc());
    EXPECT_EQ(end, decl->getEndLoc());
    EXPECT_EQ(SourceRange(beg, end), decl->getSourceRange());
  }
}

/// Add some tests exclusive to PBD & derived classes because it has some
/// special features.
TEST_F(DeclTest, PatternBindingDecl) {
  SourceLoc loc = SourceLoc::fromPointer("");

  Expr *expr = new (*ctxt) DiscardExpr({});
  PatternBindingDecl *pbd = LetDecl::create(*ctxt, {}, nullptr, loc, expr);

  EXPECT_TRUE(pbd->hasInitializer());
  EXPECT_EQ(pbd->getInitializer(), expr);
  EXPECT_EQ(pbd->getEqualLoc(), loc);

  pbd = LetDecl::create(*ctxt, {}, nullptr, loc);

  EXPECT_FALSE(pbd->hasInitializer());
  EXPECT_EQ(pbd->getInitializer(), nullptr);
  EXPECT_EQ(pbd->getEqualLoc(), SourceLoc());
}