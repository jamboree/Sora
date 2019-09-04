//===--- ExprTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class ExprTest : public ::testing::Test {
protected:
  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};
};
} // namespace

TEST_F(ExprTest, rtti) {
  // UnresolvedDeclRefExpr
  {
    auto expr = new (*ctxt) UnresolvedDeclRefExpr(Identifier(), SourceLoc());
    EXPECT_TRUE(isa<UnresolvedDeclRefExpr>(expr));
    EXPECT_TRUE(isa<UnresolvedExpr>(expr));
  }

  // DiscardExpr
  {
    auto expr = new (*ctxt) DiscardExpr(SourceLoc());
    EXPECT_TRUE(isa<DiscardExpr>(expr));
  }

  // IntegerLiteralExpr
  {
    auto expr = new (*ctxt) IntegerLiteralExpr("0", SourceLoc());
    EXPECT_TRUE(isa<IntegerLiteralExpr>(expr));
  }

  // FloatLiteralExpr
  {
    auto expr = new (*ctxt) FloatLiteralExpr("0", SourceLoc());
    EXPECT_TRUE(isa<FloatLiteralExpr>(expr));
  }

  // BooleanLiteralExpr
  {
    auto expr = new (*ctxt) BooleanLiteralExpr(false, SourceLoc());
    EXPECT_TRUE(isa<BooleanLiteralExpr>(expr));
  }

  // NullLiteralExpr
  {
    auto expr = new (*ctxt) NullLiteralExpr(SourceLoc());
    EXPECT_TRUE(isa<NullLiteralExpr>(expr));
  }

  // ErrorExpr
  {
    auto expr = new (*ctxt) ErrorExpr(SourceRange());
    EXPECT_TRUE(isa<ErrorExpr>(expr));
  }

  // TupleIndexingExpr
  {
    auto expr = new (*ctxt) TupleIndexingExpr(nullptr, SourceLoc(), nullptr);
    EXPECT_TRUE(isa<TupleIndexingExpr>(expr));
  }

  // TupleExpr
  {
    auto expr = TupleExpr::createEmpty(*ctxt, SourceLoc(), SourceLoc());
    EXPECT_TRUE(isa<TupleExpr>(expr));
  }

  // ParenExpr
  {
    auto expr = new (*ctxt) ParenExpr(SourceLoc(), nullptr, SourceLoc());
    EXPECT_TRUE(isa<ParenExpr>(expr));
  }

  // CallExpr
  {
    auto expr = new (*ctxt) CallExpr(nullptr, nullptr);
    EXPECT_TRUE(isa<CallExpr>(expr));
  }

  // BinaryExpr
  {
    auto expr = new (*ctxt)
        BinaryExpr(nullptr, BinaryOperatorKind::Add, SourceLoc(), nullptr);
    EXPECT_TRUE(isa<BinaryExpr>(expr));
  }

  // UnaryExpr
  {
    auto expr = new (*ctxt)
        UnaryExpr(UnaryOperatorKind::AddressOf, SourceLoc(), nullptr);
    EXPECT_TRUE(isa<UnaryExpr>(expr));
  }
}

TEST_F(ExprTest, getSourceRange) {
  const char *str = "Hello, World!";
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc end = SourceLoc::fromPointer(str + 10);
  SourceRange range(beg, end);

  // UnresolvedDeclRefExpr
  {
    auto expr = new (*ctxt) UnresolvedDeclRefExpr(Identifier(), beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // DiscardExpr
  {
    auto expr = new (*ctxt) DiscardExpr(beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // IntegerLiteralExpr
  {
    auto expr = new (*ctxt) IntegerLiteralExpr("0", beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // FloatLiteralExpr
  {
    auto expr = new (*ctxt) FloatLiteralExpr("0", beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // BooleanLiteralExpr
  {
    auto expr = new (*ctxt) BooleanLiteralExpr("0", beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // NullLiteralExpr
  {
    auto expr = new (*ctxt) NullLiteralExpr(beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // ErrorExpr
  {
    auto expr = new (*ctxt) ErrorExpr(range);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // TupleIndexingExpr
  {
    auto expr =
        new (*ctxt) TupleIndexingExpr(new (*ctxt) DiscardExpr(beg), SourceLoc(),
                                      new (*ctxt) IntegerLiteralExpr("0", end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // TupleExpr
  {
    auto expr = TupleExpr::createEmpty(*ctxt, beg, end);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // ParenExpr
  {
    auto expr = new (*ctxt) ParenExpr(beg, nullptr, end);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // CallExpr
  {
    auto expr = new (*ctxt) CallExpr(new (*ctxt) DiscardExpr(beg),
                                     TupleExpr::createEmpty(*ctxt, beg, end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // BinaryExpr
  {
    auto expr = new (*ctxt)
        BinaryExpr(new (*ctxt) DiscardExpr(beg), BinaryOperatorKind::Add,
                   SourceLoc(), new (*ctxt) DiscardExpr(end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // UnaryExpr
  {
    auto expr = new (*ctxt) UnaryExpr(UnaryOperatorKind::AddressOf, beg,
                                      new (*ctxt) DiscardExpr(end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }
}