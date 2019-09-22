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
    Expr *expr = new (*ctxt) UnresolvedDeclRefExpr({}, {});
    EXPECT_TRUE(isa<UnresolvedDeclRefExpr>(expr));
    EXPECT_TRUE(isa<UnresolvedExpr>(expr));
  }

  // DiscardExpr
  {
    Expr *expr = new (*ctxt) DiscardExpr({});
    EXPECT_TRUE(isa<DiscardExpr>(expr));
  }

  // IntegerLiteralExpr
  {
    Expr *expr = new (*ctxt) IntegerLiteralExpr("0", {});
    EXPECT_TRUE(isa<IntegerLiteralExpr>(expr));
  }

  // FloatLiteralExpr
  {
    Expr *expr = new (*ctxt) FloatLiteralExpr("0", {});
    EXPECT_TRUE(isa<FloatLiteralExpr>(expr));
  }

  // BooleanLiteralExpr
  {
    Expr *expr = new (*ctxt) BooleanLiteralExpr(false, {});
    EXPECT_TRUE(isa<BooleanLiteralExpr>(expr));
  }

  // NullLiteralExpr
  {
    Expr *expr = new (*ctxt) NullLiteralExpr({});
    EXPECT_TRUE(isa<NullLiteralExpr>(expr));
  }

  // ErrorExpr
  {
    Expr *expr = new (*ctxt) ErrorExpr(SourceRange());
    EXPECT_TRUE(isa<ErrorExpr>(expr));
  }

  // TupleIndexingExpr
  {
    Expr *expr = new (*ctxt) TupleIndexingExpr(nullptr, {}, nullptr);
    EXPECT_TRUE(isa<TupleIndexingExpr>(expr));
  }

  // TupleExpr
  {
    Expr *expr = TupleExpr::createEmpty(*ctxt, {}, {});
    EXPECT_TRUE(isa<TupleExpr>(expr));
  }

  // ParenExpr
  {
    Expr *expr = new (*ctxt) ParenExpr({}, nullptr, {});
    EXPECT_TRUE(isa<ParenExpr>(expr));
  }

  // CallExpr
  {
    Expr *expr = new (*ctxt) CallExpr(nullptr, nullptr);
    EXPECT_TRUE(isa<CallExpr>(expr));
  }

  // BinaryExpr
  {
    Expr *expr =
        new (*ctxt) BinaryExpr(nullptr, BinaryOperatorKind::Add, {}, nullptr);
    EXPECT_TRUE(isa<BinaryExpr>(expr));
  }

  // UnaryExpr
  {
    Expr *expr =
        new (*ctxt) UnaryExpr(UnaryOperatorKind::AddressOf, {}, nullptr);
    EXPECT_TRUE(isa<UnaryExpr>(expr));
  }
}

TEST_F(ExprTest, getSourceRange) {
  const char *str = "Hello, World!";
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc mid = SourceLoc::fromPointer(str + 5);
  SourceLoc end = SourceLoc::fromPointer(str + 10);
  SourceRange range(beg, end);

  // UnresolvedDeclRefExpr
  {
    Expr *expr = new (*ctxt) UnresolvedDeclRefExpr({}, beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // DiscardExpr
  {
    Expr *expr = new (*ctxt) DiscardExpr(beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // IntegerLiteralExpr
  {
    Expr *expr = new (*ctxt) IntegerLiteralExpr("0", beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // FloatLiteralExpr
  {
    Expr *expr = new (*ctxt) FloatLiteralExpr("0", beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // BooleanLiteralExpr
  {
    Expr *expr = new (*ctxt) BooleanLiteralExpr("0", beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // NullLiteralExpr
  {
    Expr *expr = new (*ctxt) NullLiteralExpr(beg);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(beg, expr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), expr->getSourceRange());
  }

  // ErrorExpr
  {
    Expr *expr = new (*ctxt) ErrorExpr(range);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // TupleIndexingExpr
  {
    Expr *expr =
        new (*ctxt) TupleIndexingExpr(new (*ctxt) DiscardExpr(beg), mid,
                                      new (*ctxt) IntegerLiteralExpr("0", end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(mid, expr->getLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // TupleExpr
  {
    Expr *expr = TupleExpr::createEmpty(*ctxt, beg, end);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // ParenExpr
  {
    Expr *expr = new (*ctxt) ParenExpr(beg, new (*ctxt) DiscardExpr(mid), end);
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(mid, expr->getLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // CallExpr
  {
    Expr *expr = new (*ctxt) CallExpr(new (*ctxt) DiscardExpr(beg),
                                      TupleExpr::createEmpty(*ctxt, beg, end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(beg, expr->getLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // BinaryExpr
  {
    Expr *expr = new (*ctxt)
        BinaryExpr(new (*ctxt) DiscardExpr(beg), BinaryOperatorKind::Add, mid,
                   new (*ctxt) DiscardExpr(end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(mid, expr->getLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }

  // UnaryExpr
  {
    Expr *expr = new (*ctxt) UnaryExpr(UnaryOperatorKind::AddressOf, beg,
                                       new (*ctxt) DiscardExpr(end));
    EXPECT_EQ(beg, expr->getBegLoc());
    EXPECT_EQ(end, expr->getEndLoc());
    EXPECT_EQ(end, expr->getLoc());
    EXPECT_EQ(range, expr->getSourceRange());
  }
}