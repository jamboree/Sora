//===--- ExprTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
const char *str = "Hello, World!";

class ExprTest : public ::testing::Test {
protected:
  ExprTest() {
    // Setup locs
    beg = SourceLoc::fromPointer(str);
    mid = SourceLoc::fromPointer(str + 5);
    end = SourceLoc::fromPointer(str + 10);
    // Setup nodes
    unresolvedDeclRefExpr = new (*ctxt) UnresolvedDeclRefExpr({}, beg);
    unresolvedMemberRefExpr = new (*ctxt)
        UnresolvedMemberRefExpr(unresolvedDeclRefExpr, mid, false, end, {});
    discardExpr = new (*ctxt) DiscardExpr(beg);
    integerLiteralExpr = new (*ctxt) IntegerLiteralExpr("0", beg);
    floatLiteralExpr = new (*ctxt) FloatLiteralExpr("0", beg);
    booleanLiteralExpr = new (*ctxt) BooleanLiteralExpr("0", beg);
    nullLiteralExpr = new (*ctxt) NullLiteralExpr(beg);
    errorExpr = new (*ctxt) ErrorExpr({beg, end});
    tupleElementExpr = new (*ctxt)
        TupleElementExpr(new (*ctxt) DiscardExpr(beg), mid, false, end, 0);
    tupleExpr = TupleExpr::createEmpty(*ctxt, beg, end);
    parenExpr = new (*ctxt) ParenExpr(beg, new (*ctxt) DiscardExpr(mid), end);
    callExpr = new (*ctxt) CallExpr(new (*ctxt) DiscardExpr(beg),
                                    TupleExpr::createEmpty(*ctxt, beg, end));
    binaryExpr = new (*ctxt)
        BinaryExpr(new (*ctxt) DiscardExpr(beg), BinaryOperatorKind::Add, mid,
                   new (*ctxt) DiscardExpr(end));
    unaryExpr = new (*ctxt) UnaryExpr(UnaryOperatorKind::AddressOf, beg,
                                      new (*ctxt) DiscardExpr(end));
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, mid, end;

  Expr *unresolvedDeclRefExpr;
  Expr *unresolvedMemberRefExpr;
  Expr *discardExpr;
  Expr *integerLiteralExpr;
  Expr *floatLiteralExpr;
  Expr *booleanLiteralExpr;
  Expr *nullLiteralExpr;
  Expr *errorExpr;
  Expr *tupleElementExpr;
  Expr *tupleExpr;
  Expr *parenExpr;
  Expr *callExpr;
  Expr *binaryExpr;
  Expr *unaryExpr;
};
} // namespace

TEST_F(ExprTest, rtti) {
  EXPECT_TRUE(isa<UnresolvedDeclRefExpr>(unresolvedDeclRefExpr));
  EXPECT_TRUE(isa<UnresolvedExpr>(unresolvedDeclRefExpr));
  EXPECT_TRUE(isa<UnresolvedMemberRefExpr>(unresolvedMemberRefExpr));
  EXPECT_TRUE(isa<UnresolvedExpr>(unresolvedMemberRefExpr));
  EXPECT_TRUE(isa<DiscardExpr>(discardExpr));
  EXPECT_TRUE(isa<IntegerLiteralExpr>(integerLiteralExpr));
  EXPECT_TRUE(isa<AnyLiteralExpr>(integerLiteralExpr));
  EXPECT_TRUE(isa<FloatLiteralExpr>(floatLiteralExpr));
  EXPECT_TRUE(isa<AnyLiteralExpr>(floatLiteralExpr));
  EXPECT_TRUE(isa<BooleanLiteralExpr>(booleanLiteralExpr));
  EXPECT_TRUE(isa<AnyLiteralExpr>(booleanLiteralExpr));
  EXPECT_TRUE(isa<NullLiteralExpr>(nullLiteralExpr));
  EXPECT_TRUE(isa<AnyLiteralExpr>(nullLiteralExpr));
  EXPECT_TRUE(isa<ErrorExpr>(errorExpr));
  EXPECT_TRUE(isa<TupleElementExpr>(tupleElementExpr));
  EXPECT_TRUE(isa<TupleExpr>(tupleExpr));
  EXPECT_TRUE(isa<ParenExpr>(parenExpr));
  EXPECT_TRUE(isa<CallExpr>(callExpr));
  EXPECT_TRUE(isa<BinaryExpr>(binaryExpr));
  EXPECT_TRUE(isa<UnaryExpr>(unaryExpr));
}

TEST_F(ExprTest, getSourceRange) {
  // UnresolvedDeclRefExpr
  EXPECT_EQ(beg, unresolvedDeclRefExpr->getBegLoc());
  EXPECT_EQ(beg, unresolvedDeclRefExpr->getLoc());
  EXPECT_EQ(beg, unresolvedDeclRefExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), unresolvedDeclRefExpr->getSourceRange());

  // UnresolvedDotExpr
  EXPECT_EQ(beg, unresolvedMemberRefExpr->getBegLoc());
  EXPECT_EQ(end, unresolvedMemberRefExpr->getEndLoc());
  EXPECT_EQ(mid, unresolvedMemberRefExpr->getLoc());
  EXPECT_EQ(SourceRange(beg, end), unresolvedMemberRefExpr->getSourceRange());

  // DiscardExpr
  EXPECT_EQ(beg, discardExpr->getBegLoc());
  EXPECT_EQ(beg, discardExpr->getLoc());
  EXPECT_EQ(beg, discardExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), discardExpr->getSourceRange());

  // IntegerLiteralExpr
  EXPECT_EQ(beg, integerLiteralExpr->getBegLoc());
  EXPECT_EQ(beg, integerLiteralExpr->getLoc());
  EXPECT_EQ(beg, integerLiteralExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), integerLiteralExpr->getSourceRange());

  // FloatLiteralExpr
  EXPECT_EQ(beg, floatLiteralExpr->getBegLoc());
  EXPECT_EQ(beg, floatLiteralExpr->getLoc());
  EXPECT_EQ(beg, floatLiteralExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), floatLiteralExpr->getSourceRange());

  // BooleanLiteralExpr
  EXPECT_EQ(beg, booleanLiteralExpr->getBegLoc());
  EXPECT_EQ(beg, booleanLiteralExpr->getLoc());
  EXPECT_EQ(beg, booleanLiteralExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), booleanLiteralExpr->getSourceRange());

  // NullLiteralExpr
  EXPECT_EQ(beg, nullLiteralExpr->getBegLoc());
  EXPECT_EQ(beg, nullLiteralExpr->getLoc());
  EXPECT_EQ(beg, nullLiteralExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), nullLiteralExpr->getSourceRange());

  // ErrorExpr
  EXPECT_EQ(beg, errorExpr->getBegLoc());
  EXPECT_EQ(beg, errorExpr->getLoc());
  EXPECT_EQ(end, errorExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), errorExpr->getSourceRange());

  // TupleElementExpr
  EXPECT_EQ(beg, tupleElementExpr->getBegLoc());
  EXPECT_EQ(end, tupleElementExpr->getEndLoc());
  EXPECT_EQ(mid, tupleElementExpr->getLoc());
  EXPECT_EQ(SourceRange(beg, end), tupleElementExpr->getSourceRange());

  // TupleExpr
  EXPECT_EQ(beg, tupleExpr->getBegLoc());
  EXPECT_EQ(beg, tupleExpr->getLoc());
  EXPECT_EQ(end, tupleExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), tupleExpr->getSourceRange());

  // ParenExpr
  EXPECT_EQ(beg, parenExpr->getBegLoc());
  EXPECT_EQ(end, parenExpr->getEndLoc());
  EXPECT_EQ(mid, parenExpr->getLoc());
  EXPECT_EQ(SourceRange(beg, end), parenExpr->getSourceRange());

  // CallExpr
  EXPECT_EQ(beg, callExpr->getBegLoc());
  EXPECT_EQ(beg, callExpr->getLoc());
  EXPECT_EQ(end, callExpr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), callExpr->getSourceRange());

  // BinaryExpr
  EXPECT_EQ(beg, binaryExpr->getBegLoc());
  EXPECT_EQ(end, binaryExpr->getEndLoc());
  EXPECT_EQ(mid, binaryExpr->getLoc());
  EXPECT_EQ(SourceRange(beg, end), binaryExpr->getSourceRange());

  // UnaryExpr
  EXPECT_EQ(beg, unaryExpr->getBegLoc());
  EXPECT_EQ(end, unaryExpr->getEndLoc());
  EXPECT_EQ(end, unaryExpr->getLoc());
  EXPECT_EQ(SourceRange(beg, end), unaryExpr->getSourceRange());
}