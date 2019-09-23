//===--- StmtTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
const char *str = "Hello, World!";

class StmtTest : public ::testing::Test {
protected:
  StmtTest() {
    // Setup SourceLocs
    beg = SourceLoc::fromPointer(str);
    mid = SourceLoc::fromPointer(str + 5);
    end = SourceLoc::fromPointer(str + 10);
    // Setup Nodes
    continueStmt = new (*ctxt) ContinueStmt(beg);
    breakStmt = new (*ctxt) BreakStmt(beg);
    returnStmt = new (*ctxt) ReturnStmt(beg);
    blockStmt = BlockStmt::createEmpty(*ctxt, beg, end);
    ifStmt = new (*ctxt) IfStmt(beg, nullptr, new (*ctxt) ContinueStmt(mid));
    whileStmt =
        new (*ctxt) WhileStmt(beg, nullptr, new (*ctxt) ContinueStmt(end));
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, mid, end;

  Stmt *continueStmt;
  Stmt *breakStmt;
  Stmt *returnStmt;
  Stmt *blockStmt;
  Stmt *ifStmt;
  Stmt *whileStmt;
};
} // namespace

TEST_F(StmtTest, rtti) {
  EXPECT_TRUE(isa<ContinueStmt>(continueStmt));
  EXPECT_TRUE(isa<BreakStmt>(breakStmt));
  EXPECT_TRUE(isa<ReturnStmt>(returnStmt));
  EXPECT_TRUE(isa<BlockStmt>(blockStmt));
  EXPECT_TRUE(isa<IfStmt>(ifStmt));
  EXPECT_TRUE(isa<WhileStmt>(whileStmt));
}

TEST_F(StmtTest, getSourceRange) {
  // ContinueStmt
  EXPECT_EQ(continueStmt->getLoc(), beg);
  EXPECT_EQ(continueStmt->getBegLoc(), beg);
  EXPECT_EQ(continueStmt->getEndLoc(), beg);
  EXPECT_EQ(continueStmt->getSourceRange(), SourceRange(beg, beg));
  EXPECT_EQ(continueStmt->getLoc(), beg);

  // BreakStmt
  EXPECT_EQ(breakStmt->getLoc(), beg);
  EXPECT_EQ(breakStmt->getBegLoc(), beg);
  EXPECT_EQ(breakStmt->getEndLoc(), beg);
  EXPECT_EQ(breakStmt->getSourceRange(), SourceRange(beg, beg));
  EXPECT_EQ(breakStmt->getLoc(), beg);

  // ReturnStmt
  EXPECT_EQ(returnStmt->getLoc(), beg);
  EXPECT_EQ(returnStmt->getBegLoc(), beg);
  EXPECT_EQ(returnStmt->getEndLoc(), beg);
  EXPECT_EQ(returnStmt->getSourceRange(), SourceRange(beg, beg));
  cast<ReturnStmt>(returnStmt)->setResult(new (*ctxt) DiscardExpr(end));
  EXPECT_EQ(returnStmt->getBegLoc(), beg);
  EXPECT_EQ(returnStmt->getEndLoc(), end);
  EXPECT_EQ(returnStmt->getSourceRange(), SourceRange(beg, end));
  cast<ReturnStmt>(returnStmt)->setResult(nullptr);

  // BlockStmt
  EXPECT_EQ(blockStmt->getLoc(), beg);
  EXPECT_EQ(blockStmt->getBegLoc(), beg);
  EXPECT_EQ(blockStmt->getEndLoc(), end);
  EXPECT_EQ(blockStmt->getSourceRange(), SourceRange(beg, end));

  // IfStmt
  EXPECT_EQ(ifStmt->getBegLoc(), beg);
  EXPECT_EQ(ifStmt->getLoc(), beg);
  EXPECT_EQ(ifStmt->getEndLoc(), mid);
  EXPECT_EQ(ifStmt->getSourceRange(), SourceRange(beg, mid));
  cast<IfStmt>(ifStmt)->setElse(new (*ctxt) ContinueStmt(end));
  EXPECT_EQ(ifStmt->getBegLoc(), beg);
  EXPECT_EQ(ifStmt->getEndLoc(), end);
  EXPECT_EQ(ifStmt->getSourceRange(), SourceRange(beg, end));
  cast<IfStmt>(ifStmt)->setElse(nullptr);

  // WhileStmt
  EXPECT_EQ(whileStmt->getLoc(), beg);
  EXPECT_EQ(whileStmt->getBegLoc(), beg);
  EXPECT_EQ(whileStmt->getEndLoc(), end);
  EXPECT_EQ(whileStmt->getSourceRange(), SourceRange(beg, end));
}