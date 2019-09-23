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

  ContinueStmt *continueStmt;
  BreakStmt *breakStmt;
  ReturnStmt *returnStmt;
  BlockStmt *blockStmt;
  IfStmt *ifStmt;
  WhileStmt *whileStmt;
};
} // namespace

TEST_F(StmtTest, rtti) {
  EXPECT_TRUE(isa<ContinueStmt>((Stmt *)continueStmt));
  EXPECT_TRUE(isa<BreakStmt>((Stmt *)breakStmt));
  EXPECT_TRUE(isa<ReturnStmt>((Stmt *)returnStmt));
  EXPECT_TRUE(isa<BlockStmt>((Stmt *)blockStmt));
  EXPECT_TRUE(isa<IfStmt>((Stmt *)ifStmt));
  EXPECT_TRUE(isa<WhileStmt>((Stmt *)whileStmt));
}

TEST_F(StmtTest, getSourceRange) {
  Stmt *cur = nullptr;

  // ContinueStmt
  cur = continueStmt;
  EXPECT_EQ(cur->getLoc(), beg);
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), beg);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, beg));
  EXPECT_EQ(cur->getLoc(), beg);

  // BreakStmt
  cur = breakStmt;
  EXPECT_EQ(cur->getLoc(), beg);
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), beg);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, beg));
  EXPECT_EQ(cur->getLoc(), beg);

  // ReturnStmt
  cur = returnStmt;
  EXPECT_EQ(cur->getLoc(), beg);
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), beg);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, beg));
  returnStmt->setResult(new (*ctxt) DiscardExpr(end));
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), end);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, end));
  returnStmt->setResult(nullptr);

  // BlockStmt
  cur = blockStmt;
  EXPECT_EQ(cur->getLoc(), beg);
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), end);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, end));

  // IfStmt
  cur = ifStmt;
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), mid);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, mid));
  ifStmt->setElse(new (*ctxt) ContinueStmt(end));
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), end);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, end));
  ifStmt->setElse(nullptr);

  // WhileStmt
  cur = whileStmt;
  EXPECT_EQ(cur->getLoc(), beg);
  EXPECT_EQ(cur->getBegLoc(), beg);
  EXPECT_EQ(cur->getEndLoc(), end);
  EXPECT_EQ(cur->getSourceRange(), SourceRange(beg, end));
}