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
class StmtTest : public ::testing::Test {
protected:
  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};
};
} // namespace

TEST_F(StmtTest, rtti) {
  // ContinueStmt
  {
    Stmt *stmt = new (*ctxt) ContinueStmt({});
    EXPECT_TRUE(isa<ContinueStmt>(stmt));
  }

  // BreakStmt
  {
    Stmt *stmt = new (*ctxt) BreakStmt({});
    EXPECT_TRUE(isa<BreakStmt>(stmt));
  }

  // ReturnStmt
  {
    Stmt *stmt = new (*ctxt) ReturnStmt({});
    EXPECT_TRUE(isa<ReturnStmt>(stmt));
  }

  // BlockStmt
  {
    Stmt *stmt = BlockStmt::createEmpty(*ctxt, {}, {});
    EXPECT_TRUE(isa<BlockStmt>(stmt));
  }

  // IfStmt
  {
    Stmt *stmt = new (*ctxt) IfStmt({}, nullptr, nullptr);
    EXPECT_TRUE(isa<IfStmt>(stmt));
  }

  // WhileStmt
  {
    Stmt *stmt = new (*ctxt) WhileStmt({}, nullptr, nullptr);
    EXPECT_TRUE(isa<WhileStmt>(stmt));
  }
}

TEST_F(StmtTest, getSourceRange) {
  const char *str = "Hello, World!";
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc mid = SourceLoc::fromPointer(str + 5);
  SourceLoc end = SourceLoc::fromPointer(str + 10);
  SourceRange range(beg, end);

  // ContinueStmt
  {
    Stmt *stmt = new (*ctxt) ContinueStmt(beg);
    EXPECT_EQ(stmt->getLoc(), beg);
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), beg);
    EXPECT_EQ(stmt->getSourceRange(), SourceRange(beg, beg));
    EXPECT_EQ(cast<ContinueStmt>(stmt)->getLoc(), beg);
  }

  // BreakStmt
  {
    Stmt *stmt = new (*ctxt) BreakStmt(beg);
    EXPECT_EQ(stmt->getLoc(), beg);
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), beg);
    EXPECT_EQ(stmt->getSourceRange(), SourceRange(beg, beg));
    EXPECT_EQ(cast<BreakStmt>(stmt)->getLoc(), beg);
  }

  // ReturnStmt
  {
    Stmt *stmt = new (*ctxt) ReturnStmt(beg);
    EXPECT_EQ(stmt->getLoc(), beg);
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), beg);
    EXPECT_EQ(stmt->getSourceRange(), SourceRange(beg, beg));
    cast<ReturnStmt>(stmt)->setResult(new (*ctxt) DiscardExpr(end));
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), end);
    EXPECT_EQ(stmt->getSourceRange(), range);
  }

  // BlockStmt
  {
    Stmt *stmt = BlockStmt::createEmpty(*ctxt, beg, end);
    EXPECT_EQ(stmt->getLoc(), beg);
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), end);
    EXPECT_EQ(stmt->getSourceRange(), range);
  }

  // IfStmt
  {
    Stmt *stmt =
        new (*ctxt) IfStmt(beg, nullptr, new (*ctxt) ContinueStmt(mid));
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), mid);
    EXPECT_EQ(stmt->getSourceRange(), SourceRange(beg, mid));
    cast<IfStmt>(stmt)->setElse(new (*ctxt) ContinueStmt(end));
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), end);
    EXPECT_EQ(stmt->getSourceRange(), range);
  }

  // WhileStmt
  {
    Stmt *stmt =
        new (*ctxt) WhileStmt(beg, nullptr, new (*ctxt) ContinueStmt(end));
    EXPECT_EQ(stmt->getLoc(), beg);
    EXPECT_EQ(stmt->getBegLoc(), beg);
    EXPECT_EQ(stmt->getEndLoc(), end);
    EXPECT_EQ(stmt->getSourceRange(), range);
  }
}