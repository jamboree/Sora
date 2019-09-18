//===--- PatternTests.cpp ---------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class PatternTest : public ::testing::Test {
protected:
  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};
};
} // namespace

TEST_F(PatternTest, rtti) {
  // VarPattern
  {
    Pattern *pattern = new (*ctxt) VarPattern(nullptr);
    EXPECT_TRUE(isa<VarPattern>(pattern));
  }

  // DiscardPattern
  {
    Pattern *pattern = new (*ctxt) DiscardPattern(SourceLoc());
    EXPECT_TRUE(isa<DiscardPattern>(pattern));
  }

  // TuplePattern
  {
    Pattern *pattern =
        TuplePattern::createEmpty(*ctxt, SourceLoc(), SourceLoc());
    EXPECT_TRUE(isa<TuplePattern>(pattern));
  }
}

TEST_F(PatternTest, getSourceRange) {
  const char *str = "Hello, World!";
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc mid = SourceLoc::fromPointer(str + 5);
  SourceLoc end = SourceLoc::fromPointer(str + 10);
  SourceRange range(beg, end);

  // VarPattern
  {
    Pattern *pattern = new (*ctxt) VarPattern(new (*ctxt) VarDecl(beg, Identifier()));
    EXPECT_EQ(beg, pattern->getBegLoc());
    EXPECT_EQ(beg, pattern->getLoc());
    EXPECT_EQ(beg, pattern->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), pattern->getSourceRange());
  }

  // DiscardPattern
  {
    Pattern *pattern = new (*ctxt) DiscardPattern(beg);
    EXPECT_EQ(beg, pattern->getBegLoc());
    EXPECT_EQ(beg, pattern->getLoc());
    EXPECT_EQ(beg, pattern->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), pattern->getSourceRange());
  }

  // TuplePattern
  {
    Pattern *pattern = TuplePattern::createEmpty(*ctxt, beg, end);
    EXPECT_EQ(beg, pattern->getBegLoc());
    EXPECT_EQ(beg, pattern->getLoc());
    EXPECT_EQ(end, pattern->getEndLoc());
    EXPECT_EQ(range, pattern->getSourceRange());
  }
}