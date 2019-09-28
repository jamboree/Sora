//===--- PatternTests.cpp ---------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
const char *str = "Hello, World!";

class PatternTest : public ::testing::Test {
protected:
  PatternTest() {
    // Setup SourceLocs
    beg = SourceLoc::fromPointer(str);
    end = SourceLoc::fromPointer(str + 10);
    // Setup nodes
    varPattern = new (*ctxt) VarPattern(new (*ctxt) VarDecl(nullptr, beg, {}));
    discardPattern = new (*ctxt) DiscardPattern(beg);
    tuplePattern = TuplePattern::createEmpty(*ctxt, beg, end);
    mutPattern = new (*ctxt) MutPattern(beg, new (*ctxt) DiscardPattern(end));
    typedPattern =
        new (*ctxt) TypedPattern(new (*ctxt) DiscardPattern(beg),
                                 new (*ctxt) IdentifierTypeRepr(end, {}));
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, end;

  Pattern *varPattern;
  Pattern *discardPattern;
  Pattern *mutPattern;
  Pattern *tuplePattern;
  Pattern *typedPattern;
};
} // namespace

TEST_F(PatternTest, rtti) {
  EXPECT_TRUE(isa<VarPattern>(varPattern));
  EXPECT_TRUE(isa<DiscardPattern>(discardPattern));
  EXPECT_TRUE(isa<MutPattern>(mutPattern));
  EXPECT_TRUE(isa<TuplePattern>(tuplePattern));
  EXPECT_TRUE(isa<TypedPattern>(typedPattern));
}

TEST_F(PatternTest, getSourceRange) {
  EXPECT_EQ(beg, varPattern->getBegLoc());
  EXPECT_EQ(beg, varPattern->getLoc());
  EXPECT_EQ(beg, varPattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), varPattern->getSourceRange());

  EXPECT_EQ(beg, discardPattern->getBegLoc());
  EXPECT_EQ(beg, discardPattern->getLoc());
  EXPECT_EQ(beg, discardPattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), discardPattern->getSourceRange());

  EXPECT_EQ(beg, mutPattern->getBegLoc());
  EXPECT_EQ(end, mutPattern->getLoc());
  EXPECT_EQ(end, mutPattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), mutPattern->getSourceRange());

  EXPECT_EQ(beg, tuplePattern->getBegLoc());
  EXPECT_EQ(beg, tuplePattern->getLoc());
  EXPECT_EQ(end, tuplePattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), tuplePattern->getSourceRange());

  EXPECT_EQ(beg, typedPattern->getBegLoc());
  EXPECT_EQ(beg, typedPattern->getLoc());
  EXPECT_EQ(end, typedPattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), typedPattern->getSourceRange());
}