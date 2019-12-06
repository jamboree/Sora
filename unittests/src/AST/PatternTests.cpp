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
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
const char *str = "Hello, World!";

class PatternTest : public ::testing::Test {
protected:
  PatternTest() {
    // Setup SourceLocs
    beg = SourceLoc::fromPointer(str);
    mid = SourceLoc::fromPointer(str + 5);
    end = SourceLoc::fromPointer(str + 10);
    // Setup nodes
    varPattern = new (*ctxt) VarPattern(new (*ctxt) VarDecl(nullptr, beg, {}));
    discardPattern = new (*ctxt) DiscardPattern(beg);
    parenPattern =
        new (*ctxt) ParenPattern(beg, new (*ctxt) DiscardPattern(mid), end);
    tuplePattern = TuplePattern::createEmpty(*ctxt, beg, end);
    mutPattern = new (*ctxt) MutPattern(beg, new (*ctxt) DiscardPattern(end));
    typedPattern = new (*ctxt)
        TypedPattern(discardPattern, new (*ctxt) IdentifierTypeRepr(end, {}));
    maybeValuePattern = new (*ctxt) MaybeValuePattern(discardPattern);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, mid, end;

  Pattern *varPattern;
  Pattern *discardPattern;
  Pattern *mutPattern;
  Pattern *parenPattern;
  Pattern *tuplePattern;
  Pattern *typedPattern;
  Pattern *maybeValuePattern;
};
} // namespace

TEST_F(PatternTest, rtti) {
  EXPECT_TRUE(isa<VarPattern>(varPattern));

  EXPECT_TRUE(isa<DiscardPattern>(discardPattern));

  EXPECT_TRUE(isa<MutPattern>(mutPattern));

  EXPECT_TRUE(isa<ParenPattern>(parenPattern));

  EXPECT_TRUE(isa<TuplePattern>(tuplePattern));

  EXPECT_TRUE(isa<TypedPattern>(typedPattern));

  EXPECT_TRUE(isa<MaybeValuePattern>(maybeValuePattern));
  EXPECT_TRUE(isa<RefutablePattern>(maybeValuePattern));
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

  EXPECT_EQ(beg, parenPattern->getBegLoc());
  EXPECT_EQ(mid, parenPattern->getLoc());
  EXPECT_EQ(end, parenPattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), parenPattern->getSourceRange());

  EXPECT_EQ(beg, tuplePattern->getBegLoc());
  EXPECT_EQ(beg, tuplePattern->getLoc());
  EXPECT_EQ(end, tuplePattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), tuplePattern->getSourceRange());

  EXPECT_EQ(beg, typedPattern->getBegLoc());
  EXPECT_EQ(beg, typedPattern->getLoc());
  EXPECT_EQ(end, typedPattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), typedPattern->getSourceRange());

  EXPECT_EQ(beg, maybeValuePattern->getBegLoc());
  EXPECT_EQ(beg, maybeValuePattern->getLoc());
  EXPECT_EQ(beg, maybeValuePattern->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), maybeValuePattern->getSourceRange());
}

TEST_F(PatternTest, isRefutable) {
  // Create a tuple pattern: ((), _, ((), _))
  SmallVector<Pattern *, 4> tupleElts;
  tupleElts.push_back(TuplePattern::createEmpty(*ctxt, {}, {}));
  tupleElts.push_back(new (*ctxt) DiscardPattern({}));
  tupleElts.push_back(TuplePattern::create(*ctxt, {}, tupleElts, {}));
  Pattern *thePattern = TuplePattern::create(*ctxt, {}, tupleElts, {});

  // This pattern shouldn't be considered refutable
  ASSERT_FALSE(thePattern->isRefutable());

  // Now, if we add a MaybeValuePattern somewhere, it should become refutable.
  auto elts = cast<TuplePattern>(thePattern)->getElements();
  elts[1] = new (*ctxt) MaybeValuePattern(elts[1]);

  ASSERT_TRUE(thePattern->isRefutable());
}