//===--- TypeReprTests.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class TypeReprTest : public ::testing::Test {
protected:
  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};
};
} // namespace

TEST_F(TypeReprTest, rtti) {
  // IdentifierTypeRepr
  {
    TypeRepr *tyRepr =
        new (*ctxt) IdentifierTypeRepr(SourceLoc(), Identifier());
    EXPECT_TRUE(isa<IdentifierTypeRepr>(tyRepr));
  }

  // TupleTypeRepr
  {
    TypeRepr *tyRepr =
        TupleTypeRepr::createEmpty(*ctxt, SourceLoc(), SourceLoc());
    EXPECT_TRUE(isa<TupleTypeRepr>(tyRepr));
  }
}

TEST_F(TypeReprTest, getSourceRange) {
  const char *str = "Hello, World!";
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc end = SourceLoc::fromPointer(str + 10);
  SourceRange range(beg, end);

  // IdentifierTypeRepr
  {
    TypeRepr *tyRepr = new (*ctxt) IdentifierTypeRepr(beg, Identifier());
    EXPECT_EQ(beg, tyRepr->getBegLoc());
    EXPECT_EQ(beg, tyRepr->getLoc());
    EXPECT_EQ(beg, tyRepr->getEndLoc());
    EXPECT_EQ(SourceRange(beg, beg), tyRepr->getSourceRange());
  }

  // TupleTypeRepr
  {
    TypeRepr *tyRepr = TupleTypeRepr::createEmpty(*ctxt, beg, end);
    EXPECT_EQ(beg, tyRepr->getBegLoc());
    EXPECT_EQ(beg, tyRepr->getLoc());
    EXPECT_EQ(end, tyRepr->getEndLoc());
    EXPECT_EQ(range, tyRepr->getSourceRange());
  }
}