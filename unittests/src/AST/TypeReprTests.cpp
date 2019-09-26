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
const char *str = "Hello, World!";
class TypeReprTest : public ::testing::Test {
protected:
  TypeReprTest() {
    // Setup SourceLocs
    SourceLoc beg = SourceLoc::fromPointer(str);
    SourceLoc end = SourceLoc::fromPointer(str + 10);
    // Setup nodes
    identifierTypeRepr = new (*ctxt) IdentifierTypeRepr(beg, {});
    tupleTypeRepr = TupleTypeRepr::createEmpty(*ctxt, beg, end);
    arrayTypeRepr = new (*ctxt) ArrayTypeRepr(beg, nullptr, nullptr, end);
    pointerTypeRepr = new (*ctxt)
        PointerTypeRepr(beg, true, new (*ctxt) IdentifierTypeRepr(end, {}));
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, end;

  TypeRepr *identifierTypeRepr;
  TypeRepr *tupleTypeRepr;
  TypeRepr *arrayTypeRepr;
  TypeRepr *pointerTypeRepr;
};
} // namespace

TEST_F(TypeReprTest, rtti) {
  EXPECT_TRUE(isa<IdentifierTypeRepr>(identifierTypeRepr));
  EXPECT_TRUE(isa<TupleTypeRepr>(tupleTypeRepr));
  EXPECT_TRUE(isa<ArrayTypeRepr>(arrayTypeRepr));
  EXPECT_TRUE(isa<PointerTypeRepr>(pointerTypeRepr));
}

TEST_F(TypeReprTest, getSourceRange) {
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc end = SourceLoc::fromPointer(str + 10);

  // IdentifierTypeRepr
  EXPECT_EQ(beg, identifierTypeRepr->getBegLoc());
  EXPECT_EQ(beg, identifierTypeRepr->getLoc());
  EXPECT_EQ(beg, identifierTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), identifierTypeRepr->getSourceRange());

  // TupleTypeRepr
  EXPECT_EQ(beg, tupleTypeRepr->getBegLoc());
  EXPECT_EQ(beg, tupleTypeRepr->getLoc());
  EXPECT_EQ(end, tupleTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), tupleTypeRepr->getSourceRange());

  // ArrayTypeRepr
  EXPECT_EQ(beg, arrayTypeRepr->getBegLoc());
  EXPECT_EQ(beg, arrayTypeRepr->getLoc());
  EXPECT_EQ(end, arrayTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), arrayTypeRepr->getSourceRange());

  // PointerTypeRepr
  EXPECT_EQ(beg, pointerTypeRepr->getBegLoc());
  EXPECT_EQ(beg, pointerTypeRepr->getLoc());
  EXPECT_EQ(end, pointerTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), pointerTypeRepr->getSourceRange());
}