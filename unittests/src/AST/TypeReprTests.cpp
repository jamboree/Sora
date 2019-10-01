//===--- TypeReprTests.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
const char *str = "Hello, World!";
class TypeReprTest : public ::testing::Test {
protected:
  TypeReprTest() {
    // Setup SourceLocs
    beg = SourceLoc::fromPointer(str);
    mid = SourceLoc::fromPointer(str + 5);
    end = SourceLoc::fromPointer(str + 10);
    // Setup nodes
    identifierTypeRepr = new (*ctxt) IdentifierTypeRepr(beg, {});
    parenTypeRepr = new (*ctxt)
        ParenTypeRepr(beg, new (*ctxt) IdentifierTypeRepr(mid, {}), end);
    tupleTypeRepr = TupleTypeRepr::createEmpty(*ctxt, beg, end);
    arrayTypeRepr = new (*ctxt) ArrayTypeRepr(beg, nullptr, nullptr, end);
    auto subTyRepr = new (*ctxt) IdentifierTypeRepr(end, {});
    referenceTypeRepr = new (*ctxt) ReferenceTypeRepr(beg, subTyRepr);
    maybeTypeRepr = new (*ctxt) MaybeTypeRepr(beg, subTyRepr);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, mid, end;

  TypeRepr *identifierTypeRepr;
  TypeRepr *parenTypeRepr;
  TypeRepr *tupleTypeRepr;
  TypeRepr *arrayTypeRepr;
  TypeRepr *referenceTypeRepr;
  TypeRepr *maybeTypeRepr;
};
} // namespace

TEST_F(TypeReprTest, rtti) {
  EXPECT_TRUE(isa<IdentifierTypeRepr>(identifierTypeRepr));
  EXPECT_TRUE(isa<ParenTypeRepr>(parenTypeRepr));
  EXPECT_TRUE(isa<TupleTypeRepr>(tupleTypeRepr));
  EXPECT_TRUE(isa<ArrayTypeRepr>(arrayTypeRepr));
  EXPECT_TRUE(isa<ReferenceTypeRepr>(referenceTypeRepr));
  EXPECT_TRUE(isa<MaybeTypeRepr>(maybeTypeRepr));
}

TEST_F(TypeReprTest, getSourceRange) {
  SourceLoc beg = SourceLoc::fromPointer(str);
  SourceLoc end = SourceLoc::fromPointer(str + 10);

  // IdentifierTypeRepr
  EXPECT_EQ(beg, identifierTypeRepr->getBegLoc());
  EXPECT_EQ(beg, identifierTypeRepr->getLoc());
  EXPECT_EQ(beg, identifierTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), identifierTypeRepr->getSourceRange());

  // ParenTypeRepr
  EXPECT_EQ(beg, parenTypeRepr->getBegLoc());
  EXPECT_EQ(mid, parenTypeRepr->getLoc());
  EXPECT_EQ(end, parenTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), parenTypeRepr->getSourceRange());

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

  // ReferenceTypeRepr
  EXPECT_EQ(beg, referenceTypeRepr->getBegLoc());
  EXPECT_EQ(beg, referenceTypeRepr->getLoc());
  EXPECT_EQ(end, referenceTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), referenceTypeRepr->getSourceRange());

  // MaybeTypeRepr
  EXPECT_EQ(beg, maybeTypeRepr->getBegLoc());
  EXPECT_EQ(beg, maybeTypeRepr->getLoc());
  EXPECT_EQ(end, maybeTypeRepr->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), maybeTypeRepr->getSourceRange());
}