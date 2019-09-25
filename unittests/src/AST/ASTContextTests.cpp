//===--- ASTContextTests.cpp ------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class ASTContextTest : public ::testing::Test {
public:
  SourceManager srcMgr;
  DiagnosticEngine diagEngine{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEngine)};
};
} // namespace

TEST_F(ASTContextTest, getIdentifier_normalStrings) {
#define TEST_IDENTIFIER_INTERNING(STR)                                         \
  EXPECT_EQ(ctxt->getIdentifier(STR).c_str(), ctxt->getIdentifier(STR).c_str())
  TEST_IDENTIFIER_INTERNING("foo");
  TEST_IDENTIFIER_INTERNING("0");
  TEST_IDENTIFIER_INTERNING(" ");
  TEST_IDENTIFIER_INTERNING("_Hello");
  TEST_IDENTIFIER_INTERNING(u8"おやすみなさい");
  TEST_IDENTIFIER_INTERNING("abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcab"
                            "cabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabca"
                            "bcabcabcabcabcabcabcabcabcabcabcabcabcabcabc");
#undef TEST_IDENTIFIER_INTERNING
}

TEST_F(ASTContextTest, getIdentifier_emptyAndNullStrings) {
  EXPECT_EQ(ctxt->getIdentifier(StringRef()).c_str(), nullptr);
  EXPECT_EQ(ctxt->getIdentifier(StringRef("")).c_str(), nullptr);
}

TEST_F(ASTContextTest, cleanup) {
  bool cleanupRan = false;
  ctxt->addCleanup([&]() { cleanupRan = true; });

  bool dtorRan = false;
  struct Foo {
    bool &val;
    Foo(bool &val) : val(val) {}
    ~Foo() { val = true; }
  };
  Foo foo(dtorRan);
  ctxt->addDestructorCleanup(foo);

  ctxt.reset();

  EXPECT_TRUE(cleanupRan) << "Cleanup did not run";
  EXPECT_TRUE(dtorRan) << "Destructor cleanup did not run";
}