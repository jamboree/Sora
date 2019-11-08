//===--- ASTContextTests.cpp ------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class ASTContextTest : public ::testing::Test {
public:
  SourceManager srcMgr;
  DiagnosticEngine diagEngine{srcMgr};
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

TEST_F(ASTContextTest, lookupBuiltinType) {
#define CHECK(STR, TY)                                                         \
  EXPECT_EQ(ctxt->lookupBuiltinType(ctxt->getIdentifier(STR)).getPtr(),          \
            ctxt->TY.getPtr())

  CHECK("i8", i8Type);
  CHECK("i16", i16Type);
  CHECK("i32", i32Type);
  CHECK("i64", i64Type);
  CHECK("isize", isizeType);
  CHECK("u8", u8Type);
  CHECK("u16", u16Type);
  CHECK("u32", u32Type);
  CHECK("u64", u64Type);
  CHECK("usize", usizeType);
  CHECK("f32", f32Type);
  CHECK("f64", f64Type);
  CHECK("void", voidType);
  CHECK("bool", boolType);

#undef CHECK

#define CHECK_NULL(STR)                                                        \
  EXPECT_EQ(ctxt->lookupBuiltinType(ctxt->getIdentifier(STR)).getPtr(), nullptr)
  CHECK_NULL("i");
  CHECK_NULL("viod");
  CHECK_NULL("i33");
  CHECK_NULL("i31");
  CHECK_NULL("u31");
  CHECK_NULL("u42");
  CHECK_NULL("f54");
  CHECK_NULL("baal");
#undef CHECK_NULL
}

TEST_F(ASTContextTest, getAllBuiltinTypes) {
  // TODO
}