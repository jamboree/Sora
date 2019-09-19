//===--- SourceTests.cpp ----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace sora;

/// Tests that a default-constructed BufferID is considered null.
TEST(BufferIDTest, isNull) { EXPECT_TRUE(BufferID().isNull()); }

/// Tests that a default-constructed SourceLoc is considered invalid.
TEST(SourceLocTest, isValid) {
  EXPECT_TRUE(SourceLoc().isInvalid());
  EXPECT_FALSE(SourceLoc().isValid());
}

/// Tests that a default-constructed SourceLoc is considered invalid.
TEST(SourceLocTest, print) {
  SourceManager srcMgr;
  {
    const char *str = "a";
    srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(str));
    std::string output;
    llvm::raw_string_ostream rso(output);
    SourceLoc::fromPointer(str).print(rso, srcMgr, true);
    EXPECT_EQ(rso.str(), "<unknown>:1:1");
  }
  {
    const char *str = "b";
    srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(str, "Test"));
    std::string output;
    llvm::raw_string_ostream rso(output);
    SourceLoc::fromPointer(str).print(rso, srcMgr, true);
    EXPECT_EQ(rso.str(), "Test:1:1");
  }
  {
    const char *str = "c";
    srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(str));
    std::string output;
    llvm::raw_string_ostream rso(output);
    SourceLoc::fromPointer(str).print(rso, srcMgr, false);
    EXPECT_EQ(rso.str(), "line:1:1");
  }
  {
    std::string output;
    llvm::raw_string_ostream rso(output);
    SourceLoc().print(rso, srcMgr, false);
    EXPECT_EQ(rso.str(), "<invalid>");
  }
}

/// Test for operator== and operator!= for SourceLoc
TEST(SourceLocTest, compare) {
  const char *str = "ab";

  auto a = SourceLoc::fromPointer(str);
  auto b = SourceLoc::fromPointer(str + 1);

  EXPECT_TRUE(a == a);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(a <= b);
  EXPECT_TRUE(b >= a);
  EXPECT_TRUE(a <= a);
  EXPECT_TRUE(a >= a);
}

/// Test that a default-constructed SourceRange is considered invalid.
TEST(SourceRangeTest, isValid) {
  EXPECT_TRUE(SourceRange().isInvalid());
  EXPECT_FALSE(SourceRange().isValid());
}

TEST(SourceRangeTest, print) {
  SourceManager srcMgr;

  const char *str = "a b";
  srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(str));
  std::string output;
  llvm::raw_string_ostream rso(output);
  SourceRange range(SourceLoc::fromPointer(str),
                    SourceLoc::fromPointer(str + 2));
  range.print(rso, srcMgr, true);
  EXPECT_EQ(rso.str(), "[<unknown>:1:1, line:1:3]");

  output.clear();
  range.print(rso, srcMgr, false);
  EXPECT_EQ(rso.str(), "[line:1:1, line:1:3]");
}

/// Test for operator== and operator!= for SourceRange
TEST(SourceRangeTest, compare) {
  const char *str = "ab";

  auto aRange = SourceRange(SourceLoc::fromPointer(str));
  auto bRange = SourceRange(SourceLoc::fromPointer(str + 1));

  EXPECT_EQ(aRange, aRange);
  EXPECT_EQ(bRange, bRange);

  EXPECT_NE(aRange, bRange);
  EXPECT_NE(bRange, aRange);
}

/// Test for .isValid(), .isInvalid() and .empty() for default-constructed
/// CharSourceRanges.
TEST(CharSourceRangeTest, emptyAndInvalid) {
  EXPECT_TRUE(CharSourceRange().isInvalid());
  EXPECT_TRUE(CharSourceRange().empty());
  EXPECT_FALSE(CharSourceRange().isValid());
}

/// Test for operator== and operator!= for CharSourceRange
TEST(CharSourceRangeTest, compare) {
  const char *str = "ab";

  auto aRange = CharSourceRange::fromPointers(str, str + 1);
  auto bRange = CharSourceRange::fromPointers(str + 1, str + 2);

  EXPECT_EQ(aRange, aRange);
  EXPECT_EQ(bRange, bRange);

  EXPECT_NE(aRange, bRange);
  EXPECT_NE(bRange, aRange);
}

/// Test for CharSourceRange::str
TEST(CharSourceRangeTest, str) {
  const char *str = "ab";

  auto aRange = CharSourceRange::fromPointers(str, str + 1);
  auto bRange = CharSourceRange::fromPointers(str + 1, str + 2);
  auto completeRange = CharSourceRange::fromPointers(str, str + 2);

  EXPECT_TRUE(aRange.str() == "a");
  EXPECT_TRUE(bRange.str() == "b");
  EXPECT_TRUE(completeRange.str() == "ab");
}

TEST(CharSourceRange, print) {
  SourceManager srcMgr;

  const char *str = "Hello, World!";
  srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(str, "X"));
  std::string output;
  llvm::raw_string_ostream rso(output);
  CharSourceRange range(SourceLoc::fromPointer(str), strlen(str));
  range.print(rso, srcMgr, true, false);
  EXPECT_EQ(rso.str(), "[X:1:1, line:1:14)");

  output.clear();
  range.print(rso, srcMgr, true, true);
  EXPECT_EQ(rso.str(), "[X:1:1, line:1:14)=\"Hello, World!\"");
}