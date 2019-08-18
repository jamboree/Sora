//===--- SourceTests.cpp ----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Common/SourceManager.hpp"

#include "gtest/gtest.h"

using namespace sora;

/// Tests that a default-constructed BufferID is considered null.
TEST(BufferID, isNull) { EXPECT_TRUE(BufferID().isNull()); }

/// Tests that a default-constructed SourceLoc is considered invalid.
TEST(SourceLoc, isValid) {
  EXPECT_TRUE(SourceLoc().isInvalid());
  EXPECT_FALSE(SourceLoc().isValid());
}

/// Test for operator== and operator!= for SourceLoc
TEST(SourceLoc, compare) {
  const char *str = "ab";

  auto a = SourceLoc::fromPointer(str);
  auto b = SourceLoc::fromPointer(str + 1);

  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);

  EXPECT_NE(a, b);
  EXPECT_NE(b, a);
}

/// Test that a default-constructed SourceRange is considered invalid.
TEST(SourceRange, isValid) {
  EXPECT_TRUE(SourceRange().isInvalid());
  EXPECT_FALSE(SourceRange().isValid());
}

/// Test for operator== and operator!= for SourceRange
TEST(SourceRange, compare) {
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
TEST(CharSourceRange, emptyAndInvalid) {
  EXPECT_TRUE(CharSourceRange().isInvalid());
  EXPECT_TRUE(CharSourceRange().empty());
  EXPECT_FALSE(CharSourceRange().isValid());
}

/// Test for operator== and operator!= for CharSourceRange
TEST(CharSourceRange, compare) {
  const char *str = "ab";

  auto aRange = CharSourceRange::fromPointers(str, str+1);
  auto bRange = CharSourceRange::fromPointers(str+1, str+2);

  EXPECT_EQ(aRange, aRange);
  EXPECT_EQ(bRange, bRange);

  EXPECT_NE(aRange, bRange);
  EXPECT_NE(bRange, aRange);
}

/// Test for CharSourceRange::str
TEST(CharSourceRange, str) {
  const char *str = "ab";

  auto aRange = CharSourceRange::fromPointers(str, str+1);
  auto bRange = CharSourceRange::fromPointers(str+1, str+2);
  auto completeRange = CharSourceRange::fromPointers(str, str+2);

  EXPECT_TRUE(aRange.str() == "a");
  EXPECT_TRUE(bRange.str() == "b");
  EXPECT_TRUE(completeRange.str() == "ab");
}
