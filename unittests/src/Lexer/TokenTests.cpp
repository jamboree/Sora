//===--- TokenTests.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Lexer/Token.hpp"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace sora;

TEST(TokenTest, invalidKind) {
  Token tok;
  EXPECT_FALSE(tok);
  EXPECT_TRUE(tok.is(TokenKind::Invalid))
      << "Default token doesn't have the invalid kind";
  EXPECT_FALSE(tok.isNot(TokenKind::Invalid));
}

TEST(TokenTest, is_isNot) {
  Token tok(TokenKind::Amp, CharSourceRange());
  EXPECT_TRUE(tok.is(TokenKind::Amp));
  EXPECT_TRUE(tok.isNot(TokenKind::AmpAmp));
  EXPECT_FALSE(tok.isNot(TokenKind::Amp));
}

TEST(TokenTest, startOfLine) {
  EXPECT_TRUE(Token(TokenKind::Amp, CharSourceRange(), true).isAtStartOfLine());
}

TEST(TokenTest, str) {
  const char *str = "Hello World";
  Token a(TokenKind::Amp, CharSourceRange::fromPointers(str, str + 5), false);
  Token b(TokenKind::LetKw, CharSourceRange::fromPointers(str + 6, str + 11),
          true);
  EXPECT_EQ(a.str(), "Hello");
  EXPECT_EQ(b.str(), "World");
}

TEST(TokenTest, dump) {
  const char *str = "Hello World";
  Token a(TokenKind::Amp, CharSourceRange::fromPointers(str, str + 5), false);
  Token b(TokenKind::LetKw, CharSourceRange::fromPointers(str + 6, str + 11),
          true);

  std::string out;
  llvm::raw_string_ostream ss(out);
  a.dump(ss);
  ss.str();
  EXPECT_EQ(out, "'Hello' - Amp");

  out.clear();

  b.dump(ss);
  ss.str();
  EXPECT_EQ(out, "'World' - LetKw [startOfLine]");
}
