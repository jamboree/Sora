//===--- LexerTests.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Lexer/Lexer.hpp"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace sora;

namespace {
class LexerTest : public ::testing::Test {
  std::string errStreamBuff;

public:
  SourceManager srcMgr;
  DiagnosticEngine diagEngine{srcMgr, llvm::outs()};
  Lexer lexer{srcMgr, diagEngine};
  llvm::raw_string_ostream errStream{errStreamBuff};

  bool isDone = false;

  void init(StringRef input) {
    lexer.init(srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(input)));
  }

  /// \returns checks the next token, returns true on success and false
  /// on failure. In case of failure, the message can be found in
  /// "errStream.str()"
  bool checkNext(TokenKind kind, StringRef str, bool startOfLine = false) {
    errStreamBuff.clear();
    Token token = lexer.lex();
    isDone = token.getKind() == TokenKind::EndOfFile;
    if (token.getKind() != kind) {
      errStream << "mismatched kinds: expected '" << to_string(kind)
                << "' got '" << to_string(token.getKind()) << "'";
      return false;
    }
    if (token.isStartOfLine() != startOfLine) {
      errStream << "token not at start of line";
      return false;
    }
    if (token.str() != str) {
      errStream << "mismatched token content: expected '" << str << "' got '"
                << token.str() << "'";
      return false;
    }
    return true;
  }
};
} // namespace

#define CHECK_NEXT(KIND, STR, SOL)                                             \
  EXPECT_TRUE(checkNext(KIND, STR, SOL)) << errStream.str();                   \
  ASSERT_FALSE(isDone)
#define CHECK_EOF()                                                            \
  EXPECT_TRUE(checkNext(TokenKind::EndOfFile, "", false)) << errStream.str()

TEST_F(LexerTest, keywordCommentsAndIdentifiers) {
  const char *input = "foo // this is a comment\n"
                      "break /* another comment*/ continue\n"
                      "else false\n"
                      "for func if in let\n"
                      "maybe mut null\n"
                      "return struct true type\n"
                      "_ while //goodbye";
  init(input);
  // Test
  CHECK_NEXT(TokenKind::Identifier, "foo", true);
  CHECK_NEXT(TokenKind::BreakKw, "break", true);
  CHECK_NEXT(TokenKind::ContinueKw, "continue", false);
  CHECK_NEXT(TokenKind::ElseKw, "else", true);
  CHECK_NEXT(TokenKind::FalseKw, "false", false);
  CHECK_NEXT(TokenKind::ForKw, "for", true);
  CHECK_NEXT(TokenKind::FuncKw, "func", false);
  CHECK_NEXT(TokenKind::IfKw, "if", false);
  CHECK_NEXT(TokenKind::InKw, "in", false);
  CHECK_NEXT(TokenKind::LetKw, "let", false);
  CHECK_NEXT(TokenKind::MaybeKw, "maybe", true);
  CHECK_NEXT(TokenKind::MutKw, "mut", false);
  CHECK_NEXT(TokenKind::NullKw, "null", false);
  CHECK_NEXT(TokenKind::ReturnKw, "return", true);
  CHECK_NEXT(TokenKind::StructKw, "struct", false);
  CHECK_NEXT(TokenKind::TrueKw, "true", false);
  CHECK_NEXT(TokenKind::TypeKw, "type", false);
  CHECK_NEXT(TokenKind::UnderscoreKw, "_", true);
  CHECK_NEXT(TokenKind::WhileKw, "while", false);
  CHECK_EOF();
}

TEST_F(LexerTest, unknownTokens) {
  const char *input = u8"ê€";

  init(input);
  CHECK_NEXT(TokenKind::Unknown, u8"ê", true);
  CHECK_NEXT(TokenKind::Unknown, u8"€", false);
  CHECK_EOF();
}

TEST_F(LexerTest, operators) {
  const char *input = "()[]{}/=/++=--=&&&=&;";

  init(input);
  CHECK_NEXT(TokenKind::LParen, "(", true);
  CHECK_NEXT(TokenKind::RParen, ")", false);
  CHECK_NEXT(TokenKind::LSquare, "[", false);
  CHECK_NEXT(TokenKind::RSquare, "]", false);
  CHECK_NEXT(TokenKind::LCurly, "{", false);
  CHECK_NEXT(TokenKind::RCurly, "}", false);
  CHECK_NEXT(TokenKind::SlashEqual, "/=", false);
  CHECK_NEXT(TokenKind::Slash, "/", false);
  CHECK_NEXT(TokenKind::Plus, "+", false);
  CHECK_NEXT(TokenKind::PlusEqual, "+=", false);
  CHECK_NEXT(TokenKind::Minus, "-", false);
  CHECK_NEXT(TokenKind::MinusEqual, "-=", false);
  CHECK_NEXT(TokenKind::AmpAmp, "&&", false);
  CHECK_NEXT(TokenKind::AmpEqual, "&=", false);
  CHECK_NEXT(TokenKind::Amp, "&", false);
  CHECK_NEXT(TokenKind::Semi, ";", false);
  CHECK_EOF();
}
