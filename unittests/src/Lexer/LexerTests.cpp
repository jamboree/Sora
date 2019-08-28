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

  void
  installForwardingHandler(ForwardingDiagnosticConsumer::HandlerFunction func) {
    diagEngine.replaceConsumer(
        std::make_unique<ForwardingDiagnosticConsumer>(func));
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

/// Test for invalid/illegal UTF-8 codepoint (ill-formed codepoints)
TEST_F(LexerTest, invalidUTF8) {
  const char *input = "\xa0\xa1";

  unsigned diagCount = 0;
  std::string diagMsg;
  SourceLoc diagLoc;

  installForwardingHandler([&](SourceManager &srcMgr, const Diagnostic &diag) {
    diagMsg = diag.message;
    diagLoc = diag.loc;
    ++diagCount;
  });

  init(input);
  CHECK_NEXT(TokenKind::Unknown, "\xa0\xa1", true);
  CHECK_EOF();

  EXPECT_EQ(diagMsg, "illegal utf-8 codepoint");
  EXPECT_EQ(diagLoc, SourceLoc::fromPointer(input));
  EXPECT_EQ(diagCount, 1);
}

/// Test for unfinished UTF-8 codepoint (found EOF before rest of codepoint)
TEST_F(LexerTest, incompleteUTF8) {
  const char *input = "\xc3";

  unsigned diagCount = 0;
  std::string diagMsg;
  SourceLoc diagLoc;

  installForwardingHandler([&](SourceManager &srcMgr, const Diagnostic &diag) {
    diagMsg = diag.message;
    diagLoc = diag.loc;
    ++diagCount;
  });

  init(input);
  CHECK_NEXT(TokenKind::Unknown, "\xc3", true);
  CHECK_EOF();

  EXPECT_EQ(diagMsg, "incomplete utf-8 codepoint");
  EXPECT_EQ(diagLoc, SourceLoc::fromPointer(input));
  EXPECT_EQ(diagCount, 1);
}

TEST_F(LexerTest, punctuationAndOperators) {
  const char *input =
      "()[]{}/=/++=--=&&&=&**=%%=|||=|>>>=>>=><<<=<<=<:,.!!=^^=->";

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
  CHECK_NEXT(TokenKind::Star, "*", false);
  CHECK_NEXT(TokenKind::StarEqual, "*=", false);
  CHECK_NEXT(TokenKind::Percent, "%", false);
  CHECK_NEXT(TokenKind::PercentEqual, "%=", false);
  CHECK_NEXT(TokenKind::PipePipe, "||", false);
  CHECK_NEXT(TokenKind::PipeEqual, "|=", false);
  CHECK_NEXT(TokenKind::Pipe, "|", false);
  CHECK_NEXT(TokenKind::GreaterGreater, ">>", false);
  CHECK_NEXT(TokenKind::GreaterEqual, ">=", false);
  CHECK_NEXT(TokenKind::GreaterGreaterEqual, ">>=", false);
  CHECK_NEXT(TokenKind::Greater, ">", false);
  CHECK_NEXT(TokenKind::LessLess, "<<", false);
  CHECK_NEXT(TokenKind::LessEqual, "<=", false);
  CHECK_NEXT(TokenKind::LessLessEqual, "<<=", false);
  CHECK_NEXT(TokenKind::Less, "<", false);
  CHECK_NEXT(TokenKind::Colon, ":", false);
  CHECK_NEXT(TokenKind::Comma, ",", false);
  CHECK_NEXT(TokenKind::Dot, ".", false);
  CHECK_NEXT(TokenKind::Exclaim, "!", false);
  CHECK_NEXT(TokenKind::ExclaimEqual, "!=", false);
  CHECK_NEXT(TokenKind::Caret, "^", false);
  CHECK_NEXT(TokenKind::CaretEqual, "^=", false);
  CHECK_NEXT(TokenKind::Arrow, "->", false);
  CHECK_EOF();
}

TEST_F(LexerTest, numbers) {
  const char *input = "0 1 2 3 4 5 6 7 8 9\n"
                      "0.0 1.1 2.2 3.3 4.4 5.5 6.6 7.7 8.8 9.9\n"
                      "0.0.0 9999999999\n"
                      "123456.123456";

  init(input);
#define CHECK_NEXT_INT(STR, SOL) CHECK_NEXT(TokenKind::IntegerLiteral, STR, SOL)
#define CHECK_NEXT_FLT(STR, SOL)                                               \
  CHECK_NEXT(TokenKind::FloatingPointLiteral, STR, SOL)

  CHECK_NEXT_INT("0", true);
  CHECK_NEXT_INT("1", false);
  CHECK_NEXT_INT("2", false);
  CHECK_NEXT_INT("3", false);
  CHECK_NEXT_INT("4", false);
  CHECK_NEXT_INT("5", false);
  CHECK_NEXT_INT("6", false);
  CHECK_NEXT_INT("7", false);
  CHECK_NEXT_INT("8", false);
  CHECK_NEXT_INT("9", false);

  CHECK_NEXT_FLT("0.0", true);
  CHECK_NEXT_FLT("1.1", false);
  CHECK_NEXT_FLT("2.2", false);
  CHECK_NEXT_FLT("3.3", false);
  CHECK_NEXT_FLT("4.4", false);
  CHECK_NEXT_FLT("5.5", false);
  CHECK_NEXT_FLT("6.6", false);
  CHECK_NEXT_FLT("7.7", false);
  CHECK_NEXT_FLT("8.8", false);
  CHECK_NEXT_FLT("9.9", false);

  CHECK_NEXT_FLT("0.0", true);
  CHECK_NEXT(TokenKind::Dot, ".", false);
  CHECK_NEXT_INT("0", false);
  CHECK_NEXT_INT("9999999999", false);

  CHECK_NEXT_FLT("123456.123456", true);

#undef CHECK_NEXT_INT
#undef CHECK_NEXT_FLT
  CHECK_EOF();
}
