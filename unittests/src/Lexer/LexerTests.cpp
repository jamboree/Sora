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
  EXPECT_TRUE(checkNext(TokenKind::KIND, STR, SOL)) << errStream.str();        \
  ASSERT_FALSE(isDone)
#define CHECK_LAST(KIND, STR, SOL)                                             \
  EXPECT_TRUE(checkNext(TokenKind::KIND, STR, SOL)) << errStream.str()

TEST_F(LexerTest, keywordAndIdentifiers) {
  const char *input = "foo\n"
                      "break continue\n"
                      "else false\n"
                      "for func if in let\n"
                      "maybe mut null\n"
                      "return struct true type\n"
                      "_ while";
  init(input);
  // Test
  CHECK_NEXT(Identifier, "foo", true);
  CHECK_NEXT(BreakKw, "break", true);
  CHECK_NEXT(ContinueKw, "continue", false);
  CHECK_NEXT(ElseKw, "else", true);
  CHECK_NEXT(FalseKw, "false", false);
  CHECK_NEXT(ForKw, "for", true);
  CHECK_NEXT(FuncKw, "func", false);
  CHECK_NEXT(IfKw, "if", false);
  CHECK_NEXT(InKw, "in", false);
  CHECK_NEXT(LetKw, "let", false);
  CHECK_NEXT(MaybeKw, "maybe", true);
  CHECK_NEXT(MutKw, "mut", false);
  CHECK_NEXT(NullKw, "null", false);
  CHECK_NEXT(ReturnKw, "return", true);
  CHECK_NEXT(StructKw, "struct", false);
  CHECK_NEXT(TrueKw, "true", false);
  CHECK_NEXT(TypeKw, "type", false);
  CHECK_NEXT(UnderscoreKw, "_", true);
  CHECK_NEXT(WhileKw, "while", false);
  CHECK_LAST(EndOfFile, "", false);
}

TEST_F(LexerTest, unknownTokens) {
  const char *input = u8"ê€";

  init(input);
  CHECK_NEXT(Unknown, u8"ê", true);
  CHECK_NEXT(Unknown, u8"€", false);
  CHECK_LAST(EndOfFile, "", false);
}