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
public:
  SourceManager srcMgr;
  DiagnosticEngine diagEngine{srcMgr, llvm::outs()};
  Lexer lexer{srcMgr, diagEngine};

  bool isDone = false;

  void init(StringRef input) {
    lexer.init(srcMgr.giveBuffer(llvm::MemoryBuffer::getMemBuffer(input)));
  }

  /// \returns checks the next token, returns true on success and false
  /// on failure.
  bool checkNext(llvm::raw_string_ostream &out, TokenKind kind, StringRef str,
                 bool startOfLine = false) {
    Token token = lexer.lex();
    isDone = token.getKind() == TokenKind::EndOfFile;
    if (token.getKind() != kind) {
      out << "mismatched kinds: expected '" << to_string(kind) << "' got '"
          << to_string(token.getKind()) << "'";
      return false;
    }
    if (token.isStartOfLine() != startOfLine) {
      out << "token not at start of line";
      return false;
    }
    if (token.str() != str) {
      out << "mismatched token content: expected '" << str << "' got '"
          << token.str() << "'";
      return false;
    }
    return true;
  }
};
} // namespace

TEST_F(LexerTest, keywordAndIdentifiers) {
  const char *input = "foo\n"
                      "break continue\n"
                      "else false\n"
                      "for func if in let\n"
                      "maybe mut null\n"
                      "return struct true type\n"
                      "_ while";
  init(input);
  std::string failureReason;
  llvm::raw_string_ostream errStream(failureReason);
#define CHECK(KIND, STR, SOL)                                                  \
  EXPECT_TRUE(checkNext(errStream, TokenKind::KIND, STR, SOL))                 \
      << errStream.str();                                                      \
  failureReason.clear();                                                       \
  ASSERT_FALSE(isDone)
  // Test
  CHECK(Identifier, "foo", true);
  CHECK(BreakKw, "break", true);
  CHECK(ContinueKw, "continue", false);
  CHECK(ElseKw, "else", true);
  CHECK(FalseKw, "false", false);
  CHECK(ForKw, "for", true);
  CHECK(FuncKw, "func", false);
  CHECK(IfKw, "if", false);
  CHECK(InKw, "in", false);
  CHECK(LetKw, "let", false);
  CHECK(MaybeKw, "maybe", true);
  CHECK(MutKw, "mut", false);
  CHECK(NullKw, "null", false);
  CHECK(ReturnKw, "return", true);
  CHECK(StructKw, "struct", false);
  CHECK(TrueKw, "true", false);
  CHECK(TypeKw, "type", false);
  CHECK(UnderscoreKw, "_", true);
  CHECK(WhileKw, "while", false);
#undef CHECK
}
