//===--- Lexer.hpp - Lexical Analysis ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Main Interface of the Sora Lexer.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "Sora/Lexer/Token.hpp"
#include "llvm/ADT/Optional.h"

namespace sora {
class SourceManager;
class DiagnosticEngine;
class BufferID;

/// The Sora Lexer, which produces tokens from an input buffer.
///
/// This is the very first step of the compilation process. Tokens are
/// consumed by the Parser to produce the AST.
///
/// However, please note that the Sora lexer is separate from the Parser
/// because the Lexer is also the tool that allows you to expand a SourceLoc
/// or a SourceRange into a CharSourceRange (and you shouldn't have
/// to depend on the Parser to do that - hence the Parser/Lexer separation)
///
/// As an implementation detail, the Lexer always operates one token in advance:
/// lex() returns a pre-lexed token and prepares (lexs) the next token.
/// This allows us to have an efficient "peek" method.
class Lexer {
public:
  Lexer(SourceManager &srcMgr, DiagnosticEngine &diagEng)
      : srcMgr(srcMgr), diagEng(diagEng) {}

  /// Prepares the Lexer to lex the string \p str
  void init(StringRef str);
  /// Prepares the Lexer to lex the buffer \p id
  void init(BufferID id);

  /// Lex a token and return it.
  /// If we reached EOF, this will simply return the EOF token whenever
  /// it is called.
  Token lex();

  /// \returns the next token to be returned by "lex" without consuming
  /// it or changing the state of the lexer.
  Token peek() const;

  /// The SourceManager instance
  SourceManager &srcMgr;
  /// The DiagnosticEngine instance
  DiagnosticEngine &diagEng;

private:
  /// Sentinel value for invalid UTF8 codepoints
  static constexpr uint32_t invalidCP = ~0U;

  /// Returns the next valid UTF8 codepoint and moves the 'cur' iterator past
  /// the end of that codepoint.
  /// \returns the codepoint or "invalidCP" if an error occured
  /// (note: errors are diagnosed directly, no need to diagnose them again)
  uint32_t advance();

  /// Peeks the next character without consuming it.
  ///
  /// \returns the codepoint or ~0U if an error occured
  uint32_t peekChar() const;

  /// Finishes lexing (sets nextToken = EOF and cur = end)
  void stopLexing();

  /// Performs the actual lexing.
  void doLex();
  /// The beginning of the current token
  const char *tokBeg = nullptr;
  /// The current iterator into the file
  const char *cur = nullptr;
  /// The past-the-end iterator of the file
  const char *end = nullptr;
  /// The next position of the "cur" pointer
  const char *nextCur = nullptr;
  /// The next codepoint that'll be returned by advance()
  uint32_t nextCP = ~0U;
  /// The next token that'll be returned.
  Token nextToken;
};
} // namespace sora