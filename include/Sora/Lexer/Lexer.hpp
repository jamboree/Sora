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
  /// Finishes lexing (sets nextToken = EOF and cur = end)
  void stopLexing();

  /// Lexs the body of an identifier which starts at tokBegPtr.
  /// curPtr must point past-the-end of the head of the identifier.
  void lexIdentifierBody();

  /// Lexs an unknown (possibly utf-8) character sequence that starts at
  /// tokBegPtr and moves curPtr past-the-end of the sequence (including
  /// potential continuation codepoints)
  void lexUnknown();

  /// Lexing entry point
  void lexImpl();

  const char *tokBegPtr = nullptr;
  const char *curPtr = nullptr;
  const char *endPtr = nullptr;

  /// The next token that'll be returned.
  Token nextToken;
};
} // namespace sora