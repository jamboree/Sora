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
#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
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
  Lexer(const Lexer &) = delete;
  Lexer &operator=(const Lexer &) = delete;

public
      : Lexer(const SourceManager &srcMgr, BufferID buffer,
              DiagnosticEngine *diagEng);

  /// Lex a token and return it.
  /// If we reached EOF, this will simply return the EOF token whenever
  /// it is called.
  Token lex();

  /// \returns the token that begins at \p loc
  static Token getTokenAtLoc(const SourceManager &srcMgr, SourceLoc loc);

  /// Converst a SourceRange/Loc to a CharSourceRange.
  static CharSourceRange toCharSourceRange(const SourceManager &srcMgr,
                                           SourceRange range) {
    return CharSourceRange(srcMgr, range.begin,
                           getTokenAtLoc(srcMgr, range.end).getEndLoc());
  }

  /// \returns a view of the next token that will be returned by "lex" without
  /// consuming it or changing the state of the lexer.
  const Token &peek() const { return nextToken; }

  /// The SourceManager instance
  const SourceManager &srcMgr;

private:
  void moveTo(const char *ptr) {
    assert(((begPtr <= ptr) && (ptr <= endPtr)) &&
           "ptr is from a different buffer");
    curPtr = ptr;
    tokBegPtr = nullptr;
    tokenIsAtStartOfLine = (curPtr == begPtr);
  }

  /// NOTE: This returns an invalid InFlightDiagnostic if diagnostic emission is
  /// not supported.
  template <typename... Args>
  InFlightDiagnostic
  diagnose(const char *loc, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    if (diagEng)
      return diagEng->diagnose<Args...>(SourceLoc::fromPointer(loc), diag,
                                        args...);
    return InFlightDiagnostic();
  }

  /// Finishes lexing (sets nextToken = EOF and cur = end)
  void stopLexing() {
    assert(nextToken.isNot(TokenKind::EndOfFile) && "already EOF");
    nextToken = Token(TokenKind::EndOfFile,
                      CharSourceRange(SourceLoc::fromPointer(endPtr)),
                      tokenIsAtStartOfLine);
    curPtr = endPtr;
  }

  /// Lexs an unknown (possibly utf-8) character sequence that starts at
  /// tokBegPtr and moves curPtr past-the-end of the sequence (including
  /// potential continuation codepoints)
  void lexUnknown();

  /// Lexs a number (int or float) that starts at tokBegPtr and moves
  /// curPtr past-the-end of the number
  void lexNumberLiteral();

  /// Lexs the body of an identifier that starts at tokBegPtr and moves
  /// curPtr past-the-end of the number. curPtr is assumed
  /// to be positioned past-the-end of the head of the identifier.
  ///
  /// If the identifier is recognized as a language keyword, the
  /// TokenKind will be the keyword's kind, else it'll simply
  /// be Identifier.
  void lexIdentifierBody();

  /// Handles a line (//) comment.
  /// curPtr must be at the position of the first '/'.
  /// Sets tokenIsAtStartOfLine to true if a '\n' was found.
  void handleLineComment();

  /// Handles a block (/* */) comment.
  /// curPtr must be at the position of the first '/'.
  /// Sets tokenIsAtStartOfLine to true if a '\n' was found
  /// while skipping the block.
  void handleBlockComment();

  /// Lexing entry point
  void lexImpl();

  /// Creates a new token and puts it in nextToken.
  /// Sets tokenIsAtStartOfLine to false after pushing the token.
  void pushToken(TokenKind kind) {
    nextToken = Token(kind, CharSourceRange::fromPointers(tokBegPtr, curPtr),
                      tokenIsAtStartOfLine);
    tokenIsAtStartOfLine = false;
  }

  /// Consumes trivia characters in front of curPtr. This sets
  /// tokenIsAtStartOfLine if a \n is found.
  ///
  /// Trivia is any whitespace character, or comments.
  void consumeTrivia();

  /// \returns the current token as a string
  StringRef getTokStr() const;

  /// The (optional) DiagnosticEngine instance
  DiagnosticEngine *diagEng = nullptr;

  /// Whether the next token is at the start of a line.
  bool tokenIsAtStartOfLine = false;

  const char *begPtr = nullptr;
  const char *tokBegPtr = nullptr;
  const char *curPtr = nullptr;
  const char *endPtr = nullptr;

  /// The next token that'll be returned.
  Token nextToken;
};
} // namespace sora
