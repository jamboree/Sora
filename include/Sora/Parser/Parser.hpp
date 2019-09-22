//===--- Parser.hpp - Sora Language Parser ----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Lexer/Lexer.hpp"
#include "Sora/Parser/ParserResult.hpp"

namespace sora {
class ASTContext;
class Lexer;

/// Sora Language Parser
class Parser final {
public:
  Parser(Lexer &lexer, ASTContext &ctxt, DiagnosticEngine &diagEng)
      : lexer(lexer), ctxt(ctxt), diagEng(diagEng) {}

  Lexer &lexer;
  ASTContext &ctxt;
  DiagnosticEngine &diagEng;

private:
  /// Emits a diagnostic at \p tok's location.
  template <typename... Args>
  InFlightDiagnostic
  diagnose(const Token &tok, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagEng.diagnose<Args...>(tok.getLoc(), diag, args...);
  }

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagEng.diagnose<Args...>(loc, diag, args...);
  }

  /// The current token being considered by the parser
  Token tok;

  /// Peeks the next token
  const Token &peek() const;

  /// Consumes the current token, replacing it with the next one.
  /// \returns the SourceLoc of the consumed token.
  SourceLoc consume();

  /// Consumes the current token, replacing it with the next one.
  /// This check that the current token's kind is equal to \p kind
  /// \returns the SourceLoc of the consumed token.
  SourceLoc consume(TokenKind kind) {
    assert(tok.getKind() == kind && "Wrong kind!");
    return consume();
  }

  /// Consumes the current token if its kind is equal to \p kind
  /// \returns the SourceLoc of the consumed token, or SourceLoc() if no token
  /// was consumed.
  SourceLoc consumeIf(TokenKind kind) {
    if (tok.is(kind))
      return consume();
    return SourceLoc();
  }

  /// \returns true if the parser is positioned at the start of a declaration.
  bool isStartOfDecl() const;
  /// \returns true if the parser is positioned at the start of a statement.
  bool isStartOfStmt() const;

  /// \returns true if the parser has reached EOF
  bool isEOF() const { return tok.is(TokenKind::EndOfFile); }

  /// Skips the current token, matching parentheses.
  /// (e.g. if the current token is {, this skips until past the next })
  void skip();

  /// Skips until the next token of kind \p kind without consuming it.
  void skipUntil(TokenKind kind);

  /// Skips to the next Decl or }
  /// \returns true if the current token begins a statement.
  bool skipUntilDeclRCurly();

  /// Skips to the next Decl, Stmt or }
  /// \returns true if the current token begins a declaration.
  bool skipUntilDeclStmtRCurly();
};
} // namespace sora