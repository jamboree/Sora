//===--- Parser.hpp - Sora Language Parser ----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/DiagnosticsParser.hpp"
#include "Sora/Lexer/Lexer.hpp"
#include "Sora/Parser/ParserResult.hpp"
#include <functional>

namespace sora {
class ASTContext;
class Decl;
class FuncDecl;
class Identifier;
class Lexer;
class ParamDecl;
class ParamList;
class SourceFile;
class TypeRepr;

/// Sora Language Parser
class Parser final {
public:
  /// \param ctxt the ASTContext that owns the SourceFile. This is where we'll
  /// allocate memory for the AST. We'll also use its DiagnosticEngine to emit
  /// diagnostic and its SourceManager to retrieve the SourceFile's contents.
  /// \param sf the SourceFile that this parser will be working on
  Parser(ASTContext &ctxt, SourceFile &file);

  /// Parses everything in the SourceFile until the whole file has been
  /// consumed.
  void parseAll();

  /// The ASTContext
  ASTContext &ctxt;
  /// The Diagnostic Engine
  DiagnosticEngine &diagEng;
  /// The SourceFile that this parser is working on
  SourceFile &file;

private:
  /// Our lexer instance
  Lexer lexer;

  /// The current token being considered by the parser
  Token tok;

  /// The SourceLoc that's right past-the-end of the last token consumed by the
  /// parser.
  SourceLoc prevTokPastTheEnd;

  //===- Declaration Parsing ----------------------------------------------===//

  /// \returns true if the parser is positioned at the start of a declaration.
  bool isStartOfDecl() const;

  /// Parses a parameter-declaration
  /// The parser must be positioned on the identifier.
  ParserResult<ParamDecl> parseParamDecl();

  /// Parses a parameter-declaration-list
  /// The parser must be positioned on the first "("
  ParserResult<ParamList> parseParamDeclList();

  /// Parses a function-declaration.
  /// The parser must be positioned on the "func" keyword.
  ParserResult<FuncDecl> parseFuncDecl();

  //===- Statement Parsing ------------------------------------------------===//

  /// \returns true if the parser is positioned at the start of a statement.
  bool isStartOfStmt() const;

  //===- Expression Parsing -----------------------------------------------===//

  // TODO

  //===- Type Parsing -----------------------------------------------------===//

  /// Parses a type. Calls \p onNoType if no type was found.
  ParserResult<TypeRepr> parseType(std::function<void()> onNoType);

  //===- Pattern Parsing --------------------------------------------------===//

  // TODO

  //===- Diagnostic Emission ----------------------------------------------===//

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

  /// Emits a "expected" diagnostic.
  /// The diagnostic points at the beginning of the current token, or, if it's
  /// at the beginning of a line, right past the end of the previous token
  /// consumed by the parser.
  template <typename... Args>
  InFlightDiagnostic
  diagnoseExpected(TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    SourceLoc loc;
    if (tok.isAtStartOfLine() && prevTokPastTheEnd)
      loc = prevTokPastTheEnd;
    else
      loc = tok.getLoc();
    assert(loc && "loc is null?");
    return diagEng.diagnose<Args...>(loc, diag, args...);
  }

  //===- Token Consumption & Peeking --------------------------------------===//

  /// Peeks the next token
  const Token &peek() const;

  /// Consumes the current token, replacing it with the next one.
  /// \returns the SourceLoc of the consumed token.
  SourceLoc consumeToken();

  /// Consumes the current token, replacing it with the next one.
  /// This check that the current token's kind is equal to \p kind
  /// \returns the SourceLoc of the consumed token.
  SourceLoc consume(TokenKind kind) {
    assert(tok.getKind() == kind && "Wrong kind!");
    return consumeToken();
  }

  /// Consumes an identifier, putting the result in \p identifier and returning
  /// its SourceLoc.
  SourceLoc consumeIdentifier(Identifier &identifier);

  /// Consumes the current token if its kind is equal to \p kind
  /// \returns the SourceLoc of the consumed token, or SourceLoc() if no token
  /// was consumed.
  SourceLoc consumeIf(TokenKind kind) {
    if (tok.is(kind))
      return consumeToken();
    return SourceLoc();
  }

  //===- Recovery ---------------------------------------------------------===//

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

  //===- Miscellaneous ----------------------------------------------------===//

  /// \returns true if the parser has reached EOF
  bool isEOF() const { return tok.is(TokenKind::EndOfFile); }
};
} // namespace sora