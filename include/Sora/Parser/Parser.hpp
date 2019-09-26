//===--- Parser.hpp - Sora Language Parser ----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/DiagnosticsParser.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Lexer/Lexer.hpp"
#include "Sora/Parser/ParserResult.hpp"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/SaveAndRestore.h"
#include <functional>

namespace sora {
class ASTContext;
class BlockStmt;
class Decl;
class DeclContext;
class Expr;
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
  void parseSourceFile();

  /// The ASTContext
  ASTContext &ctxt;
  /// The Diagnostic Engine
  DiagnosticEngine &diagEng;
  /// The SourceFile that this parser is working on
  SourceFile &sourceFile;
  /// The current DeclContext
  DeclContext *declContext = nullptr;

  llvm::SaveAndRestore<DeclContext *> setDeclContextRAII(DeclContext *newDC) {
    return {declContext, newDC};
  }

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
  /// The parser must be positioned on the "("
  ParserResult<ParamList> parseParamDeclList();

  /// Parses a function-declaration.
  /// The parser must be positioned on the "func" keyword.
  ParserResult<FuncDecl> parseFuncDecl();

  //===- Statement Parsing ------------------------------------------------===//

  /// \returns true if the parser is positioned at the start of a statement.
  bool isStartOfStmt() const;

  /// Parses a block-statement
  /// The parser must be positioned on the "{"
  ParserResult<BlockStmt> parseBlockStmt();

  //===- Expression Parsing -----------------------------------------------===//

  ParserResult<Expr> parseExpr(const std::function<void()> &onNoExpr);

  //===- Type Parsing -----------------------------------------------------===//

  /// Parses a type. Calls \p onNoType if no type was found.
  ParserResult<TypeRepr> parseType(const std::function<void()> &onNoType);

  /// Parses a tuple type.
  /// The parser must be positioned on the "("
  ParserResult<TypeRepr> parseTupleType();

  /// Parses an array type.
  /// The parser must be positioned on the "["
  ParserResult<TypeRepr> parseArrayType();

  /// Parses a reference or pointer type.
  /// The parser must be positioned on the "&" or "*"
  ParserResult<TypeRepr> parseReferenceOrPointerType();

  //===- Pattern Parsing --------------------------------------------------===//

  // TODO

  //===- Other Parsing Utilities ------------------------------------------===//

  /// Parses a matching token (parentheses or square/curly brackets).
  /// Emits a diagnostic and a note if the token is not found.
  /// \param the SourceLoc of the left matching token
  /// \param the kind of the right matching token (RParen, RCurly or RSquare)
  /// \param customErr if a custom diagnostic is provided, it'll be used
  ///                  instead of the default error message.
  /// \returns a valid SourceLoc on success, and invalid one on failure.
  SourceLoc parseMatchingToken(SourceLoc lLoc, TokenKind kind,
                               Optional<TypedDiag<>> customErr = None);

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

  /// Emits a diagnostic at the beginning of \p buffer
  template <typename... Args>
  InFlightDiagnostic
  diagnose(BufferID buffer, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagEng.diagnose<Args...>(buffer, diag, args...);
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

  /// Skips to the next \p tok, Decl or }
  void skipUntilDeclRCurly(TokenKind tok = TokenKind::Invalid);

  /// Skips to the next \p tok, Decl, Stmt or }
  void skipUntilDeclStmtRCurly(TokenKind tok = TokenKind::Invalid);

  //===- Miscellaneous ----------------------------------------------------===//

  /// \returns true if the parser has reached EOF
  bool isEOF() const { return tok.is(TokenKind::EndOfFile); }
};
} // namespace sora