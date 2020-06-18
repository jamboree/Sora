//===--- Parser.hpp - Sora Language Parser ----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Decl.hpp"
#include "Sora/AST/OperatorKinds.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "Sora/Diagnostics/DiagnosticsParser.hpp"
#include "Sora/Lexer/Lexer.hpp"
#include "Sora/Parser/ParserResult.hpp"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include <functional>

namespace sora {
class ASTContext;
class BlockStmt;
class Expr;
class Identifier;
class Lexer;
class LetDecl;
class Pattern;
class SourceFile;
class StmtCondition;
class TypeRepr;

/// Sora Language Parser
///
/// Note: Parsing method should return nullptr when they fail to parse
/// something and can't recover, and return a value when they successfully
/// recovered. They can also use makeParserErrorResult to create a result
/// with an error bit set to tell the caller that an error occured but
/// we successfully recovered.
/// Alternatively, parsing methods can also use makeParserResult(false, node)
/// to create an error parser result with a value. This can be used to notify
/// the caller that something went wrong during the parsing but it recovered
/// successfully and thus parsing can continue.
class Parser final {
  Parser(const Parser &) = delete;
  Parser &operator=(const Parser &) = delete;

public:
  /// \param sf the SourceFile that this parser will be working on.
  /// Its ASTContext, DiagnosticEngine and SourceManager will be used to
  /// allocate memory, emit diagnostics and access the file's text.
  Parser(SourceFile &file);

  /// The ASTContext
  ASTContext &ctxt;
  /// The Diagnostic Engine
  DiagnosticEngine &diagEng;
  /// The SourceFile that this parser is working on
  SourceFile &sourceFile;

private:
  /// The current DeclContext
  DeclContext *declContext = nullptr;

  /// Our lexer instance
  Lexer lexer;

  /// The current token being considered by the parser
  Token tok;

  /// The SourceLoc that's right past-the-end of the last token consumed by the
  /// parser.
  SourceLoc prevTokPastTheEnd;

  /// Whether we can apply the "mut" specifier when parsing a pattern.
  bool canApplyMutSpecifier = true;

public:
  //===- Source-File Parsing ----------------------------------------------===//

  /// Parses a source-file
  void parseSourceFile();

  //===- Declaration Parsing ----------------------------------------------===//

  /// \returns true if the parser is positioned at the start of a declaration.
  bool isStartOfDecl() const;

  /// Parses a declaration or top-level-declaration.
  /// \param isTopLevel if true, only top-level-declarations are allowed, and
  /// declarations that can't appear at the top level are diagnosed.
  ///
  /// isStartOfDecl() must return true.
  ParserResult<Decl> parseDecl(bool isTopLevel = false);

  /// Parses a let-declaration
  /// The parser must be positioned on the "let" keyword.
  ParserResult<LetDecl> parseLetDecl();

  /// Parses a parameter-declaration
  /// The parser must be positioned on the identifier.
  ParserResult<ParamDecl> parseParamDecl();

  /// Parses a parameter-declaration-list
  /// The parser must be positioned on the "("
  ParserResult<ParamList> parseParamDeclList();

  /// Parses a function-declaration.
  /// The parser must be positioned on the "func" keyword.
  ParserResult<FuncDecl> parseFuncDecl();

  //===- Expression Parsing -----------------------------------------------===//

  /// Parses an expression
  ParserResult<Expr> parseExpr(llvm::function_ref<void()> onNoExpr);

  /// Parses an assignement-expression
  ParserResult<Expr> parseAssignementExpr(llvm::function_ref<void()> onNoExpr);

  /// Consumes an assignement-operator
  /// \param result the operator that was found. Will not be changed if no
  /// operator was found.
  /// \returns SourceLoc() if not found.
  SourceLoc consumeAssignementOperator(BinaryOperatorKind &result);

  /// Parses a conditional-expression
  ParserResult<Expr> parseConditionalExpr(llvm::function_ref<void()> onNoExpr);

  /// Binary operator precedences, from highest (0) to lowest (last).
  enum class PrecedenceKind : uint8_t {
    /// Multiplicative Operators: * / %
    Multiplicative = 0,
    /// Shift operators: << >>
    Shift,
    /// Bitwise operators: | ^ &
    Bitwise,
    /// Additive operators: + -
    Additive,
    /// Relational operators: == != < <= > >=
    Relational,
    /// Logical operators: && ||
    Logical,
    /// Null-Coalescing operator: ??
    NullCoalesce,

    HighestPrecedence = Multiplicative,
    LowestPrecedence = NullCoalesce
  };

  /// Parses a binary-expression.
  ///
  /// NOTE: This uses precedence-climbing (lowest to highest) in order to
  /// respect operator precedences, and \p precedence will be the "starting"
  /// precedence (usually it's LowestPrecedence).
  ParserResult<Expr>
  parseBinaryExpr(llvm::function_ref<void()> onNoExpr,
                  PrecedenceKind precedence = PrecedenceKind::LowestPrecedence);

  /// Consumes an binary-operator
  /// \param result the operator that was found. Will not be changed if no
  /// operator was found.
  /// \returns SourceLoc() if not found
  ///
  /// Note that some binary operators will be ignored when they're at the start
  /// of a line, because they can be confused with unary operators. e.g. +
  SourceLoc consumeBinaryOperator(BinaryOperatorKind &result,
                                  PrecedenceKind precedence);

  /// Parses a cast-expression
  ParserResult<Expr> parseCastExpr(llvm::function_ref<void()> onNoExpr);

  /// Parses a prefix-expression
  ParserResult<Expr> parsePrefixExpr(llvm::function_ref<void()> onNoExpr);

  /// Consumes a prefix-operator
  /// \param result the operator that was found. Will not be changed if no
  /// operator was found.
  /// \returns SourceLoc() if not found.
  SourceLoc consumePrefixOperator(UnaryOperatorKind &result);

  /// Parses a postfix-expression
  ParserResult<Expr> parsePostfixExpr(llvm::function_ref<void()> onNoExpr);

  /// Parses a member-access on \p base (a suffix).
  /// The parser must be positioned on the '.' or '->'
  ParserResult<Expr> parseMemberAccessExpr(Expr *base);

  /// Parses a primary-expression
  ParserResult<Expr> parsePrimaryExpr(llvm::function_ref<void()> onNoExpr);

  /// Parses a tuple-expression, returning a TupleExpr/ParenExpr on success.
  /// The parser must be positioned on the '('.
  ParserResult<Expr> parseTupleExpr();

  /// Parses a tuple-expression.
  /// The parser must be positioned on the '('.
  bool parseTupleExpr(SourceLoc &lParenLoc, SmallVectorImpl<Expr *> &exprs,
                      SourceLoc &rParenLoc);

  //===- Pattern Parsing --------------------------------------------------===//

  /// Parses a pattern
  ParserResult<Pattern> parsePattern(llvm::function_ref<void()> onNoPat);

  /// Parses a tuple-pattern.
  /// The parse must be positioned on the '('
  ParserResult<Pattern> parseTuplePattern();

  //===- Statement Parsing ------------------------------------------------===//

  /// \returns true if the parser is positioned at the start of a statement.
  bool isStartOfStmt() const;

  /// Parses a statement.
  /// isStartOfStmt() must return true.
  ParserResult<Stmt> parseStmt();

  /// Parses a block-statement
  /// The parser must be positioned on the "{"
  ParserResult<BlockStmt> parseBlockStmt();

  /// Parses a return-statement
  /// The parser must be positioned on the 'return' keyword.
  ParserResult<Stmt> parseReturnStmt();

  /// Parses a if-statement.
  /// The parser must be positioned on the 'if' keyword.
  ParserResult<Stmt> parseIfStmt();

  /// Parses a while-statement.
  /// The parser must be positioned on the 'while' keyword.
  ParserResult<Stmt> parseWhileStmt();

  // Parses a condition
  /// \param cond where the result will be stored
  /// \param name the name of the condition (for diagnostics), e.g. "if".
  /// \returns true if no parsing error occured, false otherwise.
  bool parseCondition(StmtCondition &cond, StringRef name);

  //===- Type Parsing -----------------------------------------------------===//

  /// Parses a type. Calls \p onNoType if no type was found.
  ParserResult<TypeRepr> parseType(llvm::function_ref<void()> onNoType);

  /// Parses a tuple type.
  /// The parser must be positioned on the "("
  ParserResult<TypeRepr> parseTupleType();

  /// Parses a reference type.
  /// The parser must be positioned on the "&".
  ParserResult<TypeRepr> parseReferenceType();

  /// Parses a "maybe" type
  /// The parser must be positioned on the "maybe" keyword.
  ParserResult<TypeRepr> parseMaybeType();

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

  /// Parses a comma-separated list of values.
  ///
  /// \param callBack The element parsing function. Returns a boolean indicating
  /// whether parsing should continue. It takes a single argument which is the
  /// position of the element we're parsing.
  ///
  /// The callback is always called at least once.
  void parseList(llvm::function_ref<bool(size_t)> callback);

  /// Parses a comma-separated list of values inside a parentheses.
  /// The parser must be positioned on the '('
  ///
  /// \param rParenloc If found, the SourceLoc of the ')' will be stored in this
  /// variable. If not found, this is set to prevTokPastTheEnd.
  /// \param callBack The element parsing function. Returns
  /// true on success, false on parsing error. The callback is not called when
  /// the next token is a ')', so you don't need to handle ')' in the callback.
  /// \param missingRParenDiag passed to parseMatchingToken
  ///
  /// \returns true on success, false on failure.
  bool parseTuple(SourceLoc &rParenLoc,
                  llvm::function_ref<bool(size_t)> callback,
                  Optional<TypedDiag<>> missingRParenDiag = None);

  //===- Diagnostic Emission ----------------------------------------------===//

  /// Emits a diagnostic at \p tok's location.
  template <typename... Args>
  InFlightDiagnostic
  diagnose(const Token &tok, const TypedDiag<Args...> &diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagnose(tok.getLoc(), diag, args...);
  }

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, const TypedDiag<Args...> &diag,
           typename detail::PassArgument<Args>::type... args) {
    assert(loc && "Parser can't emit diagnostics without SourceLocs");
    return diagEng.diagnose<Args...>(loc, diag, args...);
  }

  /// Emits a "expected" diagnostic.
  /// The diagnostic points at the beginning of the current token, or, if it's
  /// at the beginning of a line, right past the end of the previous token
  /// consumed by the parser.
  template <typename... Args>
  InFlightDiagnostic
  diagnoseExpected(const TypedDiag<Args...> &diag,
                   typename detail::PassArgument<Args>::type... args) {
    SourceLoc loc;
    if (tok.isAtStartOfLine() && prevTokPastTheEnd)
      loc = prevTokPastTheEnd;
    else
      loc = tok.getLoc();
    assert(loc && "loc is null?");
    return diagEng.diagnose<Args...>(loc, diag, args...);
  }

  //===- Current DeclContext Management -----------------------------------===//

  /// Sets the DeclContext that will be used by the parser.
  /// \returns a RAII object that restores the previous DeclContext on
  /// destruction.
  llvm::SaveAndRestore<DeclContext *> setDeclContextRAII(DeclContext *newDC) {
    return {declContext, newDC};
  }

  /// \returns the current DeclContext used by the parser
  DeclContext *getDeclContext() const { return declContext; }

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
  // NOTE: All of those methods will also stop at the EOF token.
  //===--------------------------------------------------------------------===//

  /// Skips the current token, matching parentheses.
  /// (e.g. if the current token is {, this skips until past the next })
  void skip();

  /// Skips until the next token of kind \p kind without consuming it.
  void skipUntil(TokenKind kind);

  /// Skips to the next Decl
  void skipUntilDecl();

  /// Skips until the next tok or newline.
  void skipUntilTokOrNewline(TokenKind tok = TokenKind::Invalid);

  /// Skips to the next \p tok, Decl or }
  void skipUntilTokDeclRCurly(TokenKind tok = TokenKind::Invalid);

  /// Skips to the next \p tok, Decl, Stmt or }
  void skipUntilTokDeclStmtRCurly(TokenKind tok = TokenKind::Invalid);

  //===- Miscellaneous ----------------------------------------------------===//

  /// \returns true if the parser has reached EOF
  bool isEOF() const { return tok.is(TokenKind::EndOfFile); }

  /// \returns an identifier object for the contents (string) of \p tok
  Identifier getIdentifier(const Token &tok);

  /// \returns the difference between the column number of a and b.
  /// e.g if a is column 5, and b is column 4, returns -1.
  int getColumnDifference(SourceLoc a, SourceLoc b) const;
};
} // namespace sora