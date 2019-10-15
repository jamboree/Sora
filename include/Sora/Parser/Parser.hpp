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
class Pattern;
class SourceFile;
class TypeRepr;

/// Sora Language Parser
class Parser final {
  Parser(const Parser &) = delete;
  Parser &operator=(const Parser &) = delete;

public:
  /// \param ctxt the ASTContext that owns the SourceFile. This is where we'll
  /// allocate memory for the AST. We'll also use its DiagnosticEngine to emit
  /// diagnostic and its SourceManager to retrieve the SourceFile's contents.
  /// \param sf the SourceFile that this parser will be working on
  Parser(ASTContext &ctxt, SourceFile &file);

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
  /// \param vars for LetDecls, the vector where the vars declared by the
  /// LetDecl will be stored.
  /// \param isTopLevel if true, only top-level-declarations are allowed, and
  /// declarations that can't appear at the top level are diagnosed.
  ///
  /// isStartOfDecl() must return true.
  ParserResult<Decl> parseDecl(SmallVectorImpl<VarDecl *> &vars,
                                       bool isTopLevel = false);

  /// Parses a let-declaration
  /// \param vars the vector where the vars declared by the LetDecl will be
  /// stored.
  ///
  /// The parser must be positioned on the "let" keyword.
  ParserResult<Decl> parseLetDecl(SmallVectorImpl<VarDecl *> &vars);

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
  /// \returns SourceLoc() if not found.
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

  /// Parses a primary-expression
  ParserResult<Expr> parsePrimaryExpr(llvm::function_ref<void()> onNoExpr);

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

  //===- Type Parsing -----------------------------------------------------===//

  /// Parses a type. Calls \p onNoType if no type was found.
  ParserResult<TypeRepr> parseType(llvm::function_ref<void()> onNoType);

  /// Parses a tuple type.
  /// The parser must be positioned on the "("
  ParserResult<TypeRepr> parseTupleType();

  /// Parses an array type.
  /// The parser must be positioned on the "["
  /// NOTE: ArrayTypes are currenlty unsupported by Sora, so this method
  /// is currently disabled. Do not use it.
  ParserResult<TypeRepr> parseArrayType();

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

  /// Parses a comma-separated list of values. The callback is called to parse
  /// elements, and the function takes care of consuming the commas.
  /// \param callBack The element parsing function. Returns a boolean indicating
  /// whether parsing should continue. It takes a single argument which is the
  /// position of the element we're parsing.
  /// The callback is responsible for emitting diagnostics as this function
  /// won't emit any on its own.
  void parseList(llvm::function_ref<bool(unsigned)> callback);

  //===- Diagnostic Emission ----------------------------------------------===//

  /// Emits a diagnostic at \p tok's location.
  template <typename... Args>
  InFlightDiagnostic
  diagnose(const Token &tok, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    return diagnose(tok.getLoc(), diag, args...);
  }

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
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
};
} // namespace sora