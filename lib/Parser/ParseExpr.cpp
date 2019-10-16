//===--- ParseExpr.cpp - Expression Parsing Impl. ---------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Expr.hpp"
#include "Sora/Parser/Parser.hpp"

using namespace sora;

/*
expression = assignement-expression
*/
ParserResult<Expr> Parser::parseExpr(llvm::function_ref<void()> onNoExpr) {
  // TODO
  // This is just to make some tests work.
  if (tok.is(TokenKind::IntegerLiteral)) {
    auto loc = consumeToken();
    return makeParserResult(new (ctxt) ErrorExpr(loc));
  }
  onNoExpr();
  return nullptr;
}

/*
assignement-expression =
  conditional-expression (assignement-operator assignement-expression)?
*/
ParserResult<Expr>
Parser::parseAssignementExpr(llvm::function_ref<void()> onNoExpr) {
  // conditional-expression
  Expr *lhs = parseConditionalExpr(onNoExpr).getOrNull();
  if (!lhs)
    return nullptr;
  // (assignement-operator assignement-expression)?
  BinaryOperatorKind op;
  SourceLoc opLoc = consumeAssignementOperator(op);
  if (!opLoc)
    return makeParserResult(lhs);
  onNoExpr = [&]() {
    diagnoseExpected(diag::expected_expr_after, getSpelling(op));
  };
  Expr *rhs = parseAssignementExpr(onNoExpr).getOrNull();
  if (!rhs)
    return nullptr;
  return makeParserResult(new (ctxt) BinaryExpr(lhs, op, opLoc, rhs));
}

/*
assignement-operator = '=' | "+=" | "-=" | "/=" | "*=" | "%="
 | ">>=" | "<<=" | "&=" | "|=" | "^=" | '??='
*/
SourceLoc Parser::consumeAssignementOperator(BinaryOperatorKind &result) {
  using TK = TokenKind;
  using Op = BinaryOperatorKind;

#define CASE(TOK, OP)                                                          \
  case TOK:                                                                    \
    result = OP;
  return consumeToken();
  switch (tok.getKind()) {
  default:
    return {};
    CASE(TK::Equal, Op::Assign);
    CASE(TK::PlusEqual, Op::AddAssign);
    CASE(TK::MinusEqual, Op::SubAssign);
    CASE(TK::SlashEqual, Op::DivAssign);
    CASE(TK::StarEqual, Op::MulAssign);
    CASE(TK::PercentEqual, Op::RemAssign);
    CASE(TK::LessLessEqual, Op::ShlAssign);
    CASE(TK::GreaterGreaterEqual, Op::ShrAssign);
    CASE(TK::AmpEqual, Op::AndAssign);
    CASE(TK::PipeEqual, Op::OrAssign);
    CASE(TK::CaretEqual, Op::XOrAssign);
    CASE(TK::QuestionQuestionEqual, Op::NullCoalesceAssign);
  }
#undef CASE
  return {}; // Avoid C4715 on MSC.
}

/*
conditional-expression =
  binary-expression ('?' expression ':' conditional-expression)?
*/
ParserResult<Expr>
Parser::parseConditionalExpr(llvm::function_ref<void()> onNoExpr) {
  // binary-expression
  Expr *cond = parseBinaryExpr(onNoExpr).getOrNull();
  if (!cond)
    return nullptr;

  // ('?' expression ':' conditional-expression)?
  if (!tok.is(TokenKind::Question))
    return makeParserResult(cond);

  // '?'
  SourceLoc questionLoc = consumeToken();
  // expression
  onNoExpr = [&]() {
    diagnoseExpected(diag::expected_expr_after_in_ternary, "?");
  };
  Expr *thenExpr = parseExpr(onNoExpr).getOrNull();
  // If we don't have an expr, check if we got a colon, because if we do, we can
  // still recover & return a failure after.
  if (!thenExpr && tok.isNot(TokenKind::Colon))
    return nullptr;

  // ':'
  if (!tok.is(TokenKind::Colon)) {
    diagnoseExpected(diag::expected_colon);
    return nullptr;
  }

  SourceLoc colonLoc = consumeToken();
  // conditional-expression
  onNoExpr = [&]() {
    diagnoseExpected(diag::expected_expr_after_in_ternary, ":");
  };
  Expr *elseExpr = parseConditionalExpr(onNoExpr).getOrNull();

  // If everything parsed correctly, return a fully-formed ConditionalExpr,
  // else, just return the cond with the error bit set.
  if (!thenExpr || !elseExpr)
    return makeParserErrorResult(cond);

  Expr *result = new (ctxt)
      ConditionalExpr(cond, questionLoc, thenExpr, colonLoc, elseExpr);
  return makeParserResult(result);
}

/*
binary-expression = cast-expression (binary-operator cast-expression)*
*/
ParserResult<Expr> Parser::parseBinaryExpr(llvm::function_ref<void()> onNoExpr,
                                           PrecedenceKind precedence) {
  // Parses an operand, which is either a cast-expression or a binary-expression
  // of higher precedence.
  auto parseOperand = [&]() {
    if (precedence == PrecedenceKind::HighestPrecedence)
      return parseCastExpr(onNoExpr);
    return parseBinaryExpr(onNoExpr, PrecedenceKind(unsigned(precedence) - 1));
  };

  // left operaand
  Expr *current = parseOperand().getOrNull();
  if (!current)
    return nullptr;

  // (binary-operator (right operand))*
  BinaryOperatorKind op;
  while (SourceLoc opLoc = consumeBinaryOperator(op, precedence)) {
    onNoExpr = [&]() {
      diagnoseExpected(diag::expected_expr_after, getSpelling(op));
    };
    Expr *rhs = parseOperand().getOrNull();
    if (!rhs)
      return nullptr;

    current = new (ctxt) BinaryExpr(current, op, opLoc, rhs);
  }

  return makeParserResult(current);
}

/*
binary-operator = '+' | '-' | '/' | '*' | '%' | ">>" | "<<" | '&' | '|' | '^' |
  "==" | "!=" | '<' | '>' | "<=" | ">=" | "||" | "&&" | '??'
*/
SourceLoc Parser::consumeBinaryOperator(BinaryOperatorKind &result,
                                        PrecedenceKind precedence) {
  using TK = TokenKind;
  using Op = BinaryOperatorKind;
#define CASE(TOK, OP)                                                          \
  case TOK:                                                                    \
    result = OP;                                                               \
    return consumeToken()
  switch (precedence) {
  default:
    return {};
  case PrecedenceKind::Multiplicative:
    switch (tok.getKind()) {
    default:
      return {};
      CASE(TK::Star, Op::Mul);
      CASE(TK::Slash, Op::Div);
      CASE(TK::Percent, Op::Rem);
    }
  case PrecedenceKind::Shift:
    switch (tok.getKind()) {
    default:
      return {};
      CASE(TK::LessLess, Op::Shl);
      CASE(TK::GreaterGreater, Op::Shr);
    }
  case PrecedenceKind::Bitwise:
    switch (tok.getKind()) {
    default:
      return {};
      CASE(TK::Pipe, Op::Or);
      CASE(TK::Caret, Op::XOr);
      CASE(TK::Amp, Op::And);
    }
  case PrecedenceKind::Additive:
    switch (tok.getKind()) {
    default:
      return {};
      CASE(TK::Plus, Op::Add);
      CASE(TK::Minus, Op::Sub);
    }
  case PrecedenceKind::Relational:
    switch (tok.getKind()) {
    default:
      return {};
      CASE(TK::EqualEqual, Op::Eq);
      CASE(TK::ExclaimEqual, Op::NEq);
      CASE(TK::Less, Op::LT);
      CASE(TK::LessEqual, Op::LE);
      CASE(TK::Greater, Op::GT);
      CASE(TK::GreaterEqual, Op::GE);
    }
  case PrecedenceKind::Logical:
    switch (tok.getKind()) {
    default:
      return {};
      CASE(TK::PipePipe, Op::LOr);
      CASE(TK::AmpAmp, Op::LAnd);
    }
  case PrecedenceKind::NullCoalesce:
    if (tok.is(TokenKind::QuestionQuestion)) {
      result = BinaryOperatorKind::NullCoalesce;
      return consumeToken();
    }
    break;
  }
#undef CASE
  return {}; // Avoid C4715 on MSC.
}

/*
cast-expression = prefix-expression ("as" type)*
*/
ParserResult<Expr> Parser::parseCastExpr(llvm::function_ref<void()> onNoExpr) {
  // prefix-expression
  Expr *expr = parsePrefixExpr(onNoExpr).getOrNull();
  // ("as" type)*
  // Note: we parse this in a loop, even if it has limited usefulness so the
  // compiler won't emit cryptic error messages if you type "foo as A as B".
  // FIXME: Should the extra casts be diagnosed?
  while (SourceLoc asLoc = consumeIf(TokenKind::AsKw)) {
    auto result =
        parseType([&]() { diagnoseExpected(diag::expected_type_after, "as"); });
    if (!result.hasValue())
      return nullptr;
    expr = new (ctxt) CastExpr(expr, asLoc, result.get());
  }
  return makeParserResult(expr);
}

/*
prefix-expression = prefix-operator prefix-expression
                  | postfix-expression
*/
ParserResult<Expr>
Parser::parsePrefixExpr(llvm::function_ref<void()> onNoExpr) {
  UnaryOperatorKind op;
  SourceLoc opLoc = consumePrefixOperator(op);
  if (!opLoc)
    return parsePostfixExpr(onNoExpr);

  auto result = parsePrefixExpr(
      [&]() { diagnoseExpected(diag::expected_expr_after, getSpelling(op)); });

  if (!result.hasValue())
    return nullptr;

  return makeParserResult(new (ctxt) UnaryExpr(op, opLoc, result.get()));
}

/*
prefix-operator = '+' | '-' | '!' | '*' | '&' | '~'
*/
SourceLoc Parser::consumePrefixOperator(UnaryOperatorKind &result) {
  using TK = TokenKind;
  using Op = UnaryOperatorKind;
#define CASE(TOK, OP)                                                          \
  case TOK:                                                                    \
    result = OP;                                                               \
    return consumeToken()
  switch (tok.getKind()) {
  default:
    return {};
    CASE(TK::Plus, Op::Plus);
    CASE(TK::Minus, Op::Minus);
    CASE(TK::Exclaim, Op::LNot);
    CASE(TK::Star, Op::Deref);
    CASE(TK::Amp, Op::AddressOf);
    CASE(TK::Tilde, Op::Not);
  }
#undef CASE
  return {}; // Avoid C4715 on MSC.
}

/*
postfix-expression = primary-expression suffix*
suffix = tuple-expression
       | member-access
       | '!'
*/
ParserResult<Expr>
Parser::parsePostfixExpr(llvm::function_ref<void()> onNoExpr) {
  Expr *base = parsePrimaryExpr(onNoExpr).getOrNull();
  if (!base)
    return nullptr;

parse_suffix:
  assert(base && "no base!");
  switch (tok.getKind()) {
  default:
    break;
  case TokenKind::LParen: {
    auto *args = parseTupleExpr().getOrNull();
    if (!args)
      return nullptr;
    base = new (ctxt) CallExpr(base, args);
    goto parse_suffix;
  }
  case TokenKind::Dot:
  case TokenKind::Arrow: {
    base = parseMemberAccessExpr(base).getOrNull();
    goto parse_suffix;
  }
  case TokenKind::Exclaim: {
    base = new (ctxt) ForceUnwrapExpr(base, consumeToken());
    goto parse_suffix;
  }
  }

  return makeParserResult(base);
}

/*
member-access-expression = ('.' | "->") (identifier | integer-literal)
*/
ParserResult<Expr> Parser::parseMemberAccessExpr(Expr *base) {
  assert(tok.isAny(TokenKind::Dot, TokenKind::Arrow));
  // ('.' | "->")
  bool isArrow = tok.is(TokenKind::Arrow);
  SourceLoc opLoc = consumeToken();

  // (identifier | integer-literal)
  Identifier ident;
  SourceLoc identLoc;
  switch (tok.getKind()) {
  default:
    diagnoseExpected(diag::expected_member_name_or_index_after, ".");
    return nullptr;
  case TokenKind::Identifier:
    identLoc = consumeIdentifier(ident);
    break;
  case TokenKind::IntegerLiteral:
    ident = getIdentifier(tok);
    identLoc = consumeToken();
    break;
  }

  Expr *expr =
      new (ctxt) UnresolvedMemberRefExpr(base, opLoc, isArrow, identLoc, ident);
  return makeParserResult(expr);
}

/*
primary-expression = identifier | '_' | tuple-expression | literal
literal = null-literal | integer-literal | floating-point-literal |
          boolean-literal
*/
ParserResult<Expr>
Parser::parsePrimaryExpr(llvm::function_ref<void()> onNoExpr) {
  Expr *expr = nullptr;
  switch (tok.getKind()) {
  default:
    onNoExpr();
    break;
  // identifier
  case TokenKind::Identifier: {
    Identifier ident;
    SourceLoc identLoc = consumeIdentifier(ident);
    expr = new (ctxt) UnresolvedDeclRefExpr(ident, identLoc);
    break;
  }
  // '_'
  case TokenKind::UnderscoreKw:
    expr = new (ctxt) DiscardExpr(consumeToken());
    break;
  // tuple-expression
  case TokenKind::LParen:
    // This may leave expr "null" in error. This is intended.
    expr = parseTupleExpr().getOrNull();
    break;
  // literal
  //    null-literal
  case TokenKind::NullKw:
    expr = new (ctxt) NullLiteralExpr(consumeToken());
    break;
  //    integer-literal
  case TokenKind::IntegerLiteral:
    expr = new (ctxt) IntegerLiteralExpr(tok.str(), consumeToken());
    break;
  //    floating-point-literal
  case TokenKind::FloatingPointLiteral:
    expr = new (ctxt) FloatLiteralExpr(tok.str(), consumeToken());
    break;
  //    boolean-literal = "true" | "false"
  case TokenKind::TrueKw:
    expr = new (ctxt) BooleanLiteralExpr(true, consumeToken());
    break;
  case TokenKind::FalseKw:
    expr = new (ctxt) BooleanLiteralExpr(false, consumeToken());
    break;
  }
  return expr ? makeParserResult(expr) : nullptr;
}

/*
tuple-expression = '(' expression-list? ')'
expression-list = expression (',' expression)*
*/
ParserResult<Expr> Parser::parseTupleExpr() {
  assert(tok.is(TokenKind::LParen));
  // '('
  SourceLoc lParen = consumeToken();

  /*Short path for empty tuples*/
  // ')'
  if (SourceLoc rParen = consumeIf(TokenKind::RParen))
    return makeParserResult(TupleExpr::createEmpty(ctxt, lParen, rParen));

  SmallVector<Expr *, 4> elements;

  // Utility function to create a ParenExpr or a TupleExpr depending on
  // the number of elements.
  auto createResult = [&](SourceLoc rParen) -> Expr * {
    if (elements.size() == 1)
      return new (ctxt) ParenExpr(lParen, elements[0], rParen);
    return TupleExpr::create(ctxt, lParen, elements, rParen);
  };

  // expression (',' expression)*
  do {
    auto result = parseExpr([&]() {
      // If we got no elements, the last thing we parsed was a '(', if we have
      // an element, the last thing we parsed was a ','
      diagnoseExpected(diag::expected_expr_after, elements.empty() ? "(" : ",");
    });
    if (!result.hasValue()) {
      skipUntilTokDeclStmtRCurly(TokenKind::LParen);
      if (tok.isNot(TokenKind::LParen))
        return nullptr;
      return makeParserErrorResult(createResult(consumeToken()));
    }
    elements.push_back(result.get());
  } while (consumeIf(TokenKind::Comma));

  // ')'
  SourceLoc rParen = parseMatchingToken(
      lParen, TokenKind::RParen, diag::expected_rparen_at_end_of_tuple_expr);
  if (!rParen)
    return nullptr;
  return makeParserResult(createResult(rParen));
}