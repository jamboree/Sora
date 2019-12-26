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
  return parseAssignementExpr(onNoExpr);
}

/*
assignement-expression =
  conditional-expression (assignement-operator assignement-expression)?
*/
ParserResult<Expr>
Parser::parseAssignementExpr(llvm::function_ref<void()> onNoExpr) {
  // conditional-expression
  auto result = parseConditionalExpr(onNoExpr);
  if (result.isNull())
    return result;

  // (assignement-operator assignement-expression)?
  BinaryOperatorKind op;
  bool startOfLine = tok.isAtStartOfLine();
  SourceLoc opLoc = consumeAssignementOperator(op);
  if (opLoc.isInvalid())
    return result;

  // Binary operators can't appear at the start of a line
  if (startOfLine)
    diagnose(opLoc, diag::binary_op_at_start_of_line, getSpelling(op));

  auto rhs = parseAssignementExpr(
      [&]() { diagnoseExpected(diag::expected_expr_after, getSpelling(op)); });
  if (rhs.isNull())
    return nullptr;
  return makeParserResult(new (ctxt)
                              BinaryExpr(result.get(), op, opLoc, rhs.get()));
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
    result = OP;                                                               \
    return consumeToken()
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
  auto result = parseBinaryExpr(onNoExpr);
  if (result.isNull())
    return result;

  // ('?' expression ':' conditional-expression)?
  if (tok.isNot(TokenKind::Question))
    return result;

  Expr *cond = result.get();

  // '?'
  SourceLoc questionLoc = consumeToken();
  // expression
  Expr *thenExpr =
      parseExpr([&]() {
        diagnoseExpected(diag::expected_expr_after_in_ternary, "?");
      }).getOrNull();
  // If we don't have an expr, check if we got a colon, because if we do, we can
  // still recover & return a failure after.
  if (!thenExpr && tok.isNot(TokenKind::Colon))
    return nullptr;

  // ':'
  if (tok.isNot(TokenKind::Colon)) {
    diagnoseExpected(diag::expected_colon);
    return nullptr;
  }

  SourceLoc colonLoc = consumeToken();
  // conditional-expression
  Expr *elseExpr =
      parseConditionalExpr(onNoExpr = [this]() {
        diagnoseExpected(diag::expected_expr_after_in_ternary, ":");
      }).getOrNull();

  // If everything parsed correctly, return a fully-formed ConditionalExpr,
  if (thenExpr && elseExpr) {
    Expr *expr = new (ctxt)
        ConditionalExpr(cond, questionLoc, thenExpr, colonLoc, elseExpr);
    return makeParserResult(expr);
  }
  // If we got a then but no else, return a cond with an ErrorExpr in the
  // middle.
  if (!thenExpr && elseExpr) {
    return makeParserErrorResult(new (ctxt) ConditionalExpr(
        cond, questionLoc, new (ctxt) ErrorExpr({}), colonLoc, elseExpr));
  }
  // Else just return the condition with the error bit set.
  return makeParserErrorResult(cond);
}

/*
binary-expression = cast-expression (binary-operator cast-expression)*
*/
ParserResult<Expr> Parser::parseBinaryExpr(llvm::function_ref<void()> onNoExpr,
                                           PrecedenceKind precedence) {
  // Parses an operand, which is either a cast-expression or a binary-expression
  // of higher precedence.
  auto parseOperand = [&](llvm::function_ref<void()> onNoExpr) {
    if (precedence == PrecedenceKind::HighestPrecedence)
      return parseCastExpr(onNoExpr);
    return parseBinaryExpr(onNoExpr, PrecedenceKind(unsigned(precedence) - 1));
  };

  // left operand
  auto result = parseOperand(onNoExpr);
  if (result.isNull())
    return result;
  Expr *current = result.get();

  // (binary-operator (right operand))*
  BinaryOperatorKind op;
  bool hasParsedOperator = false;
  while (true) {
    // Parse the operator, break if we can't parse it
    bool startOfLine = tok.isAtStartOfLine();
    SourceLoc opLoc = consumeBinaryOperator(op, precedence);
    if (opLoc.isInvalid())
      break;

    // Binary operators can't appear at the start of a line
    if (startOfLine)
      diagnose(opLoc, diag::binary_op_at_start_of_line, getSpelling(op));

    hasParsedOperator = true;
    Expr *rhs = parseOperand([&]() {
                  diagnoseExpected(diag::expected_expr_after, getSpelling(op));
                }).getOrNull();
    if (!rhs)
      return nullptr;

    current = new (ctxt) BinaryExpr(current, op, opLoc, rhs);
  }

  // If we didn't parse any operator, just forward the result
  return hasParsedOperator ? makeParserResult(current) : result;
}

/*
binary-operator = '+' | '-' | '/' | '*' | '%' | ">>" | "<<" | '&' | '|' | '^' |
  "==" | "!=" | '<' | '>' | "<=" | ">=" | "||" | "&&" | '??'
*/
SourceLoc Parser::consumeBinaryOperator(BinaryOperatorKind &result,
                                        PrecedenceKind precedence) {
  using TK = TokenKind;
  using Op = BinaryOperatorKind;
  // Case for Binary operators that have the same syntax as an unary operator.
  // Those are ignored when they're at the start of a line.
#define CONFUSABLE_CASE(TOK, OP)                                               \
  case TOK:                                                                    \
    if (tok.isAtStartOfLine())                                                 \
      return {};                                                               \
    result = OP;                                                               \
    return consumeToken()
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
      CONFUSABLE_CASE(TK::Star, Op::Mul);
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
      CONFUSABLE_CASE(TK::Amp, Op::And);
    }
  case PrecedenceKind::Additive:
    switch (tok.getKind()) {
    default:
      return {};
      CONFUSABLE_CASE(TK::Plus, Op::Add);
      CONFUSABLE_CASE(TK::Minus, Op::Sub);
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
  auto result = parsePrefixExpr(onNoExpr);
  if (result.isNull() || tok.isNot(TokenKind::AsKw))
    return result;

  Expr *expr = result.get();
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
  if (opLoc.isInvalid())
    return parsePostfixExpr(onNoExpr);

  auto result = parsePrefixExpr(
      [&]() { diagnoseExpected(diag::expected_expr_after, getSpelling(op)); });

  if (result.isNull())
    return nullptr;

  return makeParserResult(new (ctxt) UnaryExpr(op, opLoc, result.get()));
}

/*
prefix-operator = '+' | '-' | '!' | '*' | '&' | '~'
*/
SourceLoc Parser::consumePrefixOperator(UnaryOperatorKind &result) {
  using TK = TokenKind;
  using Op = UnaryOperatorKind;
  // NOTE: When adding new prefix operators that are confusable with an
  // existing binary operator (e.g. like '+', it can be unary and binary),
  // you'll need to teach consumeBinaryOperator to ignore them when they're at
  // the start of a line.
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
  auto result = parsePrimaryExpr(onNoExpr);
  if (result.isNull())
    return nullptr;

  Expr *base = result.get();

parse_suffix:
  assert(base && "no base!");
  switch (tok.getKind()) {
  default:
    break;
  case TokenKind::LParen: {
    // If the '(' is on another line, this isn't a call.
    if (tok.isAtStartOfLine())
      break;
    SourceLoc lParenLoc, rParenLoc;
    SmallVector<Expr *, 8> exprs;
    if (!parseTupleExpr(lParenLoc, exprs, rParenLoc))
      return nullptr;
    base = CallExpr::create(ctxt, base, lParenLoc, exprs, rParenLoc);
    goto parse_suffix;
  }
  case TokenKind::Dot:
  case TokenKind::Arrow: {
    /// FIXME: Ideally, a warning should be emitted if the token is
    /// at the start of a line w/ the same indent level.
    auto result = parseMemberAccessExpr(base);
    if (result.isNull())
      return nullptr;
    base = result.getOrNull();
    goto parse_suffix;
  }
  case TokenKind::Exclaim: {
    // If the '!' is on another line, this isn't a forced unwrapping.
    if (tok.isAtStartOfLine())
      break;
    base = new (ctxt) ForceUnwrapExpr(base, consumeToken());
    goto parse_suffix;
  }
  }

  // If we didn't parse any suffixes, just return the result so the error bit is
  // preserved.
  if (base == result.get())
    return result;
  return makeParserResult(base);
}

/*
member-access-expression = ('.' | "->") (identifier | integer-literal)
*/
ParserResult<Expr> Parser::parseMemberAccessExpr(Expr *base) {
  assert(tok.isAny(TokenKind::Dot, TokenKind::Arrow));
  assert(base && "base is nullptr");
  // ('.' | "->")
  bool isArrow = tok.is(TokenKind::Arrow);
  SourceLoc opLoc = consumeToken();

  // (identifier | integer-literal)
  Identifier ident;
  SourceLoc identLoc;
  switch (tok.getKind()) {
  default:
    diagnoseExpected(diag::expected_member_name_or_index_after,
                     isArrow ? "->" : ".");
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
    return parseTupleExpr();
  // literal
  //    null-literal
  case TokenKind::NullKw:
    expr = new (ctxt) NullLiteralExpr(consumeToken());
    break;
  //    integer-literal
  case TokenKind::IntegerLiteral: {
    StringRef str = tok.str();
    SourceLoc loc = consumeToken();
    expr = new (ctxt) IntegerLiteralExpr(str, loc);
    break;
  }
  //    floating-point-literal
  case TokenKind::FloatingPointLiteral: {
    StringRef str = tok.str();
    SourceLoc loc = consumeToken();
    expr = new (ctxt) FloatLiteralExpr(str, loc);
    break;
  }
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

ParserResult<Expr> Parser::parseTupleExpr() {
  assert(tok.is(TokenKind::LParen));
  SmallVector<Expr *, 4> exprs;
  SourceLoc lParen, rParen;
  bool success = parseTupleExpr(lParen, exprs, rParen);

  assert(rParen && "no rParenLoc!");

  // Create a ParenExpr or TupleExpr depending on the number of elements.
  Expr *expr = nullptr;
  if (exprs.size() == 1)
    expr = new (ctxt) ParenExpr(lParen, exprs[0], rParen);
  else
    expr = TupleExpr::create(ctxt, lParen, exprs, rParen);
  return makeParserResult(!success, expr);
}

/*
tuple-expression = '(' expression-list? ')'
expression-list = expression (',' expression)*
*/
bool Parser::parseTupleExpr(SourceLoc &lParenLoc,
                            SmallVectorImpl<Expr *> &exprs,
                            SourceLoc &rParenLoc) {
  assert(tok.is(TokenKind::LParen));
  lParenLoc = tok.getLoc();
  auto parseFn = [&](size_t k) -> bool {
    auto result = parseExpr(
        [&]() { diagnoseExpected(diag::expected_expr_after, k ? "," : "("); });

    if (result.hasValue())
      exprs.push_back(result.get());
    return result.hasValue();
  };

  return parseTuple(rParenLoc, parseFn,
                    diag::expected_rparen_at_end_of_tuple_expr);
}