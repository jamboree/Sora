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
  return nullptr;
}

/*
prefix-expression = prefix-operator prefix-expression
                  | postfix-expression
*/
ParserResult<Expr>
Parser::parsePrefixExpr(llvm::function_ref<void()> onNoExpr) {
  return nullptr;
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
       | member-access-expression
       | array-subscript
       | postfix-operator
*/
ParserResult<Expr>
Parser::parsePostfixExpr(llvm::function_ref<void()> onNoExpr) {
  return nullptr;
}

/*
primary-expression = identifier | literal | tuple-expression | '_'
literal = null-literal | integer-literal | floating-point-literal |
          boolean-literal
*/
ParserResult<Expr>
Parser::parsePrimaryExpr(llvm::function_ref<void()> onNoExpr) {
  return nullptr;
}