//===--- Parser.cpp ---------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Implementation of Parser methods which are not related to type, expr, stmt,
// decl and pattern parsing.
//===----------------------------------------------------------------------===//

#include "Sora/Parser/Parser.hpp"
#include "Sora/Lexer/Lexer.hpp"

using namespace sora;

const Token &Parser::peek() const { return lexer.peek(); }

SourceLoc Parser::consume() {
  assert(tok.getKind() != TokenKind::EndOfFile && "Consuming EOF!");
  SourceLoc loc = tok.getLoc();
  tok = lexer.lex();
  return loc;
}

bool Parser::isStartOfDecl() const {
  switch (tok.getKind()) {
  case TokenKind::LetKw:
  case TokenKind::FuncKw:
    return true;
  default:
    return false;
  }
}

bool Parser::isStartOfStmt() const {
  switch (tok.getKind()) {
  case TokenKind::IfKw:
  case TokenKind::ReturnKw:
  case TokenKind::WhileKw:
  case TokenKind::ContinueKw:
  case TokenKind::BreakKw:
    return true;
  default:
    return false;
  }
}

void Parser::skip() {
  auto kind = tok.getKind();
  consume();
  switch (tok.getKind()) {
  case TokenKind::LCurly:
    skipUntil(TokenKind::RCurly);
    consumeIf(TokenKind::RCurly);
    return;
  case TokenKind::LSquare:
    skipUntil(TokenKind::RSquare);
    consumeIf(TokenKind::RSquare);
    return;
  case TokenKind::LParen:
    skipUntil(TokenKind::RParen);
    consumeIf(TokenKind::RParen);
    return;
  default:
    return;
  }
}

void Parser::skipUntil(TokenKind kind) {
  while (tok.isNot(kind) && tok.isNot(TokenKind::EndOfFile))
    skip();
}

bool Parser::skipUntilDeclRCurly() {
  while (tok.isNot(TokenKind::EndOfFile)) {
    if (isStartOfDecl())
      return true;
    if (tok.is(TokenKind::RCurly))
      return false;
  }
  return false;
}

bool Parser::skipUntilDeclStmtRCurly() {
  while (tok.isNot(TokenKind::EndOfFile)) {
    if (isStartOfDecl())
      return true;
    if (isStartOfStmt())
      return true;
    if (tok.is(TokenKind::RCurly))
      return false;
  }
  return false;
}
