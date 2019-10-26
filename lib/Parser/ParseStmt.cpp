//===--- ParseStmt.cpp - Statement Parsing Impl. ----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Stmt.hpp"
#include "Sora/Parser/Parser.hpp"

using namespace sora;

bool Parser::isStartOfStmt() const {
  // Must handle everything that parseStmt handles.
  switch (tok.getKind()) {
  case TokenKind::LCurly:
  case TokenKind::IfKw:
  case TokenKind::WhileKw:
  case TokenKind::ReturnKw:
  case TokenKind::ContinueKw:
  case TokenKind::BreakKw:
    return true;
  default:
    return false;
  }
}

/*
statement =
  | block-statement
  | if-statement
  | while-statement
  | control-transfer-statement

control-transfer-statement =
  | continue-statement
  | break-statement
  | return-statement
*/
ParserResult<Stmt> Parser::parseStmt() {
  assert(isStartOfStmt() && "not a stmt");
  switch (tok.getKind()) {
  case TokenKind::LCurly:
    return parseBlockStmt();
  case TokenKind::IfKw:
    return parseIfStmt();
  case TokenKind::WhileKw:
    return parseWhileStmt();
  case TokenKind::ReturnKw:
    return parseReturnStmt();
  // continue-statement = "continue"
  case TokenKind::ContinueKw:
    return makeParserResult(new (ctxt) ContinueStmt(consumeToken()));
  // break-statement = "break"
  case TokenKind::BreakKw:
    return makeParserResult(new (ctxt) BreakStmt(consumeToken()));
  default:
    llvm_unreachable("unknown start of stmt");
  }
}

/*
block-statement = '{' block-statement-item* '}'
block-statement-item = expression | statement | declaration
*/
ParserResult<BlockStmt> Parser::parseBlockStmt() {
  assert(tok.is(TokenKind::LCurly) && "not a block stmt!");
  // '{'
  SourceLoc lCurly = consumeToken();

  // block-statement-item*
  SmallVector<ASTNode, 16> elements;

  while (!isEOF() && tok.isNot(TokenKind::RCurly)) {
    // Always skip 'unknown' tokens
    if (tok.is(TokenKind::Unknown))
      consumeToken();

    // If the next token doesn't begin a line and we already have elements,
    // diagnose a missing newline.
    if (!elements.empty() && !tok.isAtStartOfLine())
      diagnose(prevTokPastTheEnd, diag::expected_newline);

    // declaration
    if (isStartOfDecl()) {
      if (Decl *decl = parseDecl().getOrNull())
        elements.push_back(decl);
      else
        skipUntilTokDeclStmtRCurly();
    }
    // statement
    else if (isStartOfStmt()) {
      if (Stmt *stmt = parseStmt().getOrNull())
        elements.push_back(stmt);
      else
        skipUntilTokDeclStmtRCurly();
    }
    // expression
    else {
      // FIXME: Is it a good diagnostic?
      auto result =
          parseExpr([&]() { diagnoseExpected(diag::expected_block_item); });
      if (Expr *expr = result.getOrNull())
        elements.push_back(expr);
      else
        skipUntilTokDeclStmtRCurly();
    }
  }

  // '}'
  SourceLoc rCurly = parseMatchingToken(lCurly, TokenKind::RCurly);
  // If we couldn't find the rCurly and we got no elements, return an error,
  // else, use prevTokPastTheEnd as } loc to recover.
  // FIXME: Is this the right thing to do?
  if (rCurly.isInvalid()) {
    if (elements.empty())
      return nullptr;
    rCurly = prevTokPastTheEnd;
  }
  return makeParserResult(BlockStmt::create(ctxt, lCurly, elements, rCurly));
}

/*
return-statement = "return" expression?
*/
ParserResult<Stmt> Parser::parseReturnStmt() {
  // "return"
  assert(tok.is(TokenKind::ReturnKw));
  SourceLoc retLoc = consumeToken();

  ReturnStmt *ret = new (ctxt) ReturnStmt(retLoc);

  // expression?
  if (tok.isAny(TokenKind::EndOfFile, TokenKind::RCurly) || isStartOfStmt() ||
      isStartOfDecl())
    return makeParserResult(ret);

  // Fetch the beg SourceLoc from the Expr here (so we don't have to include
  // Expr.hpp just for ->getBegLoc())
  SourceLoc exprBeg = tok.getLoc();
  auto result =
      parseExpr([&]() { diagnoseExpected(diag::expected_expr_return); });
  if (!result.hasValue())
    return makeParserErrorResult(ret);

  Expr *expr = result.get();
  ret->setResult(expr);
  // Warn about code like this because it's visually ambiguous
  /*
    return
    0
  */
  // Warn when the expression has the same indent (or less) as the return.
  if (getColumnDifference(retLoc, exprBeg) <= 0) {
    diagnose(exprBeg, diag::unindented_expr_after_ret);
    diagnose(exprBeg, diag::indent_expr_to_silence);
  }

  return makeParserResult(ret);
}

/*
if-statement = "if" condition block-statement
            ("else" (block-statement | if-statement))?
*/
ParserResult<Stmt> Parser::parseIfStmt() {
  // "if"
  assert(tok.is(TokenKind::IfKw));
  SourceLoc ifLoc = consumeToken();

  // condition
  StmtCondition cond;
  parseCondition(cond, "if");
  if (!cond)
    return nullptr;

  // block-statement
  if (!cond && tok.isNot(TokenKind::LCurly))
    return nullptr;

  if (tok.isNot(TokenKind::LCurly)) {
    diagnoseExpected(diag::expected_lcurly_after_cond, "if");
    return nullptr;
  }

  BlockStmt *then = parseBlockStmt().getOrNull();
  if (!then)
    return nullptr;

  // ("else" (block-statement | if-statement))?
  if (tok.isNot(TokenKind::ElseKw))
    return makeParserResult(new (ctxt) IfStmt(ifLoc, cond, then));

  SourceLoc elseLoc = consumeToken();

  Stmt *elseStmt = nullptr;
  switch (tok.getKind()) {
  default:
    diagnoseExpected(diag::expected_lcurly_or_if_after_else);
    break;
  case TokenKind::LCurly:
    elseStmt = parseBlockStmt().getOrNull();
    break;
  case TokenKind::IfKw:
    elseStmt = parseIfStmt().getOrNull();
    break;
  }

  if (!elseStmt)
    return nullptr;

  return makeParserResult(new (ctxt)
                              IfStmt(ifLoc, cond, then, elseLoc, elseStmt));
}

/*
while-statement = "while" condition block-statement
*/
ParserResult<Stmt> Parser::parseWhileStmt() {
  // "while"
  assert(tok.is(TokenKind::WhileKw));
  SourceLoc whileLoc = consumeToken();

  // condition
  StmtCondition cond;
  parseCondition(cond, "while");
  if (!cond)
    return nullptr;

  // block-statement
  if (!cond && tok.isNot(TokenKind::LCurly))
    return nullptr;

  if (tok.isNot(TokenKind::LCurly)) {
    diagnoseExpected(diag::expected_lcurly_after_cond, "while");
    return nullptr;
  }

  BlockStmt *body = parseBlockStmt().getOrNull();
  if (!body)
    return nullptr;
  return makeParserResult(new (ctxt) WhileStmt(whileLoc, cond, body));
}

/*
condition = expression | let-declaration
*/
bool Parser::parseCondition(StmtCondition &cond, StringRef name) {
  // let-declaration
  if (tok.is(TokenKind::LetKw)) {
    auto result = parseLetDecl();
    cond = result.getOrNull();
    return result.isParseError();
  }
  // expression
  auto result =
      parseExpr([&]() { diagnoseExpected(diag::expected_cond_in, name); });
  cond = result.getOrNull();
  return result.isParseError();
}