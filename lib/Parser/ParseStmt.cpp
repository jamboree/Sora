//===--- ParseStmt.cpp - Statement Parsing Impl. ----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Stmt.hpp"
#include "Sora/Parser/Parser.hpp"

using namespace sora;

/*
statement =
  | block-statement // starts with '{'
  | if-statement // starts with 'if'
  | while-statement // starts with 'while'
  | do-while-statement // starts with 'do'
  | control-transfer-statement

control-transfer-statement =
  | continue-statement
  | break-statement
  | return-statement
*/
bool Parser::isStartOfStmt() const {
  switch (tok.getKind()) {
  case TokenKind::RCurly:
  case TokenKind::IfKw:
  case TokenKind::WhileKw:
  case TokenKind::DoKw:
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
  | do-while-statement
  | control-transfer-statement

control-transfer-statement =
  | continue-statement
  | break-statement
  | return-statement
*/
ParserResult<Stmt> Parser::parseStmt() {
  assert(isStartOfStmt() && "not a stmt");
  // NOTE: This must handle everything that isStartOfStmt handles.
  return nullptr;
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
      SmallVector<VarDecl *, 4> vars;
      if (Decl *decl = parseDecl(vars).getOrNull()) {
        elements.push_back(decl);
        elements.append(vars.begin(), vars.end());
      }
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
  if (!rCurly) {
    if (elements.empty())
      return nullptr;
    rCurly = prevTokPastTheEnd;
  }
  return makeParserResult(BlockStmt::create(ctxt, lCurly, elements, rCurly));
}