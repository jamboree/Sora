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

ParserResult<BlockStmt> Parser::parseBlockStmt() {
  assert(tok.is(TokenKind::LCurly) && "not a block stmt!");
  SourceLoc lCurly = consumeToken();
  // As a hack, parse a let-declaration if there's one in the body. This is just
  // for parser testing.
  if (tok.is(TokenKind::LetKw)) {
    // consume the { and parse the letdecl
    SourceLoc lCurlyLoc = consumeToken();
    SmallVector<ASTNode, 8> elements;
    SmallVector<VarDecl *, 4> vars;
    auto let = parseLetDecl(vars);
    if (let.hasValue()) {
      elements.push_back(let.get());
      for (VarDecl *var : vars)
        elements.push_back(var);
    }
    // FIXME: better recovery - try to find the }
    SourceLoc rCurlyLoc = parseMatchingToken(lCurlyLoc, TokenKind::RCurly);
    if (!rCurlyLoc)
      return nullptr;
    return makeParserResult(
        BlockStmt::create(ctxt, lCurlyLoc, elements, rCurlyLoc));
  }
  if (SourceLoc rCurly = parseMatchingToken(lCurly, TokenKind::RCurly))
    return makeParserResult(BlockStmt::createEmpty(ctxt, lCurly, rCurly));
  return nullptr;
}