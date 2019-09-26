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
  // TODO
  skip();
  return makeParserResult(BlockStmt::createEmpty(ctxt, {}, {}));
}