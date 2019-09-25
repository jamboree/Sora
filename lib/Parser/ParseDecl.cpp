//===--- ParseDecl.cpp - Declaration Parsing Impl. --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Parser/Parser.hpp"

using namespace sora;

bool Parser::isStartOfDecl() const {
  switch (tok.getKind()) {
  case TokenKind::LetKw:
  case TokenKind::FuncKw:
    return true;
  default:
    return false;
  }
}

ParserResult<ParamDecl> Parser::parseParamDecl() {
  return makeParserErrorResult<ParamDecl>();
}

ParserResult<ParamList> Parser::parseParamDeclList() {
  return makeParserErrorResult<ParamList>();
}

/*
function-declaration = "func" identifier parameter-declaration-list
                      ("->" type)? block-statement
*/
ParserResult<FuncDecl> Parser::parseFuncDecl() {
  assert(tok.is(TokenKind::FuncKw) && "Not a func!");
  consumeToken();

  // identifier
  Identifier identifier;
  SourceLoc identifierLoc;
  if (tok.isIdentifier())
    identifierLoc = consumeIdentifier(identifier);
  else {
    diagnoseExpected(diag::expected_ident_in_fn);
    return nullptr;
  }

  // parameter-declaration-list
  ParamList *paramList = nullptr;
  if (tok.is(TokenKind::LParen)) {
    if (auto paramListResult = parseParamDeclList())
      paramList = paramListResult.get();
    else
      return nullptr;
  }
  else {
    diagnoseExpected(diag::expected_lparen_in_fn_arg_list);
    return nullptr;
  }

  // "->" type
  TypeLoc retTL;
  if (consumeIf(TokenKind::Arrow)) {
    auto result =
        parseType([&]() { diagnoseExpected(diag::expected_fn_ret_ty); });
    if (result)
      retTL = TypeLoc(result.get());
    else
      return nullptr;
  }

  // TODO: Body parsing.

  return makeParserErrorResult<FuncDecl>();
}