//===--- ParseType.cpp - Type Parsing Impl. ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Parser/Parser.hpp"

using namespace sora;

/*
type = identifier
     | tuple-type
     | reference-or-pointer-type
*/
ParserResult<TypeRepr>
Parser::parseType(const std::function<void()> &onNoType) {
  switch (tok.getKind()) {
  case TokenKind::Identifier: {
    Identifier ident;
    SourceLoc identLoc = consumeIdentifier(ident);
    return makeParserResult(new (ctxt) IdentifierTypeRepr(identLoc, ident));
  }
  case TokenKind::LParen:
    return parseTupleType();
  case TokenKind::Star:
  case TokenKind::Amp:
    return parseReferenceOrPointerType();
  default:
    onNoType();
    return nullptr;
  }
}

/*
tuple-type = '(' (type (',' type)*)? ')'
*/
ParserResult<TypeRepr> Parser::parseTupleType() {
  assert(tok.is(TokenKind::LParen) && "not a tuple type");
  // '('
  SourceLoc lParenLoc = consumeToken();

  /*Short path for empty tuple types*/
  // ')'
  if (SourceLoc rParenLoc = consumeIf(TokenKind::RParen))
    return makeParserResult(
        TupleTypeRepr::createEmpty(ctxt, lParenLoc, rParenLoc));

  SmallVector<TypeRepr *, 4> elements;

  // Utility function to try to recover and return something.
  auto tryRecoverAndReturn = [&]() -> ParserResult<TypeRepr> {
    skipUntilDeclStmtRCurly(TokenKind::LParen);
    if (!tok.is(TokenKind::LParen))
      return nullptr;
    SourceLoc rParenLoc = consumeToken();
    return makeParserErrorResult(
        TupleTypeRepr::create(ctxt, lParenLoc, elements, rParenLoc));
  };

  // type (',' type)*
  do {
    auto result = parseType([&]() { diagnoseExpected(diag::expected_type); });
    if (TypeRepr *type = result.getOrNull())
      elements.push_back(result.get());
    else
      return tryRecoverAndReturn();
  } while (consumeIf(TokenKind::Comma));

  // ')'
  SourceLoc rParenLoc = parseMatchingToken(
      lParenLoc, TokenKind::RParen, diag::expected_rparen_at_end_of_tuple_type);
  if (!rParenLoc)
    return nullptr;

  return makeParserResult(
      TupleTypeRepr::create(ctxt, lParenLoc, elements, rParenLoc));
}

/*
array-type = '[' type (';' expr)? ']'
*/
ParserResult<TypeRepr> Parser::parseArrayType() {
  llvm_unreachable("Currently, ArrayTypes are not supported by Sora");
  assert(tok.is(TokenKind::LSquare) && "not an array type");
  // '['
  SourceLoc lSquareLoc = consumeToken();

  // type
  auto subTypeResult =
      parseType([&]() { diagnoseExpected(diag::expected_type); });
  if (subTypeResult.isNull())
    return nullptr;
  TypeRepr *subTyRepr = subTypeResult.get();

  /// (';' expr)?
  Expr *sizeExpr = nullptr;
  if (consumeIf(TokenKind::Semicolon)) {
    auto result = parseExpr([&]() { diagnoseExpected(diag::expected_expr); });
    if (result.isNull())
      return nullptr;

    sizeExpr = result.get();
  }

  /// ']'
  SourceLoc rSquareLoc = parseMatchingToken(lSquareLoc, TokenKind::RSquare);
  if (!rSquareLoc)
    return nullptr;

  return makeParserResult(
      new (ctxt) ArrayTypeRepr(lSquareLoc, subTyRepr, sizeExpr, rSquareLoc));
}

/*
reference-or-pointer-type = ('&' | '*') "mut"? type
*/
ParserResult<TypeRepr> Parser::parseReferenceOrPointerType() {
  assert(tok.isAny(TokenKind::Amp, TokenKind::Star) &&
         "not a reference or pointer type");
  // ('&' | '*')
  bool isReference = false;
  SourceLoc signLoc;
  if (SourceLoc ampLoc = consumeIf(TokenKind::Amp)) {
    isReference = true;
    signLoc = ampLoc;
  }
  else
    signLoc = consume(TokenKind::Star);
  assert(signLoc && "no signLoc");

  /// "mut"?
  bool isMut = false;
  SourceLoc mutLoc;
  if (SourceLoc kwLoc = consumeIf(TokenKind::MutKw))
    mutLoc = kwLoc;

  /// Parse the type
  auto result = parseType([&]() { diagnoseExpected(diag::expected_type); });
  if (result.isNull())
    return nullptr;

  return makeParserResult(
      new (ctxt) PointerTypeRepr(signLoc, isReference, mutLoc, result.get()));
}