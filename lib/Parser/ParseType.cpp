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
  | reference-type
  | maybe-type
*/
ParserResult<TypeRepr> Parser::parseType(llvm::function_ref<void()> onNoType) {
  switch (tok.getKind()) {
  default:
    onNoType();
    return nullptr;
  case TokenKind::Identifier: {
    Identifier ident;
    SourceLoc identLoc = consumeIdentifier(ident);
    return makeParserResult(new (ctxt) IdentifierTypeRepr(identLoc, ident));
  }
  case TokenKind::LParen:
    return parseTupleType();
  case TokenKind::Star:
  case TokenKind::Amp:
    return parseReferenceType();
  case TokenKind::MaybeKw:
    return parseMaybeType();
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

  // Utility function to create a ParenTypeRepr or a TupleTypeRepr depending on
  // the number of elements.
  auto createResult = [&](SourceLoc rParenLoc) -> TypeRepr * {
    if (elements.size() == 1)
      return new (ctxt) ParenTypeRepr(lParenLoc, elements[0], rParenLoc);
    return TupleTypeRepr::create(ctxt, lParenLoc, elements, rParenLoc);
  };

  // Utility function to try to recover and return something.
  auto tryRecoverAndReturn = [&]() -> ParserResult<TypeRepr> {
    skipUntilTokDeclStmtRCurly(TokenKind::LParen);
    if (!tok.is(TokenKind::LParen))
      return nullptr;
    SourceLoc rParenLoc = consumeToken();
    return makeParserErrorResult(createResult(rParenLoc));
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
  return makeParserResult(createResult(rParenLoc));
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
reference-type = '&' "mut"? type
*/
ParserResult<TypeRepr> Parser::parseReferenceType() {
  assert(tok.is(TokenKind::Amp) && "not a reference type");
  // '&'
  SourceLoc ampLoc = consume(TokenKind::Amp);

  /// "mut"?
  SourceLoc mutLoc;
  if (SourceLoc kwLoc = consumeIf(TokenKind::MutKw))
    mutLoc = kwLoc;

  /// Parse the type
  auto result = parseType([&]() { diagnoseExpected(diag::expected_type); });
  if (result.isNull())
    return nullptr;

  return makeParserResult(new (ctxt)
                              ReferenceTypeRepr(ampLoc, mutLoc, result.get()));
}

/*
maybe-type = "maybe" type
*/
ParserResult<TypeRepr> Parser::parseMaybeType() {
  assert(tok.is(TokenKind::MaybeKw) && "not a maybe type");
  SourceLoc maybeLoc = consumeToken();

  auto result = parseType([&]() { diagnoseExpected(diag::expected_type); });
  if (result.isNull())
    return nullptr;

  return makeParserResult(new (ctxt) MaybeTypeRepr(maybeLoc, result.get()));
}