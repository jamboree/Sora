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
  assert(tok.is(TokenKind::LParen));
  SmallVector<TypeRepr *, 4> elements;
  SourceLoc lParen, rParen;
  lParen = tok.getLoc();
  auto parseFn = [&](size_t k) -> bool {
    auto result = parseType(
        [&]() { diagnoseExpected(diag::expected_type_after, k ? "," : "("); });

    if (result.hasValue())
      elements.push_back(result.get());
    return result.hasValue();
  };
  bool success =
      parseTuple(rParen, parseFn, diag::expected_rparen_at_end_of_tuple_type);

  assert(rParen && "no rParenLoc!");

  // Create a TupleTypeRepr or ParenTypeRepr depending on the number of elements
  TypeRepr *type = nullptr;
  if (elements.size() == 1)
    type = new (ctxt) ParenTypeRepr(lParen, elements[0], rParen);
  else
    type = TupleTypeRepr::create(ctxt, lParen, elements, rParen);
  return makeParserResult(!success, type);
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