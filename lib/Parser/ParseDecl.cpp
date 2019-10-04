//===--- ParseDecl.cpp - Declaration Parsing Impl. --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"
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

/*
source-file = top-level-declaration+
top-level-declaration = function-declaration
                      | type-declaration
                      | struct-declaration
*/
void Parser::parseSourceFile() {
  while (!isEOF()) {
    if (tok.isNot(TokenKind::FuncKw)) {
      if (tok.isNot(TokenKind::Unknown))
        diagnose(tok, diag::expected_fn_decl);
      skipUntil(TokenKind::FuncKw);
      continue;
    }
    auto result = parseFuncDecl();
    if (result.isNull()) {
      skipUntil(TokenKind::FuncKw);
      continue;
    }
    sourceFile.addMember(result.get());
  }
}

/*
let-declaration = "let" pattern ('=' expression)?
*/
ParserResult<Decl> Parser::parseLetDecl(SmallVectorImpl<VarDecl *> &vars) {
  assert(tok.is(TokenKind::LetKw));
  // "let"
  SourceLoc letLoc = consumeToken();
  // pattern
  auto result =
      parsePattern([&]() { diagnoseExpected(diag::expected_pat_after_let); });

  if (!result.hasValue())
    return nullptr;

  Pattern *pattern = result.get();
  pattern->forEachVarDecl([&](VarDecl *var) { vars.push_back(var); });

  // ('=' expression)?
  bool hadError = false;
  if (SourceLoc eqLoc = consumeIf(TokenKind::Equal)) {
    auto result = parseExpr(
        [&]() { diagnoseExpected(diag::expected_initial_value_after_equal); });
    if (result.hasValue()) {
      return makeParserResult(new (ctxt) LetDecl(declContext, letLoc, pattern,
                                                 eqLoc, result.get()));
    }
    hadError = true;
  }
  LetDecl *decl = new (ctxt) LetDecl(declContext, letLoc, pattern);
  return hadError ? makeParserErrorResult(decl) : makeParserResult(decl);
}

/*
identifier ':' type
*/
ParserResult<ParamDecl> Parser::parseParamDecl() {
  assert(tok.is(TokenKind::Identifier));
  // identifier
  Identifier ident;
  SourceLoc identLoc = consumeIdentifier(ident);
  auto paramDecl = new (ctxt) ParamDecl(declContext, identLoc, ident, {});
  //  ':'
  if (!consumeIf(TokenKind::Colon)) {
    diagnoseExpected(diag::expected_colon);
    return makeParserErrorResult(paramDecl);
  }
  // type
  {
    auto result = parseType([&]() { diagnoseExpected(diag::expected_type); });
    if (result.isNull())
      return nullptr;
    paramDecl->getTypeLoc() = TypeLoc(result.get());
  }
  return makeParserResult(paramDecl);
}

/*
parameter-declaration-list
  = '(' parameter-declaration (',' parameter-declaration)* ')'
  | '(' ')'
*/
ParserResult<ParamList> Parser::parseParamDeclList() {
  assert(tok.is(TokenKind::LParen) && "not a param list!");
  // '('
  SourceLoc lParenLoc = consumeToken();
  // ')'
  if (SourceLoc rParenLoc = consumeIf(TokenKind::RParen))
    return makeParserResult(ParamList::createEmpty(ctxt, lParenLoc, rParenLoc));
  // parameter-declaration (',' parameter-declaration)*
  SmallVector<ParamDecl *, 4> params;
  bool complainedInLastIteration = false;
  parseList([&](unsigned idx) {
    if (tok.is(TokenKind::RParen))
      return false;
    if (tok.isNot(TokenKind::Identifier)) {
      diagnoseExpected(diag::expected_param_decl);
      complainedInLastIteration = true;
      return false;
    }
    auto result = parseParamDecl();
    complainedInLastIteration = result.isParseError();
    if (!result.hasValue())
      return false;
    params.push_back(result.get());
    return true;
  });
  // ')'
  SourceLoc rParenLoc;
  // If we already complained in the last iteration, don't emit missing RParen
  // errors.
  if (complainedInLastIteration)
    rParenLoc = consumeIf(TokenKind::RParen);
  else
    rParenLoc = parseMatchingToken(lParenLoc, TokenKind::RParen);
  // If we found the ')', create our ParamList & return.
  if (rParenLoc)
    return makeParserResult(
        ParamList::create(ctxt, lParenLoc, params, rParenLoc));
  // When we don't have the ')', but we have a -> or {, recover by returning an
  // empty parameter list.
  if (tok.isAny(TokenKind::Arrow, TokenKind::LCurly))
    // FIXME: It isn't ideal to have the lParenLoc == rParenLoc in the ParamList
    // but that's the best I can do currently.
    return makeParserErrorResult(
        ParamList::createEmpty(ctxt, lParenLoc, lParenLoc));
  // Else just return an error.
  return nullptr;
}

/*
function-declaration = "func" identifier parameter-declaration-list
                      ("->" type)? block-statement
*/
ParserResult<FuncDecl> Parser::parseFuncDecl() {
  assert(tok.is(TokenKind::FuncKw) && "Not a func!");
  SourceLoc fnLoc = consumeToken();

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
    auto paramListResult = parseParamDeclList();
    if (paramListResult.isNull())
      return nullptr;
    paramList = paramListResult.get();
  }
  else {
    diagnoseExpected(diag::expected_lparen_in_fn_arg_list);
    // If we can parse something else, just leave the paramList to nullptr and
    // keep going, else return directly.
    if (tok.isNot(TokenKind::Arrow, TokenKind::LCurly))
      return nullptr;
  }

  // ("->" type)?
  TypeLoc retTL;
  if (consumeIf(TokenKind::Arrow)) {
    auto result =
        parseType([&]() { diagnoseExpected(diag::expected_fn_ret_ty); });
    if (result.isNull())
      return nullptr;
    retTL = TypeLoc(result.get());
  }

  auto node = new (ctxt)
      FuncDecl(declContext, fnLoc, identifierLoc, identifier, paramList, retTL);

  // block-statement
  if (tok.is(TokenKind::LCurly)) {
    // Set the DeclContext for the parsing of the body.
    auto bodyDC = setDeclContextRAII(node);
    // Parse the body
    auto result = parseBlockStmt();
    if (result.isNull())
      return nullptr;
    node->setBody(result.get());
  }
  else {
    diagnoseExpected(diag::expected_lcurly_fn_body);
    return nullptr;
  }

  return makeParserResult(node);
}