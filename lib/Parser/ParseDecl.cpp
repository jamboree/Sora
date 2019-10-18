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
  // Must handle everything that parseDecl handles.
  switch (tok.getKind()) {
  case TokenKind::LetKw:
  case TokenKind::FuncKw:
    return true;
  default:
    return false;
  }
}

/*
declaration =  top-level-declaration | let-declaration
top-level-declaration = function-declaration
*/
ParserResult<Decl> Parser::parseDecl(bool isTopLevel) {
  assert(isStartOfDecl() && "not a decl");
  switch (tok.getKind()) {
  case TokenKind::LetKw:
    return parseLetDecl();
  case TokenKind::FuncKw:
    return parseFuncDecl();
  default:
    llvm_unreachable("unknown start of decl");
  }
}

/*
source-file = top-level-declaration+
top-level-declaration = function-declaration
*/
void Parser::parseSourceFile() {
  auto recover = [&]() { skipUntilDecl(); };
  // Once parsing of a member is done, adds to it to the source file and checks
  // that we got a newline after it.
  auto addMember = [&](Decl *decl) {
    // Add it to the file
    sourceFile.addMember(decl);
    // Check that we got a newline after the declaration, else, complain.
    if (!tok.isAtStartOfLine() && tok.isNot(TokenKind::EndOfFile))
      diagnose(prevTokPastTheEnd, diag::no_newline_after_decl);
  };
  while (!isEOF()) {
    switch (tok.getKind()) {
    // function-declaration
    case TokenKind::FuncKw: {
      if (FuncDecl *func = parseFuncDecl().getOrNull())
        addMember(func);
      else
        recover();
      break;
    }
    // Trying to declare global variables is a relatively common mistake, so
    // parse them but ignore them & inform the user that it's not allowed.
    case TokenKind::LetKw: {
      // FIXME: Should this be emitted only on successful parsing of the
      // LetDecl?
      diagnose(tok, diag::let_not_allowed_at_global_level);
      if (parseLetDecl().isParseError())
        recover();
      break;
    }
    default:
      if (tok.isNot(TokenKind::Unknown))
        diagnose(tok, diag::expected_fn_decl);
      recover();
      break;
    }
  }
}

/*
let-declaration = "let" pattern ('=' expression)?
*/
ParserResult<Decl> Parser::parseLetDecl() {
  assert(tok.is(TokenKind::LetKw));
  // "let"
  SourceLoc letLoc = consumeToken();
  // pattern
  auto result =
      parsePattern([&]() { diagnoseExpected(diag::expected_pat_after_let); });

  if (!result.hasValue())
    return nullptr;

  Pattern *pattern = result.get();

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
  // When we don't have the ')', but we have a -> or {, recover by returning the
  // parameter list, using the SourceLoc of the lParen or the end of the last
  // Parameter as the RParenLoc.
  if (tok.isAny(TokenKind::Arrow, TokenKind::LCurly))
    return makeParserErrorResult(ParamList::create(
        ctxt, lParenLoc, params,
        params.empty() ? lParenLoc : params.back()->getEndLoc()));
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
  if (!tok.isIdentifier()) {
    diagnoseExpected(diag::expected_ident_in_fn);
    return nullptr;
  }

  Identifier identifier;
  SourceLoc identifierLoc = consumeIdentifier(identifier);

  // parameter-declaration-list
  ParamList *paramList = nullptr;
  if (tok.isNot(TokenKind::LParen)) {
    diagnoseExpected(diag::expected_lparen_in_fn_arg_list);
    // If we can parse something else (the user just forgot the parameter list),
    // just leave the paramList to nullptr and keep going, else stop because
    // this doesn't really look like a function.
    if (tok.isNot(TokenKind::Arrow, TokenKind::LCurly))
      return nullptr;
  }

  auto paramListResult = parseParamDeclList();
  bool hadParseError = paramListResult.isParseError();
  if (paramListResult.isNull())
    return nullptr;
  paramList = paramListResult.get();

  // ("->" type)?
  TypeLoc retTL;
  if (consumeIf(TokenKind::Arrow)) {
    auto result =
        parseType([&]() { diagnoseExpected(diag::expected_fn_ret_ty); });
    hadParseError |= result.isParseError();
    if (result.isNull())
      return nullptr;
    retTL = TypeLoc(result.get());
  }

  // Now that we have enough elements, create the node.
  auto node = new (ctxt)
      FuncDecl(declContext, fnLoc, identifierLoc, identifier, paramList, retTL);

  if (tok.isNot(TokenKind::LCurly)) {
    // Try to find the LCurly on the same line or at the start of the next line.
    skipUntilTokOrNewline(TokenKind::LCurly);
    // Check if we found our LCurly
    if (tok.isNot(TokenKind::LCurly)) {
      if (!hadParseError)
        diagnoseExpected(diag::expected_lcurly_fn_body);
      return nullptr;
    }
  }

  // block-statement
  auto bodyDC = setDeclContextRAII(node);
  auto body = parseBlockStmt();
  if (body.isNull())
    return nullptr;
  node->setBody(body.get());

  return makeParserResult(node);
}