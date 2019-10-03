//===--- ParsePattern.cpp - Pattern Parsing Impl. ---------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/Parser/Parser.hpp"
#include "llvm/ADT/DenseMap.h"

using namespace sora;

namespace {
/// Handles the application of a "mut" specifier, diagnosting redudant
/// mutability in the process.
class ApplyMutSpecifier : public ASTWalker {
public:
  /// Maps a VarDecl to the SourceLoc of the mut specifier that made it mutable.
  /// FIXME: If I ever store the MutLoc inside VarDecls, this can go away and
  /// the code of this class simplified.
  llvm::DenseMap<VarDecl *, SourceLoc> appliedMuts;

  ApplyMutSpecifier(Parser &parser) : parser(parser) {}

  Parser &parser;

  void makeMutable(VarDecl *var, SourceLoc mutLoc) {
    assert(mutLoc);
    var->setIsMutable();
    appliedMuts[var] = mutLoc;
  }

  SourceLoc getOriginalMutLoc(VarDecl *var) {
    assert(var->isMutable() && "only for mut vars");
    SourceLoc loc = appliedMuts[var];
    assert(loc && "no applied mut?");
    return loc;
  }

  // Walk from outer-in (pre-order)
  Action walkToPatternPre(Pattern *pattern) override {
    // FIXME: This walks the whole subtree of each MutPattern, so it's not the
    // most efficient thing there is. However it shouldn't be a problem unless
    // we have hundred of muts in a single pattern, but can this function be
    // more efficient?

    // We only care about "mut" patterns
    if (MutPattern *mutPat = dyn_cast<MutPattern>(pattern)) {
      mutPat->forEachVarDecl([&](VarDecl *var) {
        if (var->isMutable()) {
          parser.diagnose(var->getLoc(), diag::var_already_mutable)
              .fixitRemove(mutPat->getMutLoc());
          SourceLoc noteLoc = getOriginalMutLoc(var);
          parser.diagnose(noteLoc, diag::var_made_mutable_here)
              .highlight(var->getLoc());
        }
        else
          makeMutable(var, mutPat->getMutLoc());
      });
    }
    return Action::Continue;
  }
};
} // namespace

/*
pattern = "mut" (tuple-pattern | identifier | '_') (':' type)?
*/
ParserResult<Pattern> Parser::parsePattern(llvm::function_ref<void()> onNoPat) {
  Pattern *pattern = nullptr;

  // "mut"
  SourceLoc mutLoc = consumeIf(TokenKind::MutKw);
  if (mutLoc && tok.is(TokenKind::MutKw)) {
    SourceLoc firstExtraMut = mutLoc;
    SourceLoc lastExtraMut = mutLoc;
    // Consume the extra "mut" specifiers
    while (SourceLoc extraMut = consumeIf(TokenKind::MutKw)) {
      lastExtraMut = mutLoc;
      mutLoc = extraMut;
    }
    // Diagnose the extra "mut"s
    diagnose(mutLoc, diag::pat_can_only_have_one_mut)
        .fixitRemove({firstExtraMut, lastExtraMut});
  }
  bool isMutable = mutLoc.isValid();

  // (tuple-pattern | identifier | '_')
  {
    // We want to apply the "mut" specifier only once and from outer-in, so
    // disable the application of the "mut" specifier when parsing the
    // subpattern of a mut-pattern so we can handle it ourselves later.
    auto disableMutApplication =
        llvm::SaveAndRestore<bool>(canApplyMutSpecifier, !isMutable);

    switch (tok.getKind()) {
    default:
      // If we had a "mut" qualifier, we expected a pattern. If we didn't have
      // one, we just didn't find anything.
      if (isMutable)
        diagnoseExpected(diag::expected_pat_after_mut);
      else
        onNoPat();
      return nullptr;
    // tuple-pattern
    case TokenKind::LParen: {
      auto result = parseTuplePattern();
      if (!result.hasValue())
        return nullptr;
      pattern = result.get();
      break;
    }
    // identifier
    case TokenKind::Identifier: {
      Identifier ident;
      SourceLoc identLoc = consumeIdentifier(ident);
      VarDecl *var = new (ctxt) VarDecl(declContext, identLoc, ident);
      pattern = new (ctxt) VarPattern(var);
      break;
    }
    // '_'
    case TokenKind::UnderscoreKw:
      pattern = new (ctxt) DiscardPattern(consumeToken());
      break;
    }
  }

  // Wrap our pattern inside a MutPattern if we have a mut pattern.
  if (isMutable)
    pattern = new (ctxt) MutPattern(mutLoc, pattern);

  // Apply the "mut" specifier if we're allowed to.
  if (canApplyMutSpecifier)
    pattern->walk(ApplyMutSpecifier(*this));

  // Parse the type annotation if present
  // (':' type)?
  if (consumeIf(TokenKind::Colon)) {
    auto result = parseType([&]() { diagnoseExpected(diag::expected_type); });
    // FIXME: Can we recover better here?
    if (!result.hasValue())
      return nullptr;
    pattern = new (ctxt) TypedPattern(pattern, result.get());
  }
  return makeParserResult(pattern);
}

/*
tuple-pattern = '(' (pattern (',' pattern)*)? ')'
*/
ParserResult<Pattern> Parser::parseTuplePattern() {
  assert(tok.is(TokenKind::LParen) && "not a tuple-pattern!");
  // '('
  SourceLoc lParenLoc = consumeToken();

  /*Short path for empty tuple patterns*/
  // ')'
  if (SourceLoc rParenLoc = consumeIf(TokenKind::RParen))
    return makeParserResult(
        TuplePattern::createEmpty(ctxt, lParenLoc, rParenLoc));

  SmallVector<Pattern *, 4> elements;

  // Utility function to create a ParenPattern or a TuplePattern depending on
  // the number of elements.
  auto createResult = [&](SourceLoc rParenLoc) -> Pattern * {
    if (elements.size() == 1)
      return new (ctxt) ParenPattern(lParenLoc, elements[0], rParenLoc);
    return TuplePattern::create(ctxt, lParenLoc, elements, rParenLoc);
  };

  // Utility function to try to recover and return something.
  auto tryRecoverAndReturn = [&]() -> ParserResult<Pattern> {
    skipUntilDeclStmtRCurly(TokenKind::LParen);
    if (!tok.is(TokenKind::LParen))
      return nullptr;
    SourceLoc rParenLoc = consumeToken();
    return makeParserErrorResult(createResult(rParenLoc));
  };

  // pattern (',' pattern)
  do {
    auto result = parsePattern([&]() { diagnoseExpected(diag::expected_pat); });
    if (Pattern *type = result.getOrNull())
      elements.push_back(result.get());
    else
      return tryRecoverAndReturn();
  } while (consumeIf(TokenKind::Comma));

  // ')'
  SourceLoc rParenLoc = parseMatchingToken(
      lParenLoc, TokenKind::RParen, diag::expected_rparen_at_end_of_tuple_pat);
  if (!rParenLoc)
    return nullptr;
  return makeParserResult(createResult(rParenLoc));
}