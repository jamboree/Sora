//===--- Lexer.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Implementation of the Lexer & Token classes.
//===----------------------------------------------------------------------===//

#include "Sora/Lexer/Lexer.hpp"
#include "Sora/Common/DiagnosticsLexer.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Lexer/Token.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

StringRef sora::to_string(TokenKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown TokenKind");
#define TOKEN(KIND)                                                            \
  case TokenKind::KIND:                                                        \
    return #KIND;
#include "Sora/Lexer/TokenKinds.def"
  }
}
StringRef Token::str() const { return charRange.str(); }

void Token::dump() const { dump(llvm::outs()); }

void Token::dump(raw_ostream &out) const {
  // Dump the token like this: 'return' - ReturnKw
  out << "'" << str() << "' - " << to_string(kind);
  if (startOfLine)
    out << " [startOfLine]";
}

void Lexer::init(StringRef str) {
  beg = cur = str.begin();
  end = str.end();
  nextToken = Token();
  // Load a codepoint by advancing at least once
  // Lex the first token
  doLex();
}

void Lexer::init(BufferID id) { init(srcMgr.getBufferStr(id)); }

Token Lexer::lex() {
  auto tok = nextToken;
  // Lex if we haven't reached EOF
  if (tok.isNot(TokenKind::EndOfFile))
    doLex();
  return tok;
}

Token Lexer::peek() const { return nextToken; }

uint32_t Lexer::advance(bool *isSourceExhaustedError) {
  if (cur >= end)
    return invalidCP;
  // fast-path for non-UTF8 characters
  char curChar = *cur;
  if (!(curChar & 0x80)) {
    ++cur;
    return curChar;
  }
  // use llvm::convertUTF8Sequence to fetch the codepoint
  // FIXME: I think this could be cleaner.
  uint32_t cp;
  auto oldCur = cur;
  auto result = llvm::convertUTF8Sequence(
      reinterpret_cast<const llvm::UTF8 **>(&cur),
      reinterpret_cast<const llvm::UTF8 *>(end), &cp, llvm::strictConversion);
  // Handle the conversion result
  switch (result) {
  case llvm::conversionOK:
    return cp;
  case llvm::targetExhausted:
    // (pierre) I don't think that this error is possible, so use
    // llvm_unreachable for now.
    llvm_unreachable("Target exhausted?");
  case llvm::sourceIllegal:
    diagEng.diagnose(SourceLoc::fromPointer(cur), diag::illegal_utf8_cp);
    return invalidCP;
  case llvm::sourceExhausted:
    if (isSourceExhaustedError)
      (*isSourceExhaustedError) = true;
    diagEng.diagnose(SourceLoc::fromPointer(cur), diag::incomplete_utf8_cp);
    return invalidCP;
  default:
    llvm_unreachable("unhandled conversion result");
  }
}

bool Lexer::recoverFromBadUTF8() {
  // always increment once to skip the codepoint
  if (cur != end)
    ++cur;
  for (; cur != end; ++cur) {
    // We are looking for an ASCII character ...
    // For ASCII character, the top bit is simply not set.
    if (!(*cur & 0x80))
      return true;
    // ... or a UTF8 leading byte.
    // UTF8 leading bytes start with 11, so if we do a right-shift
    // of 6 bits, we will end up with 0xFF for UTF8 leading bytes.
    // (the uint8_t conversion is needed due to integer promotion
    // on shifts)
    if ((uint8_t)(*cur >> 6) == 0xFF)
      return true;
  }
  // Reached EOF, couldn't recover.
  return false;
}

void Lexer::stopLexing() {
  assert(nextToken.isNot(TokenKind::EndOfFile) && "lexing is already done");
  nextToken = Token(TokenKind::EndOfFile, CharSourceRange());
  cur = end;
}

void Lexer::doLex() {
  assert(nextToken.isNot(TokenKind::EndOfFile));
  if (cur == end)
    return stopLexing();
}

// TODO: Plan Lexer more in-depth.
// NOTE: Tokens need a half-open range, so we can always use "cur" as the end of
// the token, even if it points to the beginning of the next char.
//    How will doLex() work?
//    How will token pushing work?
//      beginToken
//      pushToken
//  Ideas:
//      - doLex(): call "advance" at the beginning of the loop
//      - advance(): save the CP in a "lastCP" variable