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
  cur = str.begin();
  end = str.end();
  nextToken = Token();
  // TODO: Skip UTF8 bom if present
  // Advance at least once and discard the value.
  // The first call to advance() always returns ~0U.
  advance();
  // Lex the first token (so nextToken has a value)
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

/// Tries to recover from a bad UTF8 codepoint. This increments 'cur' at
/// least once.
/// \returns true if recovery was successful, false otherwise
static bool recoverFromBadUTF8(const char *&cur, const char *end) {
  // always increment once to skip the codepoint
  if (cur != end)
    ++cur;
  for (; cur != end; ++cur) {
    // We are looking for an ASCII character ...
    // (for ASCII characters, the top bit is simply not set.)
    if (!(*cur & 0x80))
      return true;
    // ... or a UTF8 leading byte.
    // UTF8 leading bytes start with 0b11, so if we do a right-shift
    // of 6 bits, we will end up with 0xFF for UTF8 leading bytes.
    // (the uint8_t conversion is needed due to integer promotion
    // on shifts)
    if ((uint8_t)(*cur >> 6) == 0xFF)
      return true;
  }
  // Reached EOF, couldn't recover.
  return false;
}

/// \returns the next valid UTF8 codepoint and moves the \p cur iterator past
/// the end of that codepoint. On error, returns ~0U and leaves \p cur unchanged
static uint32_t advanceUTF8(const char *&cur, const char *end,
                            llvm::ConversionResult &result) {
  if (cur >= end) {
    result = llvm::conversionOK;
    return ~0U;
  }
  // use llvm::convertUTF8Sequence to fetch the codepoint
  // FIXME: I think this could be cleaner.
  uint32_t cp;
  result = llvm::convertUTF8Sequence(
      reinterpret_cast<const llvm::UTF8 **>(&cur),
      reinterpret_cast<const llvm::UTF8 *>(end), &cp, llvm::strictConversion);
  // Handle the conversion result
  return (result == llvm::conversionOK) ? cp : ~0U;
}

uint32_t Lexer::advance() {
  // Save nextCP (because it's the value we'll return) and move "cur".
  uint32_t ret = nextCP;
  llvm::outs() << "cp: " << (char)nextCP << "\n";
  cur = nextCur;

  // Check if not EOF
  if (cur == end)
    return ret;

  // fast path for ASCII stuff
  if (!(*cur & 0x80)) {
    nextCP = *cur++;
    return ret;
  }

  // slow path for UTF8 stuff

  // try to advance
  llvm::ConversionResult result;
  nextCP = advanceUTF8(nextCur, end, result);

  // handle the  result
  switch (result) {
  default:
    llvm_unreachable("unhandled ConversionResult");
  case llvm::ConversionResult::targetExhausted:
    // (pierre) I don't think this error is possible.
    llvm_unreachable("target exhausted?");
  case llvm::ConversionResult::conversionOK:
    return ret;
  case llvm::ConversionResult::sourceExhausted:
    diagEng.diagnose(SourceLoc::fromPointer(cur), diag::incomplete_utf8_cp);
    break;
  case llvm::ConversionResult::sourceIllegal:
    diagEng.diagnose(SourceLoc::fromPointer(cur), diag::illegal_utf8_cp);
    break;
  }

  // When we have an error, nextCP will be ~0U and nextCur will be unchanged.
  assert((nextCP == ~0U) && (nextCur == cur) && "invalid error situation");
  // We can try to recover to the next ASCII character or UTF8 leading byte
  // and re-try to advance from there (but discard the return value since the
  // recursive call to advance() will just return ~0U)
  recoverFromBadUTF8(nextCur, end);
  advance();

  return ret;
}

uint32_t Lexer::peekChar() const { return nextCP; }

void Lexer::stopLexing() {
  assert(nextToken.isNot(TokenKind::EndOfFile) && "already EOF");
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
