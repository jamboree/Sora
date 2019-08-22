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
  curPtr = str.begin();
  endPtr = str.end();
  nextToken = Token();
  // TODO: Skip UTF8 bom if present

  // Lex the first token (so nextToken has a value)
  lexImpl();
}

void Lexer::init(BufferID id) { init(srcMgr.getBufferStr(id)); }

Token Lexer::lex() {
  auto tok = nextToken;
  // Lex if we haven't reached EOF
  if (tok.isNot(TokenKind::EndOfFile))
    lexImpl();
  return tok;
}

Token Lexer::peek() const { return nextToken; }

void Lexer::stopLexing() {
  assert(nextToken.isNot(TokenKind::EndOfFile) && "already EOF");
  nextToken = Token(TokenKind::EndOfFile, CharSourceRange());
  curPtr = endPtr;
}

namespace {
/// \returns true if \p ch is a valid identifier head
bool isValidIdentifierHead(char ch) {
  // identifier-head = (uppercase or lowercase letter | "_")
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_');
}

/// \returns the next valid UTF8 codepoint and moves the \p cur iterator past
/// the end of that codepoint. On error, returns ~0U and leaves \p cur unchanged
uint32_t advanceUTF8(const char *&cur, const char *end,
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

/// \returns true if \p ch is a UTF8 byte
bool isUTF8(char ch) { return ((unsigned char)ch & 0x80); }
} // namespace

void Lexer::lexIdentifierBody() {
  // TODO: Eat the body of the identifier and push the token
}

void Lexer::lexUnknown() {
  // TODO: Create an "Unknown" token
  // Handle UTF8 codepoint with continuation codepoints as well.
  // It isn't too hard. Diagnose invalid/partial codepoints as well.
}

void Lexer::lexImpl() {
  assert(nextToken.isNot(TokenKind::EndOfFile));
  if (curPtr == endPtr)
    return stopLexing();
  tokBegPtr = curPtr;
  switch (char ch = *curPtr++) {
  default:
    lexUnknown();
    break;

  // TODO: All operator cases

  // Note: for the rest of the switch we'll disable formatting, as
  // we don't want to end up with 60+ extra lines.
  // clang-format off

  // Number literals
  case '0': case '1': case '2': case '3': case '4': case '5': case '6':
  case '7': case '8': case '9':
    // lexNumber()
    break;

  // Identifiers
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z': case 'A': case 'B':
  case 'C': case 'D': case 'E': case 'F': case 'G': case 'H': case 'I':
  case 'J': case 'K': case 'L': case 'M': case 'N': case 'O': case 'P':
  case 'Q': case 'R': case 'S': case 'T': case 'U': case 'V': case 'W':
  case 'X': case 'Y': case 'Z': case '_':
    // lexIdentifierBody()
    break;
  }
  // clang-format on
}
/*
  tokStart = cur
  switch(*cur++)
    all cases (operator, trivias and stuff)
    default: identifier stuff

  lexIdentifier, lexLiteral and stuff always do cur = tokStart
*/