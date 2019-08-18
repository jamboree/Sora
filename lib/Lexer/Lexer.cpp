//===--- Lexer.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Implementation of the Lexer & Token classes.
//===----------------------------------------------------------------------===//

#include "Sora/Lexer/Lexer.hpp"
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
  // Lex the first token
  doLex();
}

void Lexer::init(BufferID id) { init(srcMgr.getBufferStr(id)); }

Token Lexer::lex() {
  auto next = nextToken;
  // Lex if we haven't reached EOF
  if (next.isNot(TokenKind::EndOfFile))
    doLex();
  return next;
}

Token Lexer::peek() const { return nextToken; }

uint32_t Lexer::advance() {
  if (cur >= end)
    return invalidCP;
  // fast-path for non-UTF8 characters
  char curChar = *cur;
  if (!(curChar & 0x80))
    return ++cur, curChar;
  // use llvm::convertUTF8Sequence to fetch the codepoint
  uint32_t cp;
  auto oldCur = cur;
  auto result = llvm::convertUTF8Sequence(
      // FIXME: Are those conversions safe?
      reinterpret_cast<const llvm::UTF8 **>(&cur),
      reinterpret_cast<const llvm::UTF8 *>(end), &cp, llvm::lenientConversion);
  // Handle the conversion result
  switch (result) {
  case llvm::conversionOK:
    return cp;
  case llvm::targetExhausted:
  case llvm::sourceIllegal:
  case llvm::sourceExhausted:
    return invalidCP;
  default:
    llvm_unreachable("unhandled conversion result");
  }
}

void Lexer::doLex() {
  assert(nextToken.isNot(TokenKind::EndOfFile));
  assert(cur != end);
  // TODO
  unsigned k = 0;
  while (true) {
    auto cp = advance();
    if (cp == invalidCP)
      break;
    llvm::outs() << "[" << ++k << "][" << std::distance(beg, cur) << "] - '" << (char)cp << "'\n";
  }
}
