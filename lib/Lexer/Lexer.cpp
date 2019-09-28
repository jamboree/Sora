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
#include "Sora/Diagnostics/DiagnosticsLexer.hpp"
#include "Sora/Lexer/Token.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

const char *sora::to_string(TokenKind kind) {
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

namespace {
/// \returns true if \p ch is a valid identifier head
bool isValidIdentifierHead(char ch) {
  // identifier-head = (uppercase or lowercase letter | "_")
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch == '_');
}

/// \returns true if \p ch is considered trivia (= a character that can be
/// ignored safely)
bool isTrivia(char ch) {
  // Don't use formatting for this switch so we can stack cases on a
  // single line to reduce clutter.
  // clang-format off
  switch (ch) {
  default: return false;
  case 0: case ' ': case '\t': case '\r': case '\v': case '\f': case '\n':
    return true;
  }
  // clang-format on
}

/// \returns true if \p ch is a UTF8 byte
bool isUTF8(char ch) { return ((unsigned char)ch & 0x80); }

/// \returns true if \p ch is a digit
bool isdigit(char ch) { return ch >= '0' && ch <= '9'; }

/// \returns true if \p ch is the first byte of a UTF8 continuation codepoint.
bool isBeginningOfContinuationCodePoint(char ch) {
  // Continuation codepoints begin with 10
  return uint8_t(ch >> 6) == 0b11111110;
}

/// \returns the next valid UTF8 codepoint and moves the \p cur iterator past
/// the end of that codepoint. On error, returns ~0U and leaves \p cur
/// unchanged
uint32_t advanceUTF8(const char *&cur, const char *end,
                     llvm::ConversionResult &result) {
  if (cur >= end) {
    result = llvm::conversionOK;
    return ~0U;
  }
  // ASCII fast path
  if (!isUTF8(*cur)) {
    result = llvm::conversionOK;
    return *cur++;
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
} // namespace

void Lexer::lexUnknown() {
  curPtr = tokBegPtr;

  bool sourceOk = true;
  auto advance = [&]() {
    llvm::ConversionResult result;
    advanceUTF8(curPtr, endPtr, result);
    switch (result) {
    default:
      llvm_unreachable("unknown ConversionResult");
    case llvm::targetExhausted:
      llvm_unreachable("target exhausted?");
    case llvm::conversionOK:
      break;
    case llvm::sourceExhausted:
      diagnose(tokBegPtr, diag::incomplete_utf8_cp);
      sourceOk = false;
      break;
    case llvm::sourceIllegal:
      diagnose(tokBegPtr, diag::illegal_utf8_cp);
      sourceOk = false;
      break;
    }
  };

  advance();

  // When the source is valid, skip potential continuation codepoints,
  // validating them in the process.
  while (sourceOk && isBeginningOfContinuationCodePoint(*curPtr))
    advance();
  // When the source can't be trusted, just skip until the next ASCII character.
  while (!sourceOk && isUTF8(*curPtr) && (curPtr != endPtr))
    ++curPtr;
  // Push the token
  pushToken(TokenKind::Unknown);
}

void Lexer::lexNumberLiteral() {
  assert(isdigit(*tokBegPtr) && (tokBegPtr + 1 == curPtr));
  // integer-literal = digit+
  // digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
  // floating-point-literal = integer-literal ('.' integer-literal)?
  auto consumeInteger = [&]() {
    while (isdigit(*curPtr)) {
      ++curPtr;
    }
  };

  // consume the integer
  consumeInteger();
  // check if there's a '.' followed by another digit, in that case we got a
  // floating-point literal.
  if (*curPtr == '.' && isdigit(*(curPtr + 1))) {
    ++curPtr;
    consumeInteger();
    pushToken(TokenKind::FloatingPointLiteral);
  }
  else
    pushToken(TokenKind::IntegerLiteral);
}

void Lexer::lexIdentifierBody() {
  // identifier-head = (uppercase or lowercase letter | "_")
  // identifier-body = (identifier-head | digit)
  // NOTE: Currently, only ASCII is allowed in identifiers, so we
  // can use ++curPtr safely, however if in the future UTF8 is
  // allowed in identifiers, we'll need advanceUTF8.
  while (!isUTF8(*curPtr) &&
         (isValidIdentifierHead(*curPtr) || isdigit(*curPtr)))
    ++curPtr;
  StringRef tokStr = getTokStr();
#define KEYWORD(KIND, TEXT)                                                    \
  if (tokStr == TEXT)                                                          \
    return pushToken(TokenKind::KIND);
#include "Sora/Lexer/TokenKinds.def"
  pushToken(TokenKind::Identifier);
}

void Lexer::handleLineComment() {
  assert(*curPtr == '/' && *(curPtr + 1) == '/');
  // line-break = '\r'? '\n'
  // line-comment-item = any character except '\n' or '\r'
  // line-comment = "//" line-comment-item* line-break
  curPtr += 2;
  while (curPtr != endPtr) {
    if (*curPtr++ == '\n') {
      tokenIsAtStartOfLine = true;
      return;
    }
  }
}

void Lexer::handleBlockComment() {
  assert(*curPtr == '/' && *(curPtr + 1) == '*');
  // line-break = '\r'? '\n'
  // block-comment-item = any character except "*/"
  // block-comment = "/*" block-comment-item* "*/"
  curPtr += 2;
  char ch;
  while (curPtr != endPtr) {
    ch = *curPtr++;
    if (ch == '\n')
      tokenIsAtStartOfLine = true;
    else if ((ch == '*') && (*curPtr == '/')) {
      ++curPtr;
      return;
    }
  }
}

void Lexer::lexImpl() {
  assert(nextToken.isNot(TokenKind::EndOfFile));
  // Consume the trivia
  consumeTrivia();
  // Check if not EOF
  if (curPtr == endPtr)
    return stopLexing();
  // Start the new token
  tokBegPtr = curPtr;
  // Don't use formatting for this switch so we can stack cases on a
  // single line to reduce clutter.
  // clang-format off
  switch (char ch = *curPtr++) {
  default:
    lexUnknown();
    break;
  case 0: case ' ': case '\t': case '\r': case '\v': case '\f': case '\n':
    llvm_unreachable("should be handled by consumeTrivia()");
  case '&':
    if (*curPtr == '&')
      ++curPtr, pushToken(TokenKind::AmpAmp);
    else if (*curPtr == '=')
      ++curPtr, pushToken(TokenKind::AmpEqual);
    else 
      pushToken(TokenKind::Amp);
    break;
  case '-':
    if(*curPtr == '>')
      ++curPtr, pushToken(TokenKind::Arrow);
    else if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::MinusEqual);
    else 
      pushToken(TokenKind::Minus);
    break;
  case '^':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::CaretEqual);
    else 
      pushToken(TokenKind::Caret);
    break;
  case '+':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::PlusEqual);
    else 
      pushToken(TokenKind::Plus);
    break;
  case '/':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::SlashEqual);
    else 
      pushToken(TokenKind::Slash);
    break;
  case '*':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::StarEqual);
    else 
      pushToken(TokenKind::Star);
    break;
  case '%':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::PercentEqual);
    else 
      pushToken(TokenKind::Percent);
    break;
  case '|':
    if(*curPtr == '|')
      ++curPtr, pushToken(TokenKind::PipePipe);
    else if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::PipeEqual);
    else 
      pushToken(TokenKind::Pipe);
    break;
  case '>':
    if (*curPtr == '>') {
      ++curPtr;
      if(*curPtr == '=')
        ++curPtr, pushToken(TokenKind::GreaterGreaterEqual);
      else
        pushToken(TokenKind::GreaterGreater);
    }
    else if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::GreaterEqual);
    else 
      pushToken(TokenKind::Greater);
    break;
  case '<':
    if (*curPtr == '<') {
      ++curPtr;
      if(*curPtr == '=')
        ++curPtr, pushToken(TokenKind::LessLessEqual);
      else
        pushToken(TokenKind::LessLess);
    }
    else if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::LessEqual);
    else 
      pushToken(TokenKind::Less);
    break;
  case ',':
    pushToken(TokenKind::Comma);
    break;
  case ';':
    pushToken(TokenKind::Semicolon);
    break;
  case '.':
    pushToken(TokenKind::Dot);
    break;
  case ':':
    pushToken(TokenKind::Colon);
    break;
  case '!':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::ExclaimEqual);
    else 
      pushToken(TokenKind::Exclaim);
    break;
  case '=':
    if(*curPtr == '=')
      ++curPtr, pushToken(TokenKind::EqualEqual);
    else 
      pushToken(TokenKind::Equal);
    break;
  case '~':
    pushToken(TokenKind::Tilde);
    break;
  case '{':
    pushToken(TokenKind::LCurly);
    break;
  case '}':
    pushToken(TokenKind::RCurly);
    break;
  case '(':
    pushToken(TokenKind::LParen);
    break;
  case ')':
    pushToken(TokenKind::RParen);
    break;
  case '[':
    pushToken(TokenKind::LSquare);
    break;
  case ']':
    pushToken(TokenKind::RSquare);
    break;
  // numeric literal
  case '0': case '1': case '2': case '3': case '4': case '5': case '6':
  case '7': case '8': case '9':
    lexNumberLiteral();
    break;
  // identifiers & keywords
  case 'a': case 'b': case 'c': case 'd': case 'e': case 'f': case 'g':
  case 'h': case 'i': case 'j': case 'k': case 'l': case 'm': case 'n':
  case 'o': case 'p': case 'q': case 'r': case 's': case 't': case 'u':
  case 'v': case 'w': case 'x': case 'y': case 'z': case 'A': case 'B':
  case 'C': case 'D': case 'E': case 'F': case 'G': case 'H': case 'I':
  case 'J': case 'K': case 'L': case 'M': case 'N': case 'O': case 'P':
  case 'Q': case 'R': case 'S': case 'T': case 'U': case 'V': case 'W':
  case 'X': case 'Y': case 'Z': case '_':
    lexIdentifierBody();
    break;
  }
  // clang-format on
}

void Lexer::consumeTrivia() {
  while (curPtr != endPtr) {
    // handle normal trivia
    if (isTrivia(*curPtr)) {
      if (*curPtr == '\n')
        tokenIsAtStartOfLine = true;
      ++curPtr;
    }
    // slash-slash "line" comments
    else if ((*curPtr == '/') && (*(curPtr + 1) == '/'))
      handleLineComment();
    // slash-start "block" comments
    else if ((*curPtr == '/') && (*(curPtr + 1) == '*'))
      handleBlockComment();
    else
      break;
  }
}

StringRef Lexer::getTokStr() const {
  return StringRef(tokBegPtr, std::distance(tokBegPtr, curPtr));
}
