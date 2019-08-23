//===--- Token.hpp - Lexer Tokens -------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include <stdint.h>

namespace sora {
/// Kinds of tokens.
enum class TokenKind : uint8_t {
#define TOKEN(KIND) KIND,
#include "TokenKinds.def"
};

/// \returns the string representation of a TokenKind
const char *to_string(TokenKind kind);

static_assert(TokenKind(0) == TokenKind::Invalid,
              "0 is not the invalid TokenKind");

/// Represents a single Token produced by the Lexer.
///
/// As the Lexer is intended to be used "on-demand" (tokens are consumed
/// as they are produced and aren't stored in an array), the Token
/// object is not intended to be space efficient.
class Token {
public:
  /// Constructs an invalid Token
  Token() = default;

  /// \param kind the kind of Token this is
  /// \param charRange the CharSourceRange of the token
  /// \param startOfLine true if the token is at the start of a line
  Token(TokenKind kind, CharSourceRange charRange, bool startOfLine = false)
      : kind(kind), charRange(charRange), startOfLine(startOfLine) {}

  /// \returns true if kind == \p kind
  bool is(TokenKind kind) const { return this->kind == kind; }
  /// \returns true if kind != \p kind
  bool isNot(TokenKind kind) const { return this->kind != kind; }

  /// \returns true if kind != Invalid
  explicit operator bool() const { return !is(TokenKind::Invalid); }

  /// \returns the token string
  StringRef str() const;

  /// \returns the SourceLoc pointing at the beginning of the Token.
  SourceLoc getBegLoc() const { return charRange.getBegin(); }

  /// \returns the kind of this token
  TokenKind getKind() const { return kind; }

  /// \returns the CharSourceRange of the Token
  CharSourceRange getCharRange() const { return charRange; }

  /// \returns "true" if the token is at the start of a line, false otherwise.
  bool isStartOfLine() const { return startOfLine; }

  /// Dumps information about the token to llvm::outs()
  void dump() const;
  /// Dumps information about the token to \p out
  /// TODO: Add support for a SourceManager& argument to also dump
  /// the locs.
  void dump(raw_ostream &out) const;

private:
  /// Set to true if the token is at the start of a line, false otherwise.
  bool startOfLine = false;
  /// The kind of this Token
  TokenKind kind = TokenKind::Invalid;
  /// The CharSourceRange of this Token
  CharSourceRange charRange;
};
} // namespace sora
