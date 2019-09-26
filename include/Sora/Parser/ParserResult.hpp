//===--- ParserResult.hpp - Parser Result -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/PointerIntPair.h"

namespace sora {
template <typename T> class ParserResult final {
  /// The Result + a flag indicating if there was a parsing error.
  llvm::PointerIntPair<T *, 1> resultAndIsParseError;

  ParserResult(T *ptr, bool isParseError)
      : resultAndIsParseError(ptr, isParseError) {}

public:
  /// Constructor to allow implict upcast, e.g. cast a ParserResult<BinaryExpr>
  /// to ParserResult<Expr>.
  template <typename U, typename = typename std::enable_if<
                            std::is_base_of<T, U>::value, U>::type>
  ParserResult(ParserResult<U> &&derived)
      : ParserResult(derived.getOrNull(), derived.isParseError()) {}
  // Constructor for parsing errors with no node returned.
  ParserResult(std::nullptr_t) : ParserResult(nullptr, true) {}
  /// Constructor for successful parser results. \p result cannot be nullptr.
  explicit ParserResult(T *result) : ParserResult(result, false) {
    assert(result && "Successful Parser Results can't be nullptr!");
  }

  /// Sets the "isParseError" flag to \p value
  void setIsParseError(bool value = true) {
    resultAndIsParseError.setInt(value);
  }
  /// \returns true if this represents a parsing error
  bool isParseError() const { return resultAndIsParseError.getInt(); }

  /// \returns true if this ParserResult has a value
  bool hasValue() const { return !isNull(); }
  /// \returns true if this ParserResult has no value
  bool isNull() const { return getOrNull() == nullptr; }

  /// \returns the result (can't be nullptr)
  T *get() const {
    T *ptr = getOrNull();
    assert(ptr && "result can't be nullptr!");
    return ptr;
  }

  /// \returns the result or nullptr (if error)
  T *getOrNull() const { return resultAndIsParseError.getPointer(); }
};

/// Creates a successful parser result
template <typename T>
static inline ParserResult<T> makeParserResult(T *result) {
  return ParserResult<T>(result);
}

/// Creates a parser error result
template <typename T>
static inline ParserResult<T> makeParserErrorResult(T *result) {
  if (!result)
    return ParserResult<T>(nullptr);
  ParserResult<T> pr(result);
  pr.setIsParseError();
  return pr;
}
} // namespace sora