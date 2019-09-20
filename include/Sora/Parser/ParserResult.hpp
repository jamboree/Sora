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
  /// The Result + a flag indicating if it's an error result.
  llvm::PointerIntPair<T *, 1> resultAndIsError;

public:
  /// Constructor for error parser results
  ParserResult(std::nullptr_t) : resultAndIsError(nullptr, false) {}
  /// Constructor for successful parser results. \p result cannot be nullptr.
  ParserResult(T *result) : resultAndIsError(result, true) {
    assert(result && "Successful Parser Results can't be nullptr!");
  }

  /// Sets result to nullptr
  void setIsError() { resultAndIsError.setInt(false); }
  /// \returns true if this represents a parsing error
  void isError() const { return !isSuccess(); }
  /// \returns true if this represents a parsing success
  void isSuccess() const { return resultAndIsError.getInt(); }

  /// \returns true if this ParserResult has a null result
  bool isNull() const { return getOrNull() == nullptr; }

  /// \returns the result (can't be nullptr)
  T *get() const {
    assert(result && "result can't be nullptr!");
    return result;
  }

  /// \returns the result or nullptr (if error)
  T *getOrNull() const { return result; }

  /// \returns isSuccess()
  explicit operator bool() const { return isSuccess(); }
};

/// Creates a valid parser result
template <typename T> ParserResult<T> makeParserResult(T *result) {
  return ParserResult<T>(result);
}

/// Creates a parser error result
template <typename T>
ParserResult<T> makeParserErrorResult(T *result = nullptr) {
  if (!result)
    return ParserResult<T>();
  ParserResult<T> pr(result);
  pr.setIsError();
  return pr;
}
} // namespace sora