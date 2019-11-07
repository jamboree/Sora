//===--- Identifier.hpp - Declaration Identifiers ---------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/DenseMapInfo.h"
#include <string>

namespace sora {
/// Represents a unique'd language identifier.
class Identifier {
  const char *value = nullptr;

  friend class ASTContext;
  friend struct llvm::DenseMapInfo<sora::Identifier>;
  Identifier(const char *value) : value(value) {}

public:
  /// Create a null identifier
  Identifier() = default;

  /// \returns this identifier as a string
  StringRef str() const;

  /// \returns the c-style string of the identifier
  /// This pointer is unique for each identifier.
  const char *c_str() const { return value; }

  /// \returns whether this is a valid identifier or not
  bool isValid() const { return value; }

  /// \returns whether this is a valid identifier or not
  explicit operator bool() const { return isValid(); }

  // Comparing with another Identifier
  bool operator==(const Identifier &rhs) const { return value == rhs.value; }
  bool operator!=(const Identifier &rhs) const { return value != rhs.value; }
  bool operator<(const Identifier &rhs) const { return value < rhs.value; }

  // Comparing with a StringRef
  bool operator==(StringRef rhs) const;
  bool operator!=(StringRef rhs) const;
  bool operator<(StringRef rhs) const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &out, Identifier ident);

template <typename Ty> struct DiagnosticArgumentFormatter;

template <> struct DiagnosticArgumentFormatter<Identifier> {
  static std::string format(Identifier ident) { return ident.c_str(); }
};
} // namespace sora

namespace llvm {
template <> struct DenseMapInfo<sora::Identifier> {
  static sora::Identifier getEmptyKey() {
    return sora::Identifier(DenseMapInfo<const char *>::getEmptyKey());
  }

  static sora::Identifier getTombstoneKey() {
    return sora::Identifier(DenseMapInfo<const char *>::getTombstoneKey());
  }

  static unsigned getHashValue(const sora::Identifier &ident) {
    return DenseMapInfo<const char *>::getHashValue(ident.c_str());
  }

  static bool isEqual(const sora::Identifier &lhs,
                      const sora::Identifier &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm
