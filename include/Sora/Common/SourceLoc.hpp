//===--- SourceLoc.hpp - Source Locations & Ranges --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Defines types used to represent source locations and ranges across the
// compiler.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "llvm/Support/SMLoc.h"

namespace sora {
class SourceManager;

/// Represents a source location which is generally the first char of a token.
///
/// This is simply a wrapper around a llvm::SMLoc.
///
/// Comparison operators (>, >=, <=, <, ==, !=) can be used to compare the
/// underyling pointers of 2 SourceLocs. Iff the SourceLocs come from the same
/// buffer, this can be used to determine if a SourceLoc is before another one.
class SourceLoc {
  friend class SourceManager;

  llvm::SMLoc value;

public:
  SourceLoc() = default;
  explicit SourceLoc(llvm::SMLoc value) : value(value) {}

  /// \returns a copy of this SourceLoc advanced by \p offset bytes.
  /// This SourceLoc must be valid.
  SourceLoc getAdvancedLoc(unsigned offset) const {
    assert(isValid() && "not valid!");
    return SourceLoc(llvm::SMLoc::getFromPointer(value.getPointer() + offset));
  }

  /// \returns the pointer value of this SourceLoc.
  const char *getPointer() const { return value.getPointer(); }

  /// \returns a copy of this SourceLoc advanced by \p offset bytes, or
  /// SourceLoc() if this SourceLoc is not valid.
  SourceLoc getAdvancedLocIfValid(unsigned numBytes) const {
    return isValid() ? getAdvancedLoc(numBytes) : SourceLoc();
  }

  /// \returns the underlying llvm::SMLoc
  llvm::SMLoc getSMLoc() const { return value; }

  /// Creates a SourceLoc from a pointer \p ptr
  /// NOTE: To be used wisely. Don't mix this with SourceLocs into files managed
  /// by the SourceManager.
  static SourceLoc fromPointer(const char *ptr) {
    return SourceLoc(llvm::SMLoc::getFromPointer(ptr));
  }

  /// \returns true if this SourceLoc is valid
  bool isValid() const { return value.isValid(); }
  /// \returns true if this SourceLoc is invalid
  bool isInvalid() const { return !isValid(); }
  /// \returns true if this SourceLoc is valid
  explicit operator bool() const { return isValid(); }

  bool operator<(const SourceLoc other) const {
    return value.getPointer() < other.value.getPointer();
  }
  bool operator<=(const SourceLoc other) const {
    return value.getPointer() <= other.value.getPointer();
  }
  bool operator>(const SourceLoc other) const {
    return value.getPointer() > other.value.getPointer();
  }
  bool operator>=(const SourceLoc other) const {
    return value.getPointer() >= other.value.getPointer();
  }
  bool operator==(const SourceLoc other) const { return value == other.value; }
  bool operator!=(const SourceLoc other) const { return value != other.value; }
};

/// Represents a source range. 'begin' is the loc of the first character
/// of the first token in the range, and 'end' is the loc of the first character
/// of the last token in the range.
///
/// This is simply a pair of SourceLocs.
class SourceRange {
  friend class SourceManager;

public:
  SourceLoc begin, end;

  SourceRange() = default;

  /// Creates a SourceRange from a single SourceLoc.
  SourceRange(SourceLoc begin) : begin(begin), end(begin) {}

  /// Creates a SourceRange from two SourceLocs.
  /// \param begin the begin loc of the range
  /// \param end the end loc of the range
  SourceRange(SourceLoc begin, SourceLoc end) : begin(begin), end(end) {
    assert(begin.isValid() == end.isValid() &&
           "begin & end should both be valid or invalid");
    assert(
        (begin.isValid() ? (begin.getPointer() <= end.getPointer()) : true) &&
        "end > begin!");
  }

  /// \returns true if this SourceRange is valid
  bool isValid() const { return begin.isValid(); }
  /// \returns true if this SourceRange is invalid
  bool isInvalid() const { return !isValid(); }
  /// \returns true if this SourceRange is valid
  explicit operator bool() const { return isValid(); }

  bool operator==(const SourceRange &other) const {
    return (begin == other.begin) && (end == other.end);
  }

  bool operator!=(const SourceRange &other) const { return !(*this == other); }
};

/// Represents a half-open range of characters in the source.
class CharSourceRange {
  SourceLoc begin;
  unsigned byteLength = 0;

public:
  CharSourceRange() = default;

  CharSourceRange(SourceLoc begin, unsigned byteLength)
      : begin(begin), byteLength(byteLength) {}

  /// Creates a CharSourceRange from 2 SourceLocs by calculating the distance
  /// between them in bytes.
  /// \param srcMgr the SourceManager that owns the buffer in which \p begin and
  /// \p end are located
  /// \param begin the begin location
  /// \param end the end location
  CharSourceRange(SourceManager &srcMgr, SourceLoc begin, SourceLoc end);

  /// \returns true if this CharSourceRange is valid
  bool isValid() const { return begin.isValid(); }
  /// \returns true if this CharSourceRange is invalid
  bool isInvalid() const { return !isValid(); }
  /// \returns true if this CharSourceRange is valid
  bool empty() const { return byteLength == 0; }
  /// \returns true if this CharSourceRange is valid
  explicit operator bool() const { return isValid(); }

  /// \returns this CharSourceRange as a llvm::SMRange
  llvm::SMRange getSMRange() const {
    return {getBegin().getSMLoc(), getEnd().getSMLoc()};
  }

  /// \returns the begin SourceLoc
  SourceLoc getBegin() const { return begin; }
  /// \returns the past-the-end SourceLoc.
  SourceLoc getEnd() const { return begin.getAdvancedLocIfValid(byteLength); }

  /// Creates a CharSourceRange from 2 pointer \p start and \p end
  /// NOTE: To be used wisely. Don't mix this with CharSourceRanges into files
  /// managed by the SourceManager.
  static CharSourceRange fromPointers(const char *start, const char *end) {
    assert(end >= start && "start > end!");
    return {SourceLoc::fromPointer(start), (unsigned)std::distance(start, end)};
  }

  /// \returns the range of characters as a read-only string.
  StringRef str() const;

  bool operator==(const CharSourceRange &other) const {
    return (begin == other.begin) && (byteLength == other.byteLength);
  }

  bool operator!=(const CharSourceRange &other) const {
    return !(*this == other);
  }
};

} // namespace sora