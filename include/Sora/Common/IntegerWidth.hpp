//===--- IntegerWidth.hpp - Integer Width Representation --------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Error.h"

namespace llvm {
class APInt;
class Triple;
} // namespace llvm

namespace sora {
/// Represents the width of an integer, which can be either a fixed value (e.g.
/// 32) or an arbitrary/target-dependent value (pointer-sized).
///
/// This class also offers a "parse" methods to parse an integer of that width.
class IntegerWidth final {
public:
  /// The type of the integer width stored inside this IntegerWidth object.
  /// Currently, this is a 16 bits unsigned integer.
  using width_t = uint16_t;

  /// The type of an IntegerWidth represented as an opaque value
  using opaque_t = uint32_t;

private:
  enum class Kind : uint8_t {
    /// DenseMap Tombstone Key
    DMTombstone,
    /// DenseMap Empty Key
    DMEmpty,
    /// Fixed width (e.g. 16)
    Fixed,
    /// Arbitrary precision
    Arbitrary,
    /// Pointer-sized (usize, 32 or 64 bits usually)
    Pointer
  };

  struct Data {
    Data(Kind kind, width_t width) : kind(kind), width(width) {}

    Kind kind;
    width_t width;
  };

  union {
    opaque_t opaque;
    Data data;
    static_assert(sizeof(Data) == sizeof(opaque_t),
                  "Data is too large to fit in opaque_t");
  };

  IntegerWidth(Kind kind, width_t width) : data(kind, width) {}
  IntegerWidth(uint32_t opaque) : opaque(opaque) {}

  friend struct llvm::DenseMapInfo<sora::IntegerWidth>;

  bool isDenseMapSpecial() const {
    return data.kind == Kind::DMEmpty || data.kind == Kind::DMTombstone;
  }

public:
  /// Creates an IntegerWidth from a raw value
  static IntegerWidth fromOpaqueValue(opaque_t raw) {
    return IntegerWidth(raw);
  }

  /// \returns this IntegerWidth as an opaque value
  opaque_t getOpaqueValue() const { return opaque; }

  /// \returns an IntegerWidth representing an integer with a fixed width of \p
  /// value. \p value can't be zero.
  static IntegerWidth fixed(width_t value) {
    assert(value != 0 && "Can't create an integer of width 0");
    return IntegerWidth(Kind::Fixed, value);
  }

  /// \returns an IntegerWidth representing an arbitrary precision integer
  static IntegerWidth arbitrary() { return IntegerWidth(Kind::Arbitrary, 0); }

  /// \returns an IntegerWidth representing a pointer-sized integer (16, 32
  /// or 64 bits depending on the platform).
  static IntegerWidth pointer(const llvm::Triple &triple);

  bool isFixedWidth() const { return data.kind == Kind::Fixed; }
  bool isArbitraryPrecision() const { return data.kind == Kind::Arbitrary; }
  bool isPointerSized() const { return data.kind == Kind::Pointer; }

  /// For fixed-width and pointer-sized integers, returns the width the integer
  /// has. Cannot be used on arbitrary-precision integers.
  width_t getWidth() const {
    assert((isFixedWidth() || isPointerSized()) &&
           "not fixed width or pointer sized");
    return data.width;
  }

  /// Integer Parsing Status
  enum class Status {
    /// Integer was parsed successfully
    Ok,
    /// An error occured while parsing the integer
    Error,
    /// The integer overflowed (can't happen on arbitrary-precision integers)
    Overflow
  };

  /// Parses an integer of this width. This works with any integer width:
  /// pointer-sized, fixed-width and arbitrary-precision.
  APInt parse(StringRef str, int isNegative, unsigned radix = 0,
              Status *status = nullptr) const;

  friend bool operator==(IntegerWidth lhs, IntegerWidth rhs) {
    return (lhs.data.kind == rhs.data.kind) &&
           (lhs.data.width == rhs.data.width);
  }

  friend bool operator!=(IntegerWidth lhs, IntegerWidth rhs) {
    return !(lhs == rhs);
  }
};
} // namespace sora

namespace llvm { // DenseMapInfo for IntegerWidth
template <> struct DenseMapInfo<sora::IntegerWidth> {
  using IntegerWidth = sora::IntegerWidth;

  static IntegerWidth getEmptyKey() {
    return IntegerWidth(IntegerWidth::Kind::DMEmpty, 0);
  }

  static IntegerWidth getTombstoneKey() {
    return IntegerWidth(IntegerWidth::Kind::DMTombstone, 0);
  }

  static unsigned getHashValue(IntegerWidth value) {
    return DenseMapInfo<unsigned>::getHashValue(value.getOpaqueValue());
  }

  static bool isEqual(IntegerWidth lhs, IntegerWidth rhs) { return lhs == rhs; }
};
} // namespace llvm