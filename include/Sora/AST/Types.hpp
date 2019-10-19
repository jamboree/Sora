//===--- Types.hpp - Types ASTs ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the whole "type" hierarchy
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/Error.h"

namespace llvm {
class APInt;
}

namespace sora {
class ASTContext;

/// Kinds of Types
enum class TypeKind : uint8_t {
#define TYPE(KIND, PARENT) KIND,
#define TYPE_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/TypeNodes.def"
};

/// Common base class for Types.
class alignas(TypeBaseAlignement) TypeBase {
  // Disable vanilla new/delete for types
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  TypeKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  // Also allow allocation of Types using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(TypeBase));

  TypeBase(TypeKind kind) : kind(kind) {}

public:
  /// \returns the kind of type this is
  TypeKind getKind() const { return kind; }
};

/// Represents the width of an integer, which can be either a fixed value (e.g.
/// 32) or an arbitrary/target-dependent value (pointer-sized).
///
/// FIXME: This class should know about the current build configuration in some
/// way (for pointer sizes). Once it has access to that, we can remove
/// getFixedWidth/getMinWidth/getMaxWidth and unify everything under a getWidth
/// function.
class IntegerWidth final {
public:
  using width_t = uint16_t;

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
    /// Pointer-sized (usize, either 32 or 64 bits usually)
    Pointer
  };

  union {
    uint32_t raw;
    struct Data {
      Data(Kind kind, width_t width) : kind(kind), width(width) {}

      Kind kind;
      width_t width;
    } data;
    static_assert(sizeof(Data) == 4, "Data must be 32 bits!");
  };

  IntegerWidth(Kind kind, width_t width = 0) : data(kind, width) {}

  friend struct llvm::DenseMapInfo<sora::IntegerWidth>;

  unsigned getRaw() const { return raw; }

  bool isDenseMapSpecial() const {
    return data.kind == Kind::DMEmpty || data.kind == Kind::DMTombstone;
  }

public:
  /// \returns an IntegerWidth representing an integer with a fixed width of \p
  /// value. \p value can't be zero.
  static IntegerWidth fixedWidth(width_t value) {
    assert(value != 0 && "Can't create an integer of width 0");
    return IntegerWidth(Kind::Fixed, value);
  }

  /// \returns an IntegerWidth representing an arbitrary precision integer
  static IntegerWidth arbitraryPrecision() {
    return IntegerWidth(Kind::Arbitrary);
  }

  /// \returns an IntegerWidth representing a pointer-sized integer
  static IntegerWidth pointerSized() { return IntegerWidth(Kind::Pointer); }

  bool isFixedWidth() const { return data.kind == Kind::Fixed; }
  bool isArbitraryPrecision() const { return data.kind == Kind::Arbitrary; }
  bool isPointerSized() const { return data.kind == Kind::Pointer; }

  /// For FixedWidth integers, returns the width the integer has.
  width_t getFixedWidth() const {
    assert(isFixedWidth() && "not fixed width");
    return data.width;
  }

  /// \returns the minimum width an integer of this IntegerWidth can have.
  width_t getMinWidth() const {
    switch (data.kind) {
    case Kind::DMEmpty:
    case Kind::DMTombstone:
      llvm_unreachable("DenseMap-specific IntegerWidths don't have a width");
    case Kind::Arbitrary:
      return 1;
    case Kind::Fixed:
      return data.width;
    case Kind::Pointer:
      return 32;
    }
    llvm_unreachable("unknown IntegerWidth::Kind");
  }

  /// \returns the maximum width an integer of this IntegerWidth can have.
  width_t getMaxWidth() const {
    switch (data.kind) {
    case Kind::DMEmpty:
    case Kind::DMTombstone:
      llvm_unreachable("DenseMap-specific IntegerWidths don't have a width");
    case Kind::Arbitrary:
      return ~width_t(0);
    case Kind::Fixed:
      return data.width;
    case Kind::Pointer:
      return 64;
    }
    llvm_unreachable("unknown IntegerWidth::Kind");
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

  /// Parses an integer.
  /// Note that for pointer-sized IntegerWidth, this uses getMaxWidth() and not
  /// the current build target's pointer size.
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
    return IntegerWidth(IntegerWidth::Kind::DMEmpty);
  }

  static IntegerWidth getTombstoneKey() {
    return IntegerWidth(IntegerWidth::Kind::DMTombstone);
  }

  static unsigned getHashValue(IntegerWidth value) {
    return DenseMapInfo<unsigned>::getHashValue(value.getRaw());
  }

  static bool isEqual(IntegerWidth lhs, IntegerWidth rhs) { return lhs == rhs; }
};
} // namespace llvm