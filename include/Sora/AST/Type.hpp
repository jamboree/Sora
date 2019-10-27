//===--- Type.hpp - Type Object ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the "Type" object which is a wrapper around TypeBase
// pointers. This is needed to disable direct type pointer comparison, which
// can be deceiving due to the canonical/non-canonical type distinction.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/Common/LLVM.hpp"
#include <cassert>
#include <string>

namespace sora {
class TypeBase;
class TypeRepr;
class SourceLoc;
class SourceRange;

/// Type Printing Options
struct TypePrintOptions {
  /// If true, null types are printed as <null_type>, if false, traps on null
  /// type.
  bool allowNullTypes = true;
  /// If true, type variables are printed as '_' instead of '$Tx'
  bool printTypeVariablesAsUnderscore = false;
  /// If true, lvalues are printed as @lvalue T instead of being "transparent"
  bool printLValues = false;
  /// If true, null types are printed as <error_type>, if false, traps on error
  /// type.
  bool allowErrorTypes = true;

  /// Creates a TypeOption for use in diagnostics
  static TypePrintOptions forDiagnostics() {
    TypePrintOptions opts;
    opts.printTypeVariablesAsUnderscore = true;
    opts.allowErrorTypes = false;
    opts.allowNullTypes = false;
    return opts;
  }

  /// Creates a TypeOption for use in debugging. This allows null types and
  /// error types.
  static TypePrintOptions forDebug() {
    TypePrintOptions opts;
    opts.printLValues = true;
    return opts;
  }
};

/// Wrapper around a TypeBase* used to disable direct pointer comparison, as it
/// can cause bugs when canonical types are involved.
class Type {
  TypeBase *ptr = nullptr;

public:
  /// Creates an invalid Type
  Type() = default;

  /// Creates a Type from a TypeBase pointer
  /*implicit*/ Type(TypeBase *ptr) : ptr(ptr) {}

  TypeBase *getPtr() const { return ptr; }

  bool isNull() const { return ptr == nullptr; }

  TypeBase *operator->() const {
    assert(ptr && "cannot use this on a null pointer");
    return ptr;
  }

  TypeBase &operator*() const {
    assert(ptr && "cannot use this on a null pointer");
    return *ptr;
  }

  explicit operator bool() const { return ptr != nullptr; }

  /// Prints this type
  void print(raw_ostream &out,
             const TypePrintOptions &printOptions = TypePrintOptions()) const;

  /// Prints this type to a string
  std::string
  getString(const TypePrintOptions &printOptions = TypePrintOptions()) const;

  bool operator<(const Type &other) const { return ptr < other.ptr; }
  // Can't compare types unless they're known canonical
  bool operator==(const Type &) const = delete;
  bool operator!=(const Type &) const = delete;
};

/// Represents a type that's statically known to be canonical (can also be
/// null).
class CanType final : public Type {
  bool isValid() const;

public:
  explicit CanType(Type type) : Type(type) { assert(isValid()); }

  // Can compare CanTypes because they're known canonical
  bool operator==(const CanType &other) const {
    return getPtr() == other.getPtr();
  }
  bool operator!=(const CanType &other) const {
    return getPtr() != other.getPtr();
  }
};

/// A simple Type/TypeRepr* pair, used to represent a type as written
/// down by the user.
///
/// This may not always have a valid TypeRepr*, because it can be used
/// in places where an explicit type is optional.
/// For instance, a TypeLoc inside a ParamDecl will always have a TypeRepr
/// because the type annotation is mandatory, but it may not have a TypeRepr
/// inside a VarDecl, because the type annotation is not mandatory for
/// variable declarations.
class TypeLoc {
  Type type;
  TypeRepr *tyRepr = nullptr;

public:
  TypeLoc() = default;
  TypeLoc(Type type) : TypeLoc(nullptr, type) {}
  TypeLoc(TypeRepr *tyRepr, Type type = Type()) : type(type), tyRepr(tyRepr) {}

  SourceRange getSourceRange() const;
  SourceLoc getBegLoc() const;
  SourceLoc getLoc() const;
  SourceLoc getEndLoc() const;

  bool hasLocation() const { return tyRepr != nullptr; }
  bool hasType() const { return !type.isNull(); }

  TypeRepr *getTypeRepr() const { return tyRepr; }

  Type getType() const { return type; }
  void setType(Type type) { this->type = type; }
};
} // namespace sora

namespace llvm {
// A Type is just a wrapper around a TypeBase*, so it can safely be considered
// as pointer-like.
template <> struct PointerLikeTypeTraits<::sora::Type> {
public:
  enum { NumLowBitsAvailable = ::sora::TypeBaseFreeLowBits };

  static inline void *getAsVoidPointer(::sora::Type type) {
    return type.getPtr();
  }

  static inline ::sora::Type getFromVoidPointer(void *ptr) {
    return (::sora::TypeBase *)ptr;
  }
};

/// Same for CanType
template <>
struct PointerLikeTypeTraits<::sora::CanType>
    : public PointerLikeTypeTraits<::sora::Type> {
public:
  static inline ::sora::CanType getFromVoidPointer(void *ptr) {
    return ::sora::CanType((::sora::TypeBase *)ptr);
  }
};
} // namespace llvm