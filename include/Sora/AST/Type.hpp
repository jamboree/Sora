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

/// Lightweight class that represents type printing options.
///
/// By default, this is configured to print types with as much information as
/// possible: lvalues are printed, null types are allowed, etc.
class TypePrintOptions {
  // Private - use the static constructors instead.
  TypePrintOptions() {}

public:
  /// If true, null types are printed as <null_type>, if false, traps on null
  /// type.
  bool allowNullTypes : 1;
  /// If true, lvalues are printed as @lvalue T instead of being "transparent"
  bool printLValues : 1;
  /// If true, null types are printed as <error_type>, if false, traps on error
  /// type.
  bool allowErrorTypes : 1;
  /// If true, all type variables are printed as
  /// "'$' ('T' | 'I' | 'F') id ('(' binding ')')?".
  bool debugTypeVariables : 1;
  /// Print the binding of bound type variables instead of using '_' or \c
  /// printDefaultForUnboundTypeVariables. This is ignored if \c
  /// debugTypeVariables is true.
  bool printBoundTypeVariablesAsBinding : 1;
  /// For unbound TypeVariables, print their default type or '_' if there is no
  /// default. This is ignored if \c debugTypeVariables is true.
  bool printDefaultForUnboundTypeVariables : 1;

  /// Creates a TypeOption for use in diagnostics
  static TypePrintOptions forDiagnostics() {
    TypePrintOptions opts;
    opts.debugTypeVariables = false;
    opts.printBoundTypeVariablesAsBinding = true;
    opts.printDefaultForUnboundTypeVariables = true;
    opts.allowErrorTypes = false;
    opts.allowNullTypes = false;
    opts.printLValues = false;
    return opts;
  }

  /// Creates a TypeOption for use in debug messages.
  static TypePrintOptions forDebug() {
    TypePrintOptions opts;
    opts.debugTypeVariables = true;
    opts.printBoundTypeVariablesAsBinding = false;
    opts.printDefaultForUnboundTypeVariables = false;
    opts.allowErrorTypes = true;
    opts.allowNullTypes = true;
    opts.printLValues = true;
    return opts;
  }
};

static_assert(sizeof(TypePrintOptions) == 1, "TypePrintOptions is too large!");

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
  void
  print(raw_ostream &out,
        TypePrintOptions printOptions = TypePrintOptions::forDebug()) const;

  /// Prints this type to a string
  std::string
  getString(TypePrintOptions printOptions = TypePrintOptions::forDebug()) const;

  bool operator<(const Type &other) const { return ptr < other.ptr; }
  // Can't compare types unless they're known canonical
  bool operator==(const Type &) const = delete;
  bool operator!=(const Type &) const = delete;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &out, Type type) {
  type.print(out);
  return out;
}

/// Represents a type that's statically known to be canonical (can also be
/// null).
class CanType final : public Type {
  bool isValid() const;

public:
  explicit CanType(std::nullptr_t) : CanType(Type(nullptr)) {}
  explicit CanType(Type type) : Type(type) { assert(isValid()); }

  // Can compare CanTypes because they're known canonical
  bool operator==(const CanType &other) const {
    return getPtr() == other.getPtr();
  }
  bool operator!=(const CanType &other) const {
    return getPtr() != other.getPtr();
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &out, CanType type) {
  type.print(out);
  return out;
}

/// A simple Type/TypeRepr* pair, used to represent a type that was explicitely
/// written down by the user.
///
/// TypeLocs either contain a TypeRepr and (optionally) a Type, or nothing. They
/// can never contain just a Type without TypeRepr (see \c isValid)
class TypeLoc {
  Type type;
  TypeRepr *typeRepr = nullptr;

public:
  TypeLoc() = default;
  TypeLoc(TypeRepr *typeRepr, Type type = Type())
      : type(type), typeRepr(typeRepr) {}

  SourceRange getSourceRange() const;
  SourceLoc getBegLoc() const;
  SourceLoc getLoc() const;
  SourceLoc getEndLoc() const;

  /// \returns whether this TypeLoc is valid (whether it has a TypeRepr)
  bool isValid() const { return typeRepr; }

  bool hasType() const { return !type.isNull(); }

  TypeRepr *getTypeRepr() const { return typeRepr; }

  Type getType() const { return type; }
  void setType(Type type) {
    assert(isValid() && "Cannot assign a type to an invalid TypeLoc!");
    this->type = type;
  }
};

template <typename Ty> struct DiagnosticArgument;

template <> struct DiagnosticArgument<Type> {
  static std::string format(Type type);
};

template <> struct DiagnosticArgument<CanType> {
  static std::string format(CanType type) {
    return DiagnosticArgument<Type>::format(type);
  }
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