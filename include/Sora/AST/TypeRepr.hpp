//===--- TypeRepr.hpp - Type Representation ASTs ----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// TypeReprs are, literally, Type Representations. They represent types as
// they were written by the user. TypeReprs aren't types, but they are converted
// (resolved) to types during semantic analysis.
//
// This is used to have a representation of types that are source-accurate and
// include source location information.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;

enum class TypeReprKind : uint8_t {
#define TYPEREPR(KIND, PARENT) KIND,
#include "Sora/AST/TypeReprNodes.def"
};

/// Base class for type representations
class alignas(TypeReprAlignement) TypeRepr {
  // Disable vanilla new/delete for patterns
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  TypeReprKind kind;
  /// Make use of the padding bits by allowing derived class to store data here.
  /// NOTE: Derived classes are expected to initialize the bitfield themselves.
  LLVM_PACKED(union Bits {
    Bits() : raw() {}
    // Raw bits (to zero-init the union)
    char raw[7];
    // TupleTypeRepr
    struct {
      uint32_t numElements;
    } tupleTypeRepr;
  });
  static_assert(sizeof(Bits) == 7, "Bits is too large!");

protected:
  Bits bits;

  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  TypeRepr(TypeReprKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of patterns using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(TypeRepr));

  /// \return the kind of TypeRepr this is
  TypeReprKind getKind() const { return kind; }

  /// \returns the SourceLoc of the first token of the TypeRepr
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the TypeRepr
  SourceLoc getEndLoc() const;
  /// \returns the preffered SourceLoc for diagnostics. This is defaults to
  /// getBegLoc but nodes can override it as they please.
  SourceLoc getLoc() const;
  /// \returns the full range of this TypeRepr
  SourceRange getSourceRange() const;
};

/// We should only use 8 bytes (1 pointers) max in 64 bits mode because we only
/// store the kind + some packed bits in the base class.
static_assert(sizeof(TypeRepr) <= 16, "TypeRepr is too big!");

/// Represents an identifier
///
/// \verbatim
///   i32
///   Foo
///   bool
/// \endverbatim
class IdentifierTypeRepr final : public TypeRepr {
  SourceLoc identLoc;
  Identifier ident;

public:
  IdentifierTypeRepr(SourceLoc identLoc, Identifier ident)
      : TypeRepr(TypeReprKind::Identifier), identLoc(identLoc), ident(ident) {}

  Identifier getIdentifier() const { return ident; }
  SourceLoc getIdentifierLoc() const { return identLoc; }

  SourceLoc getBegLoc() const { return identLoc; }
  SourceLoc getEndLoc() const { return identLoc; }

  static bool classof(const TypeRepr *typeRepr) {
    return typeRepr->getKind() == TypeReprKind::Identifier;
  }
};

/// Represents a tuple type.
///
/// \verbatim
///   ()
///   (a)
///   (a, b)
///   (a, b, c)
/// \endverbatim
class TupleTypeRepr final
    : public TypeRepr,
      private llvm::TrailingObjects<TupleTypeRepr, TypeRepr *> {
  friend llvm::TrailingObjects<TupleTypeRepr, TypeRepr *>;

  SourceLoc lParenLoc, rParenLoc;

  TupleTypeRepr(SourceLoc lParenLoc, ArrayRef<TypeRepr *> elements,
                SourceLoc rParenLoc)
      : TypeRepr(TypeReprKind::Tuple), lParenLoc(lParenLoc),
        rParenLoc(rParenLoc) {
    bits.tupleTypeRepr.numElements = elements.size();
    std::uninitialized_copy(elements.begin(), elements.end(),
                            getTrailingObjects<TypeRepr *>());
  }

public:
  /// Creates a TupleTypeRepr
  static TupleTypeRepr *create(ASTContext &ctxt, SourceLoc lParenLoc,
                               ArrayRef<TypeRepr *> elements,
                               SourceLoc rParenLoc);

  /// Creates an empty TupleTypeRepr
  static TupleTypeRepr *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                    SourceLoc rParenLoc) {
    return create(ctxt, lParenLoc, {}, rParenLoc);
  }

  size_t getNumElements() const { return bits.tupleTypeRepr.numElements; }
  ArrayRef<TypeRepr *> getElements() const {
    return {getTrailingObjects<TypeRepr *>(), getNumElements()};
  }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const TypeRepr *typeRepr) {
    return typeRepr->getKind() == TypeReprKind::Tuple;
  }
};

/// Represent a pointer or reference type.
///
/// \verbatim
/// &T
/// &mut T
/// *T
/// *mut T
/// \endverbatim
class PointerTypeRepr final : public TypeRepr {
  SourceLoc signLoc, mutLoc;
  TypeRepr *subTyRepr;

public:
  PointerTypeRepr(SourceLoc signLoc, SourceLoc mutLoc, TypeRepr *subTyRepr)
      : TypeRepr(TypeReprKind::Pointer), signLoc(signLoc), mutLoc(mutLoc),
        subTyRepr(subTyRepr) {}

  PointerTypeRepr(SourceLoc signLoc, TypeRepr *subTyRepr)
      : PointerTypeRepr(signLoc, SourceLoc(), subTyRepr) {}

  TypeRepr *getSubTypeRepr() const { return subTyRepr; }
  /// \returns the SourceLoc of the & or * sign.
  SourceLoc getSignLoc() const { return signLoc; }

  bool hasMut() const { return mutLoc.isValid(); }

  SourceLoc getMutLoc() const { return mutLoc; }

  SourceLoc getBegLoc() const { return signLoc; }
  SourceLoc getEndLoc() const { return subTyRepr->getEndLoc(); }

  static bool classof(const TypeRepr *typeRepr) {
    return typeRepr->getKind() == TypeReprKind::Pointer;
  }
};
} // namespace sora