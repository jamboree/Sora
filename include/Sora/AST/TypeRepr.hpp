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
#include "Sora/Common/InlineBitfields.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
class ASTWalker;
class Expr;

enum class TypeReprKind : uint8_t {
#define TYPEREPR(KIND, PARENT) KIND,
#define LAST_TYPEREPR(KIND) Last_TypeRepr = KIND
#include "Sora/AST/TypeReprNodes.def"
};

/// Base class for type representations
class alignas(TypeReprAlignement) TypeRepr {
  // Disable vanilla new/delete for patterns
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

protected:
  /// Number of bits needed for TypeReprKind
  static constexpr unsigned kindBits =
      countBitsUsed((unsigned)TypeReprKind::Last_TypeRepr);

  union Bits {
    Bits() : rawBits(0) {}
    uint64_t rawBits;

    // clang-format off

    // TypeRepr
    SORA_INLINE_BITFIELD_BASE(TypeRepr, kindBits, 
      kind : kindBits
    );

    // TupleTypeRepr 
    SORA_INLINE_BITFIELD_FULL(TupleTypeRepr, TypeRepr, 32, 
      : NumPadBits, 
      numElements : 32
    );

    // clang-format on
  } bits;
  static_assert(sizeof(Bits) == 8, "Bits is too large!");

  // Derived classes should be able to use placement new, as it is needed for
  // classes with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  TypeRepr(TypeReprKind kind) { bits.TypeRepr.kind = (uint64_t)kind; }

public:
  // Publicly allow allocation of patterns using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(TypeRepr));

  /// \return the kind of TypeRepr this is
  TypeReprKind getKind() const { return TypeReprKind(bits.TypeRepr.kind); }

  /// Skips parentheses around this TypeRepr: If this is a ParenTypeRepr,
  /// returns getSubTypeRepr()->ignoreParens(), else returns this.
  TypeRepr *ignoreParens();
  const TypeRepr *ignoreParens() const {
    return const_cast<TypeRepr *>(this)->ignoreParens();
  }

  /// Traverse this TypeRepr using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
  bool walk(ASTWalker &&walker) { return walk(walker); }

  /// Dumps this TypeRepr to \p out
  void dump(raw_ostream &out, const SourceManager *srcMgr = nullptr,
            unsigned indent = 2) const;
  /// Dumps this TypeRepr to llvm::dbgs(), using default options.
  void dump() const;

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

/// Represents parentheses around another type.
///
/// \verbatim
///   (i32)
/// \endverbatim
class ParenTypeRepr final : public TypeRepr {
  SourceLoc lParenLoc, rParenLoc;
  TypeRepr *subTyRepr;

public:
  ParenTypeRepr(SourceLoc lParenLoc, TypeRepr *subTyRepr, SourceLoc rParenLoc)
      : TypeRepr(TypeReprKind::Paren), lParenLoc(lParenLoc),
        rParenLoc(rParenLoc), subTyRepr(subTyRepr) {}

  TypeRepr *getSubTypeRepr() const { return subTyRepr; }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getLoc() const { return subTyRepr->getLoc(); }
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const TypeRepr *typeRepr) {
    return typeRepr->getKind() == TypeReprKind::Paren;
  }
};

/// Represents a tuple type of 0 or 2+ types.
///
/// \verbatim
///   ()
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
    assert(elements.size() != 1 &&
           "Single-element tuples don't exist - Use ParenTypeRepr!");
    bits.TupleTypeRepr.numElements = elements.size();
    assert(getNumElements() == elements.size() && "Bits dropped");
    std::uninitialized_copy(elements.begin(), elements.end(),
                            getTrailingObjects<TypeRepr *>());
  }

public:
  /// Creates a TupleTypeRepr. Note that \p elements must contain either zero
  /// elements, or 2+ elements. It can't contain a single element as one-element
  /// tuple types don't exist in Sora (There's no way to write them). Things
  /// like "(type)" are represented using a ParenTypeRepr instead.
  static TupleTypeRepr *create(ASTContext &ctxt, SourceLoc lParenLoc,
                               ArrayRef<TypeRepr *> elements,
                               SourceLoc rParenLoc);

  /// Creates an empty TupleTypeRepr
  static TupleTypeRepr *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                    SourceLoc rParenLoc) {
    return create(ctxt, lParenLoc, {}, rParenLoc);
  }

  bool isEmpty() const { return getNumElements() == 0; }
  size_t getNumElements() const {
    return (size_t)bits.TupleTypeRepr.numElements;
  }
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

/// Represents a reference type.
///
/// \verbatim
/// &T
/// &mut T
/// \endverbatim
class ReferenceTypeRepr final : public TypeRepr {
  SourceLoc ampLoc, mutLoc;
  TypeRepr *subTyRepr;

public:
  /// \param mutLoc the SourceLoc of the "mut" keyword. If invalid, the
  /// reference is considered immutable.
  ReferenceTypeRepr(SourceLoc ampLoc, SourceLoc mutLoc, TypeRepr *subTyRepr)
      : TypeRepr(TypeReprKind::Reference), ampLoc(ampLoc), mutLoc(mutLoc),
        subTyRepr(subTyRepr) {}

  ReferenceTypeRepr(SourceLoc ampLoc, TypeRepr *subTyRepr)
      : ReferenceTypeRepr(ampLoc, SourceLoc(), subTyRepr) {}

  TypeRepr *getSubTypeRepr() const { return subTyRepr; }

  /// \returns the sourceloc of the &
  SourceLoc getAmpLoc() const { return ampLoc; }

  bool hasMut() const { return mutLoc.isValid(); }
  SourceLoc getMutLoc() const { return mutLoc; }

  SourceLoc getBegLoc() const { return ampLoc; }
  SourceLoc getEndLoc() const { return subTyRepr->getEndLoc(); }

  static bool classof(const TypeRepr *typeRepr) {
    return typeRepr->getKind() == TypeReprKind::Reference;
  }
};

/// Represents a "maybe" type
///
/// \verbatim
///   maybe i32
///   maybe &mut i64
/// \endverbatim
class MaybeTypeRepr final : public TypeRepr {
  SourceLoc maybeLoc;
  TypeRepr *subTyRepr;

public:
  MaybeTypeRepr(SourceLoc maybeLoc, TypeRepr *subTyRepr)
      : TypeRepr(TypeReprKind::Maybe), maybeLoc(maybeLoc),
        subTyRepr(subTyRepr) {}

  SourceLoc getMaybeLoc() const { return maybeLoc; }

  TypeRepr *getSubTypeRepr() const { return subTyRepr; }

  SourceLoc getBegLoc() const { return maybeLoc; }
  SourceLoc getEndLoc() const { return subTyRepr->getEndLoc(); }

  static bool classof(const TypeRepr *typeRepr) {
    return typeRepr->getKind() == TypeReprKind::Maybe;
  }
};
} // namespace sora