//===--- Pattern.hpp - Pattern ASTs -----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/InlineBitfields.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>

#include <stdint.h>

namespace sora {
class ASTContext;
class ASTWalker;
class VarDecl;

/// Kinds of Patterns
enum class PatternKind : uint8_t {
#define PATTERN(KIND, PARENT) KIND,
#define PATTERN_RANGE(KIND, FIRST, LAST)                                       \
  First_##KIND = FIRST, Last_##KIND = LAST,
#define LAST_PATTERN(KIND) Last_Pattern = KIND
#include "Sora/AST/PatternNodes.def"
};

/// Base class for every Pattern node.
///
/// This is a fairly simple common base which is, in a way, similar to Expr with
/// one major difference: the type isn't stored in the base class, but inside
/// the derived classes. This means that Pattern does not offer a "setType"
/// method like Expr does. This has the added advantage that some Patterns don't
/// have to store a type at all since it's always the type of the subpattern
/// (see TransparentPatterns), it also allows TypedPattern to store the type
/// only once in its TypeLoc. This makes the storage of the Type as efficient as
/// it can be.
/// (n.b. Such a thing isn't really possible/worthwhile for the Expr tree which
/// is much larger, and whose types can be updated much more often (due to Type
/// inference).
class alignas(PatternAlignement) Pattern {
  // Disable vanilla new/delete for patterns
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

protected:
  /// Number of bits needed for PatternKind
  static constexpr unsigned kindBits =
      countBitsUsed((unsigned)PatternKind::Last_Pattern);

  union Bits {
    Bits() : rawBits(0) {}
    uint64_t rawBits;

    // clang-format off

    // Pattern
    SORA_INLINE_BITFIELD_BASE(Pattern, kindBits+1, 
      kind : kindBits,
      isImplicit : 1
    );

    // TuplePattern
    SORA_INLINE_BITFIELD_FULL(TuplePattern, Pattern, 32, 
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

  Pattern(PatternKind kind) {
    bits.Pattern.kind = (uint64_t)kind;
    bits.Pattern.isImplicit = false;
  }

public:
  // Publicly allow allocation of patterns using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Pattern));

  /// Traverse this Pattern using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
  bool walk(ASTWalker &&walker) { return walk(walker); }

  /// Dumps this pattern to \p out
  void dump(raw_ostream &out, const SourceManager *srcMgr = nullptr,
            unsigned indent = 2) const;
  /// Dumps this pattern to llvm::dbgs(), using default options.
  void dump() const;

  /// \return the kind of patterns this is
  PatternKind getKind() const { return PatternKind(bits.Pattern.kind); }

  void setImplicit(bool value = true) { bits.Pattern.isImplicit = value; }
  bool isImplicit() const { return bits.Pattern.isImplicit; }

  bool hasType() const { return !getType().isNull(); }
  Type getType() const;

  /// Skips parentheses around this Pattern: If this is a ParenPattern, returns
  /// getSubPattern()->ignoreParens(), else returns this.
  Pattern *ignoreParens();
  const Pattern *ignoreParens() const {
    return const_cast<Pattern *>(this)->ignoreParens();
  }

  /// \returns whether this pattern is refutable or not.
  /// This is not a cheap method, it has to traverse the pattern on every call.
  bool isRefutable() const;

  /// Call \p fn on each VarDecl in this Pattern.
  void forEachVarDecl(llvm::function_ref<void(VarDecl *)> fn) const;

  /// Call \p fn on each Pattern in this Pattern in pre-order.
  void forEachNode(llvm::function_ref<void(Pattern *)> fn);

  /// \returns true if this pattern contains a VarPattern.
  bool hasVarPattern() const;

  /// \returns the SourceLoc of the first token of the pattern
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the pattern
  SourceLoc getEndLoc() const;
  /// \returns the preffered SourceLoc for diagnostics. This is defaults to
  /// getBegLoc but nodes can override it as they please.
  SourceLoc getLoc() const;
  /// \returns the full range of this pattern
  SourceRange getSourceRange() const;
};

/// We should only use 16 bytes (2 pointers) max in 64 bits mode because we only
/// store the type, the kind + some packed bits in the base class.
static_assert(sizeof(Pattern) <= 16, "Pattern is too big!");

/// Represents a single variable pattern, which matches any argument and binds
/// it to a name.
///
/// This contains a pointer to the VarDecl it created. This is set when
/// constructing the VarPattern and cannot be changed afterwards.
///
/// The VarDecl is also added to its enclosing node's content (e.g. they'll also
/// become elements of the BraceStmt)
///
/// Usually, VarDecls aren't visited through their VarPattern. They are visited
/// through their enclosing "node", e.g. when walking the elements of a
/// BraceStmt.
class VarPattern final : public Pattern {
  VarDecl *const varDecl = nullptr;
  Type type;

public:
  VarPattern(VarDecl *varDecl) : Pattern(PatternKind::Var), varDecl(varDecl) {}

  VarDecl *getVarDecl() const { return varDecl; }
  Identifier getIdentifier() const;

  Type getType() const { return type; }
  void setType(Type type) { this->type = type; }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Var;
  }
};

/// Represents a wildcard pattern, which matches any argument and discards it.
/// It is spelled '_'.
class DiscardPattern final : public Pattern {
  SourceLoc loc;
  Type type;

public:
  DiscardPattern(SourceLoc loc) : Pattern(PatternKind::Discard), loc(loc) {}

  Type getType() const { return type; }
  void setType(Type type) { this->type = type; }

  SourceLoc getBegLoc() const { return loc; }
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Discard;
  }
};

/// Common base class for patterns that are considered "transparent": patterns
/// that only contain a subpattern and always have its type: fetching the type
/// of this pattern is always equivalent to fetching the type of its subpattern!
class TransparentPattern : public Pattern {
  Pattern *subPattern = nullptr;

protected:
  TransparentPattern(PatternKind kind, Pattern *subPattern)
      : Pattern(kind), subPattern(subPattern) {}

public:
  void setSubPattern(Pattern *pattern) { subPattern = pattern; }
  Pattern *getSubPattern() const { return subPattern; }

  /// \returns the type of this pattern, which is always the type of its
  /// subpattern. If the subpattern is null, return a null Type.
  Type getType() const { return subPattern ? subPattern->getType() : Type(); }
};

/// Represents a "mut" pattern which indicates that, in the following pattern,
/// every variable introduced should be mutable.
class MutPattern final : public TransparentPattern {
  SourceLoc mutLoc;

public:
  MutPattern(SourceLoc mutLoc, Pattern *subPattern)
      : TransparentPattern(PatternKind::Mut, subPattern), mutLoc(mutLoc) {}

  SourceLoc getMutLoc() const { return mutLoc; }

  SourceLoc getBegLoc() const { return mutLoc; }
  SourceLoc getLoc() const { return getSubPattern()->getLoc(); }
  SourceLoc getEndLoc() const { return getSubPattern()->getEndLoc(); }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Mut;
  }
};

/// Represents a pattern that only consists of grouping parentheses around
/// another pattern.
class ParenPattern final : public TransparentPattern {
  SourceLoc lParenLoc, rParenLoc;

public:
  ParenPattern(SourceLoc lParenLoc, Pattern *subPattern, SourceLoc rParenLoc)
      : TransparentPattern(PatternKind::Paren, subPattern),
        lParenLoc(lParenLoc), rParenLoc(rParenLoc) {}

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getLoc() const { return getSubPattern()->getLoc(); }
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Paren;
  }
};

/// Represents a Tuple pattern, which is a group of 0 or more patterns
/// in parentheses. Note that there are no single-element tuples, so patterns
/// like "(mut x)" are simply represented using ParenPattern.
class TuplePattern final
    : public Pattern,
      private llvm::TrailingObjects<TuplePattern, Pattern *> {
  friend llvm::TrailingObjects<TuplePattern, Pattern *>;

  SourceLoc lParenLoc, rParenLoc;
  Type type;

  TuplePattern(SourceLoc lParenLoc, ArrayRef<Pattern *> patterns,
               SourceLoc rParenLoc)
      : Pattern(PatternKind::Tuple), lParenLoc(lParenLoc),
        rParenLoc(rParenLoc) {
    assert(patterns.size() != 1 &&
           "Single-element tuples don't exist - Use ParenPattern!");
    bits.TuplePattern.numElements = patterns.size();
    assert(getNumElements() == patterns.size() && "Bits dropped");
    std::uninitialized_copy(patterns.begin(), patterns.end(),
                            getTrailingObjects<Pattern *>());
  }

public:
  /// Creates a TuplePattern. Note that \p patterns must contain either zero
  /// elements, or 2+ elements. It can't contain a single element as one-element
  /// tuples don't exist in Sora (There's no way to write them). Things like
  /// "(pattern)" are represented using a ParenPattern instead.
  static TuplePattern *create(ASTContext &ctxt, SourceLoc lParenLoc,
                              ArrayRef<Pattern *> patterns,
                              SourceLoc rParenLoc);

  /// Creates an empty tuple pattern
  static TuplePattern *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                   SourceLoc rParenLoc) {
    return create(ctxt, lParenLoc, {}, rParenLoc);
  }

  Type getType() const { return type; }
  void setType(Type type) { this->type = type; }

  bool isEmpty() const { return getNumElements() == 0; }
  size_t getNumElements() const {
    return (size_t)bits.TuplePattern.numElements;
  }
  ArrayRef<Pattern *> getElements() const {
    return {getTrailingObjects<Pattern *>(), getNumElements()};
  }
  MutableArrayRef<Pattern *> getElements() {
    return {getTrailingObjects<Pattern *>(), getNumElements()};
  }
  void setElement(size_t n, Pattern *pattern) { getElements()[n] = pattern; }
  Pattern *getElement(size_t n) const { return getElements()[n]; }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenLoc; }

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getEndLoc() const { return rParenLoc; }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Tuple;
  }
};

/// Represents a "typed" pattern, which is a pattern followed by a type
/// annotation
///
/// \verbatim
///   a : i32
///   (a, b) : (i32, i32)
/// \endverbatim
class TypedPattern final : public Pattern {
  Pattern *subPattern;
  TypeLoc typeLoc;

public:
  TypedPattern(Pattern *subPattern, TypeLoc typeLoc)
      : Pattern(PatternKind::Typed), subPattern(subPattern), typeLoc(typeLoc) {}

  void setSubPattern(Pattern *pattern) { subPattern = pattern; }
  Pattern *getSubPattern() const { return subPattern; }

  Type getType() const { return typeLoc.getType(); }

  TypeLoc &getTypeLoc() { return typeLoc; }
  const TypeLoc &getTypeLoc() const { return typeLoc; }

  SourceLoc getBegLoc() const { return subPattern->getBegLoc(); }
  SourceLoc getLoc() const { return subPattern->getLoc(); }
  SourceLoc getEndLoc() const { return typeLoc.getEndLoc(); }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Typed;
  }
};

/// Common base class for refutable patterns (= patterns that don't always match
/// and thus can fail at runtime)
class RefutablePattern : public Pattern {
protected:
  RefutablePattern(PatternKind kind) : Pattern(kind) {}

public:
  static bool classof(const Pattern *pattern) {
    return (pattern->getKind() >= PatternKind::First_Refutable) &&
           (pattern->getKind() <= PatternKind::Last_Refutable);
  }
};

/// Pattern that matches when a "maybe" type contains a value.
///
/// Currently, this pattern is always implicit and can't be explicitly written
/// in source. It is generated for "if let" conditions.
///
/// This pattern can only be present at the top-level of a pattern, it can not
/// be the children of another pattern.
class MaybeValuePattern : public RefutablePattern {
  Pattern *subPattern;
  Type type;

public:
  MaybeValuePattern(Pattern *subPattern, bool isImplicit = false)
      : RefutablePattern(PatternKind::MaybeValue), subPattern(subPattern) {
    setImplicit(isImplicit);
  }

  Type getType() const { return type; }
  void setType(Type type) { this->type = type; }

  void setSubPattern(Pattern *pattern) { subPattern = pattern; }
  Pattern *getSubPattern() const { return subPattern; }

  SourceLoc getBegLoc() const { return subPattern->getBegLoc(); }
  SourceLoc getLoc() const { return subPattern->getLoc(); }
  SourceLoc getEndLoc() const { return subPattern->getEndLoc(); }
  SourceRange getSourceRange() const { return subPattern->getSourceRange(); }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::MaybeValue;
  }
};
} // namespace sora