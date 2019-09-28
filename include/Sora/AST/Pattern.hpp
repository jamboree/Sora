//===--- Pattern.hpp - Pattern ASTs -----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
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
#include "Sora/AST/PatternNodes.def"
};

/// Base class for every Pattern node
class alignas(PatternAlignement) Pattern {
  // Disable vanilla new/delete for patterns
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  PatternKind kind;
  /// Make use of the padding bits by allowing derived class to store data here.
  /// NOTE: Derived classes are expected to initialize the bitfield themselves.
  LLVM_PACKED(union Bits {
    Bits() : raw() {}
    // Raw bits (to zero-init the union)
    char raw[7];
    // TuplePattern
    struct {
      uint32_t numElements;
    } tuplePattern;
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

  Pattern(PatternKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of patterns using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Pattern));

  /// Traverse this Pattern using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);

  /// Dumps this statement to \p out
  void dump(raw_ostream &out, const SourceManager &srcMgr,
            unsigned indent = 2) const;

  /// \return the kind of patterns this is
  PatternKind getKind() const { return kind; }

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

/// We should only use 8 bytes (1 pointers) max in 64 bits mode because we only
/// store the kind + some packed bits in the base class.
static_assert(sizeof(Pattern) <= 16, "Pattern is too big!");

/// Represents a single variable pattern, which matches any argument and binds
/// it to a name. This is the node that contains VarDecl nodes. A VarDecl* must
/// be given when constructing the object and it can't be changed afterwards.
class VarPattern final : public Pattern {
  VarDecl *const varDecl = nullptr;

public:
  VarPattern(VarDecl *varDecl) : Pattern(PatternKind::Var), varDecl(varDecl) {}

  VarDecl *getVarDecl() const { return varDecl; }
  Identifier getIdentifier() const;

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

public:
  DiscardPattern(SourceLoc loc) : Pattern(PatternKind::Discard), loc(loc) {}

  SourceLoc getBegLoc() const { return loc; }
  SourceLoc getEndLoc() const { return loc; }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Discard;
  }
};

/// Represents a "mut" pattern which indicates that, in the following pattern,
/// every variable introduced should be mutable.
class MutPattern final : public Pattern {
  SourceLoc mutLoc;
  Pattern *subPattern = nullptr;

public:
  MutPattern(SourceLoc mutLoc, Pattern *subPattern)
      : Pattern(PatternKind::Mut), mutLoc(mutLoc), subPattern(subPattern) {}

  SourceLoc getMutLoc() const { return mutLoc; }
  Pattern *getSubPattern() const { return subPattern; }

  SourceLoc getBegLoc() const { return mutLoc; }
  SourceLoc getLoc() const { return subPattern->getLoc(); }
  SourceLoc getEndLoc() const { return subPattern->getEndLoc(); }

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Mut;
  }
};

/// Represents a Tuple pattern, which is a group of 0 or more patterns
/// in parentheses.
class TuplePattern final
    : public Pattern,
      private llvm::TrailingObjects<TuplePattern, Pattern *> {
  friend llvm::TrailingObjects<TuplePattern, Pattern *>;

  SourceLoc lParenLoc, rParenloc;

  TuplePattern(SourceLoc lParenLoc, ArrayRef<Pattern *> patterns,
               SourceLoc rParenloc)
      : Pattern(PatternKind::Tuple), lParenLoc(lParenLoc),
        rParenloc(rParenloc) {
    bits.tuplePattern.numElements = patterns.size();
    std::uninitialized_copy(patterns.begin(), patterns.end(),
                            getTrailingObjects<Pattern *>());
  }

public:
  /// Creates a tuple pattern
  static TuplePattern *create(ASTContext &ctxt, SourceLoc lParenLoc,
                              ArrayRef<Pattern *> patterns,
                              SourceLoc rParenLoc);

  /// Creates an empty tuple pattern
  static TuplePattern *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                   SourceLoc rParenLoc) {
    return create(ctxt, lParenLoc, {}, rParenLoc);
  }

  size_t getNumElements() const { return bits.tuplePattern.numElements; }
  ArrayRef<Pattern *> getElements() const {
    return {getTrailingObjects<Pattern *>(), getNumElements()};
  }
  MutableArrayRef<Pattern *> getElements() {
    return {getTrailingObjects<Pattern *>(), getNumElements()};
  }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenloc; }

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getEndLoc() const { return rParenloc; }

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
  TypeRepr *typeRepr;

public:
  TypedPattern(Pattern *subPattern, TypeRepr *typeRepr)
      : Pattern(PatternKind::Typed), subPattern(subPattern),
        typeRepr(typeRepr) {}

  Pattern *getSubPattern() const { return subPattern; }
  TypeRepr *getTypeRepr() const { return typeRepr; }

  SourceLoc getBegLoc() const;
  SourceLoc getLoc() const { return subPattern->getLoc(); }
  SourceLoc getEndLoc() const;

  static bool classof(const Pattern *pattern) {
    return pattern->getKind() == PatternKind::Typed;
  }
};
} // namespace sora