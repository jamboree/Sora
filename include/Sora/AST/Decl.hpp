//===--- Decl.hpp - Declarations ASTs ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/DeclContext.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/InlineBitfields.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
class ASTWalker;
class DiagnosticEngine;
class PatternBindingDecl;
class BlockStmt;

/// Kinds of Declarations
enum class DeclKind : uint8_t {
#define DECL(KIND, PARENT) KIND,
#define DECL_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#define LAST_DECL(KIND) Last_Decl = KIND
#include "Sora/AST/DeclNodes.def"
};

/// Base class for every Declaration node.
class alignas(DeclAlignement) Decl {
  // Disable vanilla new/delete for declarations
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  // The parent DeclContext and a flag indicating whether this decl has been
  // checked.
  llvm::PointerIntPair<DeclContext *, 1> declCtxtAndIsChecked;

protected:
  /// Number of bits needed for ExprKind
  static constexpr unsigned kindBits =
      countBitsUsed((unsigned)DeclKind::Last_Decl);

  union Bits {
    Bits() : rawBits() {}
    uint64_t rawBits;

    // clang-format off

    // Decl
    SORA_INLINE_BITFIELD_BASE(Decl, kindBits, 
      kind : kindBits
    );

    // NamedDecl
    SORA_INLINE_BITFIELD(NamedDecl, Decl, 1, 
      isIllegalRedeclaration : 1
    );

    // ValueDecl
    SORA_INLINE_BITFIELD_EMPTY(ValueDecl, NamedDecl);

    // VarDecl
    SORA_INLINE_BITFIELD(VarDecl, ValueDecl, 1,
      isMutable : 1
    );

    // FuncDecl
    SORA_INLINE_BITFIELD(FuncDecl, ValueDecl, 1,
      isBodyChecked : 1
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

  Decl(DeclKind kind, DeclContext *declContext)
      : declCtxtAndIsChecked(declContext, false) {
    bits.Decl.kind = (uint64_t)kind;
  }

public:
  // Publicly allow allocation of declaration using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Decl));

  /// \returns whether this Decl has been type-checked
  bool isChecked() const { return declCtxtAndIsChecked.getInt(); }
  /// Sets the flag indicating whether this Decl has been type-checked
  void setChecked(bool value = true) { declCtxtAndIsChecked.setInt(value); }

  /// Fetches the parent source file using the parents.
  /// This requires a well-formed parent chain.
  SourceFile &getSourceFile() const;

  /// Fetches the ASTContext using the parents.
  /// This requires a well-formed parent chain.
  ASTContext &getASTContext() const;

  /// Fetches the DiagnosticEngine using the parents.
  /// This requires a well-formed parent chain.
  DiagnosticEngine &getDiagnosticEngine() const;

  /// \returns true if this is a local declaration. A local decl is a decl that
  /// lives inside a FuncDecl.
  bool isLocal() const;

  /// \returns the DeclContext in which this Decl is contained
  DeclContext *getDeclContext() const {
    return declCtxtAndIsChecked.getPointer();
  }

  /// If this Decl is also a DeclContext, returns it as a DeclContext*, else
  /// returns nullptr.
  DeclContext *getAsDeclContext() { return dyn_cast<DeclContext>(this); }

  /// If this Decl is also a DeclContext, returns it as a DeclContext*, else
  /// returns nullptr.
  const DeclContext *getAsDeclContext() const {
    return dyn_cast<DeclContext>(this);
  }

  /// Traverse this Decl using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
  bool walk(ASTWalker &&walker) { return walk(walker); }

  /// Dumps this declaration to \p out
  void dump(raw_ostream &out, unsigned indent = 2) const;

  /// \return the kind of declaration this is
  DeclKind getKind() const { return DeclKind(bits.Decl.kind); }

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const;
  /// \returns the preffered SourceLoc for diagnostics. This is defaults to
  /// getBegLoc but nodes can override it as they please.
  SourceLoc getLoc() const;
  /// \returns the full range of this declaration
  SourceRange getSourceRange() const;
};

/// Decl should be 2 pointers in size: parent + kind + padding bits
static_assert(sizeof(Decl) <= 16, "Decl is too large!");

inline bool DeclContext::classof(const Decl *decl) {
  switch (decl->getKind()) {
  default:
    return false;
  case DeclKind::Func:
    return true;
  }
}

/// Common base class for named declarations (= declarations that introduce a
/// new name).
class NamedDecl : public Decl {
  SourceLoc identifierLoc;
  Identifier identifier;

protected:
  NamedDecl(DeclKind kind, DeclContext *declContext, SourceLoc identifierLoc,
            Identifier identifier)
      : Decl(kind, declContext), identifierLoc(identifierLoc),
        identifier(identifier) {
    bits.NamedDecl.isIllegalRedeclaration = false;
  }

public:
  /// \returns whether this NamedDecl is an illegal redeclaration or not.
  /// Returns false if \c isChecked returns false.
  bool isIllegalRedeclaration() const {
    return bits.NamedDecl.isIllegalRedeclaration;
  }
  /// Sets the flag indicating whether this NamedDecl is an illegal
  /// redeclaration.
  void setIsIllegalRedeclaration(bool value = true) {
    bits.NamedDecl.isIllegalRedeclaration = value;
  }

  /// \returns the identifier (name) of this decl
  Identifier getIdentifier() const { return identifier; }

  /// \returns the SourceLoc of the identifier
  SourceLoc getIdentifierLoc() const { return identifierLoc; }

  /// \returns the preffered SourceLoc for diagnostics.
  SourceLoc getLoc() const { return identifierLoc; }

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::First_Named &&
           decl->getKind() <= DeclKind::Last_Named;
  }
};

/// Base class for declarations that declare a value of some type with a
/// given name.
///
/// As an implementation detail, the type is not stored in the ValueDecl,
/// but inside the derived classes. This works in the same way as
/// "getBegLoc" for instance. This is done so we don't waste space storing a
/// copy of the type in this class. e.g. VarDecl needs to store a TypeLoc,
/// and we don't want to store the type in the TypeLoc AND in the ValueDecl,
/// it'd waste space for nothing.
class ValueDecl : public NamedDecl {

protected:
  ValueDecl(DeclKind kind, DeclContext *declContext, SourceLoc identifierLoc,
            Identifier identifier)
      : NamedDecl(kind, declContext, identifierLoc, identifier) {}

public:
  /// \returns the type of this value
  Type getValueType() const;

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::First_Value &&
           decl->getKind() <= DeclKind::Last_Value;
  }
};

/// Represents a *single* variable declaration. This is always the child
/// of a VarPattern.
///
/// This DOES NOT represent something such as "let x" entirely, it only
/// represents the "x".
class VarDecl final : public ValueDecl {
  Type type;

public:
  VarDecl(DeclContext *declContext, SourceLoc identifierLoc,
          Identifier identifier)
      : ValueDecl(DeclKind::Var, declContext, identifierLoc, identifier) {
    bits.VarDecl.isMutable = false;
  }

  /// Sets the type of this value (the type of the variable)
  void setValueType(Type type) { this->type = type; }
  /// \returns the type of this value (the type of the variable)
  Type getValueType() const { return type; }

  bool isMutable() const { return bits.VarDecl.isMutable; }
  void setIsMutable(bool value = true) { bits.VarDecl.isMutable = value; }

  SourceLoc getBegLoc() const { return getIdentifierLoc(); }
  SourceLoc getEndLoc() const { return getIdentifierLoc(); }

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Var;
  }
};

/// Function parameter declarations
class ParamDecl final : public ValueDecl {
  TypeLoc tyLoc;

public:
  ParamDecl(DeclContext *declContext, SourceLoc identifierLoc,
            Identifier identifier, TypeLoc typeLoc)
      : ValueDecl(DeclKind::Param, declContext, identifierLoc, identifier),
        tyLoc(typeLoc) {}

  /// \returns the TypeLoc of the ParamDecl's type.
  /// This should always have a valid TypeRepr*.
  /// The type should be valid after semantic analysis.
  TypeLoc &getTypeLoc() { return tyLoc; }
  /// \returns the TypeLoc of the ParamDecl's type.
  /// This should always have a valid TypeRepr*.
  /// The type should be valid after semantic analysis.
  TypeLoc getTypeLoc() const { return tyLoc; }

  /// \returns the type of this value (the type of the parameter)
  Type getValueType() const { return tyLoc.getType(); }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Param;
  }
};

/// Represents a list of Parameter Declarations.
/// This list cannot be changed once created (you can't add/remove
/// parameter or change the pointer's value, but you can change the
/// ParamDecl themselves)
class ParamList final : private llvm::TrailingObjects<ParamList, ParamDecl *> {
  friend llvm::TrailingObjects<ParamList, ParamDecl *>;

  void *operator new(size_t size) throw() = delete;
  void operator delete(void *mem) throw() = delete;
  void *operator new(size_t size, void *mem) throw() {
    assert(mem);
    return mem;
  }

  SourceLoc lParenLoc, rParenloc;
  size_t numParams = 0;

  ParamList(SourceLoc lParenLoc, ArrayRef<ParamDecl *> params,
            SourceLoc rParenloc);

public:
  using iterator = ArrayRef<ParamDecl *>::iterator;

  /// Creates a parameter list
  static ParamList *create(ASTContext &ctxt, SourceLoc lParenLoc,
                           ArrayRef<ParamDecl *> params, SourceLoc rParenLoc);

  /// Creates a empty parameter list
  static ParamList *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
                                SourceLoc rParenLoc) {
    return create(ctxt, lParenLoc, {}, rParenLoc);
  }

  size_t getNumParams() const { return numParams; }
  ArrayRef<ParamDecl *> getParams() const {
    return {getTrailingObjects<ParamDecl *>(), numParams};
  }
  ParamDecl *getParam(unsigned n) const { return getParams()[n]; }

  ParamDecl *operator[](unsigned n) const { return getParam(n); }

  iterator begin() const { return getParams().begin(); }
  iterator end() const { return getParams().end(); }

  SourceLoc getLParenLoc() const { return lParenLoc; }
  SourceLoc getRParenLoc() const { return rParenloc; }

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getEndLoc() const { return rParenloc; }
  SourceRange getSourceRange() const { return {lParenLoc, rParenloc}; }
};

/// "func" declarations - function declarations
class FuncDecl final : public ValueDecl, public DeclContext {
  SourceLoc funcLoc;
  ParamList *paramList = nullptr;
  BlockStmt *body = nullptr;
  Type type;
  TypeLoc returnTypeLoc;

public:
  FuncDecl(DeclContext *declContext, SourceLoc funcLoc, SourceLoc identLoc,
           Identifier ident, ParamList *params, TypeLoc returnTypeLoc)
      : ValueDecl(DeclKind::Func, declContext, identLoc, ident),
        DeclContext(DeclContextKind::FuncDecl, declContext), funcLoc(funcLoc),
        paramList(params), returnTypeLoc(returnTypeLoc) {
    bits.FuncDecl.isBodyChecked = false;
  }

  BlockStmt *getBody() const { return body; }
  void setBody(BlockStmt *body) { this->body = body; }

  /// \returns whether the body of this function has been type-checked or not
  bool isBodyChecked() const { return bits.FuncDecl.isBodyChecked; }
  /// Sets whether the body of this function has been checked.
  void setBodyChecked(bool value = true) {
    bits.FuncDecl.isBodyChecked = value;
  }

  ParamList *getParamList() const { return paramList; }
  void setParamList(ParamList *params) { this->paramList = params; }

  TypeLoc &getReturnTypeLoc() { return returnTypeLoc; }
  const TypeLoc &getReturnTypeLoc() const { return returnTypeLoc; }
  /// \returns true if the user wrote a return type for this function
  bool hasReturnType() const { return returnTypeLoc.hasTypeRepr(); }

  /// Sets the type of this value (the type of this function)
  void setValueType(Type type) { this->type = type; }
  /// \returns the type of this value (the type of the function)
  Type getValueType() const { return type; }

  /// \returns the SourceLoc of the "func" keyword
  SourceLoc getFuncLoc() const { return funcLoc; }

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const;

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Func;
  }

  static bool classof(const DeclContext *dc) {
    return dc->getDeclContextKind() == DeclContextKind::FuncDecl;
  }
};

/// Common base for declarations that contain a pattern and a potential
/// initializer expression (with an equal sign in between).
///
/// examples:
/// \verbatim
///   a = 0
//    a
///   a : i32 = 0
///   (a, b) = (0, 1)
///   a : i64
///   mut a
/// \endverbatim
class PatternBindingDecl : public Decl {
  Pattern *pattern = nullptr;
  SourceLoc equalLoc;
  Expr *init = nullptr;

protected:
  /// Creates A PBD.
  /// If the PBD has an initializer, \p init must not be nullptr.
  /// Please note that if init is nullptr, \p equalLoc will not be stored.
  PatternBindingDecl(DeclKind kind, DeclContext *declContext, Pattern *pattern,
                     SourceLoc equalLoc = SourceLoc(), Expr *init = nullptr)
      : Decl(kind, declContext), pattern(pattern), equalLoc(equalLoc),
        init(init) {}

public:
  /// Call \p fn on each VarDecl in this PatternBindingDecl.
  void forEachVarDecl(llvm::function_ref<void(VarDecl *)> fn) const;

  /// \returns the initializer of this PBD, if it has one.
  Expr *getInitializer() const { return init; }
  /// Replaces the initializer of this PBD.
  void setInitializer(Expr *expr) { init = expr; }
  /// \returns whether this PBD has an initializer.
  bool hasInitializer() const { return init; }

  /// \returns the Pattern of this PBD
  Pattern *getPattern() const { return pattern; }
  /// \returns the SourceLoc of the '=' if there's an initializer.
  SourceLoc getEqualLoc() const { return equalLoc; }

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::First_PatternBinding &&
           decl->getKind() <= DeclKind::Last_PatternBinding;
  }
};

/// Represents a "let" declaration.
/// A "let" declaration consists of the "let" keyword, a pattern and
/// an optional initializer.
class LetDecl final : public PatternBindingDecl {
  SourceLoc letLoc;

public:
  LetDecl(DeclContext *declContext, SourceLoc letLoc, Pattern *pattern,
          SourceLoc equalLoc = SourceLoc(), Expr *init = nullptr)
      : PatternBindingDecl(DeclKind::Let, declContext, pattern, equalLoc, init),
        letLoc(letLoc) {}

  SourceLoc getLetLoc() const { return letLoc; }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Let;
  }
};
} // namespace sora