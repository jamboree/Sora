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
class DiagnosticEngine;
class PatternBindingDecl;
class BlockStmt;

/// Kinds of Declarations
enum class DeclKind : uint8_t {
#define DECL(KIND, PARENT) KIND,
#define DECL_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/DeclNodes.def"
};

/// Base class for every Declaration node.
class alignas(DeclAlignement) Decl {
  // Disable vanilla new/delete for declarations
  void *operator new(size_t) noexcept = delete;
  void operator delete(void *)noexcept = delete;

  DeclContext *declContext;
  DeclKind kind;
  /// Make use of the padding bits by allowing derived class to store data here.
  /// NOTE: Derived classes are expected to initialize the bitfields.
  LLVM_PACKED(union Bits {
    Bits() : raw() {}
    // Raw bits (to zero-init the union)
    char raw[7];
    // VarDecl bits
    struct {
      bool isMutable;
    } varDecl;
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

  Decl(DeclKind kind, DeclContext *declContext)
      : kind(kind), declContext(declContext) {}

public:
  // Publicly allow allocation of declaration using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Decl));

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
  DeclContext *getDeclContext() const { return declContext; }

  /// If this Decl is also a DeclContext, returns it as a DeclContext*, else
  /// returns nullptr.
  DeclContext *getAsDeclContext() { return dyn_cast<DeclContext>(this); }

  /// If this Decl is also a DeclContext, returns it as a DeclContext*, else
  /// returns nullptr.
  const DeclContext *getAsDeclContext() const {
    return dyn_cast<DeclContext>(this);
  }

  /// Dumps this declaration to \p out
  /// TODO: Remove srcMgr once we get access to ASTContext from all decls
  /// through DeclContexts.
  void dump(raw_ostream &out, const SourceManager &srcMgr, unsigned indent = 2);

  /// \return the kind of declaration this is
  DeclKind getKind() const { return kind; }

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

/// Base class for declarations that declare a value of some type with a
/// given name.
///
/// As an implementation detail, the type is not stored in the ValueDecl,
/// but inside the derived classes. This works in the same way as
/// "getBegLoc" for instance. This is done so we don't waste space storing a
/// copy of the type in this class. e.g. VarDecl needs to store a TypeLoc,
/// and we don't want to store the type in the TypeLoc AND in the ValueDecl,
/// it'd waste space for nothing.
class ValueDecl : public Decl {
  SourceLoc identifierLoc;
  Identifier identifier;

protected:
  ValueDecl(DeclKind kind, DeclContext *declContext, SourceLoc identifierLoc,
            Identifier identifier)
      : Decl(kind, declContext), identifierLoc(identifierLoc),
        identifier(identifier) {}

public:
  /// \returns the identifier (name) of this decl
  Identifier getIdentifier() const { return identifier; }

  /// \returns the SourceLoc of the identifier
  SourceLoc getIdentifierLoc() const { return identifierLoc; }

  /// \returns the type this value has
  Type getValueType() const;

  /// \returns the preffered SourceLoc for diagnostics.
  SourceLoc getLoc() const { return identifierLoc; }

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::First_Value &&
           decl->getKind() <= DeclKind::Last_Value;
  }
};

/// Represents a *single* variable declaration.
///
/// This DOES NOT represent something such as "let x" entirely, it only
/// represents the "x". "let x" as a whole would be represented by
/// LetDecl. This also means that getBegLoc/getEndLoc/getRange will only
/// return the loc of the identifier, ignoring the TypeLoc entirely.
///
/// Please note that the TypeLoc won't have a TypeRepr* if the type wasn't
/// explicitely written down. However, the TypeLoc should always have a
/// valid Type after semantic analysis. \verbatim
///   let x : i32 // has valid TypeRepr*
///   let (y, x) : (i32, i32) // has valid TypeRepr* (of the
/// \endverbatim
class VarDecl final : public ValueDecl {
  TypeLoc tyLoc;

public:
  VarDecl(DeclContext *declContext, SourceLoc identifierLoc,
          Identifier identifier, bool isMutable = false)
      : ValueDecl(DeclKind::Var, declContext, identifierLoc, identifier) {
    bits.varDecl.isMutable = isMutable;
  }

  /// \returns the TypeLoc of the VarDecl.
  /// This may or may not have a valid TypeRepr (it won't have one
  /// if the type was inferred).
  /// However, the type should be valid after semantic analysis.
  TypeLoc &getTypeLoc() { return tyLoc; }
  /// \returns the TypeLoc of the VarDecl.
  /// This may or may not have a valid TypeRepr (it won't have one
  /// if the type was inferred).
  /// However, the type should be valid after semantic analysis.
  TypeLoc getTypeLoc() const { return tyLoc; }

  /// \returns the type this value has (the type of the variable)
  Type getValueType() const { return tyLoc.getType(); }

  /// \returns true if the variable is mutable, false otherwise
  bool isMutable() const { return bits.varDecl.isMutable; }

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const { return getIdentifierLoc(); }
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const { return getIdentifierLoc(); }

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Var;
  }
};

/// Function parameter declarations
class ParamDecl final : public ValueDecl {
  TypeLoc tyLoc;
  SourceLoc colonLoc;

public:
  ParamDecl(DeclContext *declContext, SourceLoc identifierLoc,
            Identifier identifier, SourceLoc colonLoc, TypeLoc typeLoc)
      : ValueDecl(DeclKind::Param, declContext, identifierLoc, identifier),
        tyLoc(typeLoc), colonLoc(colonLoc) {}

  /// \returns the TypeLoc of the ParamDecl's type.
  /// This should always have a valid TypeRepr*.
  /// The type should be valid after semantic analysis.
  TypeLoc &getTypeLoc() { return tyLoc; }
  /// \returns the TypeLoc of the ParamDecl's type.
  /// This should always have a valid TypeRepr*.
  /// The type should be valid after semantic analysis.
  TypeLoc getTypeLoc() const { return tyLoc; }

  /// \returns the type this value has (the type of the parameter)
  Type getValueType() const { return tyLoc.getType(); }

  SourceLoc getColonLoc() const { return colonLoc; }

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
  unsigned numParams = 0;

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

  unsigned getNumParams() const { return numParams; }
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

public:
  FuncDecl(DeclContext *declContext, SourceLoc funcLoc, SourceLoc identLoc,
           Identifier ident)
      : ValueDecl(DeclKind::Func, declContext, identLoc, ident),
        DeclContext(DeclContextKind::FuncDecl, declContext), funcLoc(funcLoc) {}

  BlockStmt *getBody() const { return body; }
  void setBody(BlockStmt *body) { this->body = body; }

  ParamList *getParamList() const { return paramList; }
  void setParamList(ParamList *params) { this->paramList = params; }

  /// \returns the type this value has (the type of the function)
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

template <typename Derived>
using PatternBindingDeclTrailingObjects =
    llvm::TrailingObjects<Derived, SourceLoc, Expr *>;

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
///
/// As an implementation detail, the SourceLoc of the "=" and the Expr*
/// pointer are trail-allocated in order to optimize storage for
/// declarations that don't have an initializer. This means that all
/// derived classes must privately inherit from
/// PatternBindingDeclTrailingObjects and use static factory methods. The
/// derived class don't need to implement numTrailingObjects or interact
/// with their trailing objects, but they do need to friend both
/// PatternBindingDeclTrailingObjects and PatternBindingDecl.
class PatternBindingDecl
    : public Decl,
      private llvm::trailing_objects_internal::TrailingObjectsBase {
  /// The Pattern* + a flag indicating if we have an initializer or not.
  llvm::PointerIntPair<Pattern *, 1, bool> patternAndHasInit;

  template <typename Type> Type *getDerivedTrailingObjects();

  template <typename Type> const Type *getDerivedTrailingObjects() const {
    return const_cast<PatternBindingDecl *>(this)
        ->getDerivedTrailingObjects<Type>();
  }

protected:
  size_t numTrailingObjects(OverloadToken<SourceLoc>) const {
    return hasInitializer() ? 1 : 0;
  }

  /// Creates A PBD.
  /// If the PBD has an initializer, \p init must not be nullptr.
  /// Please note that if init is nullptr, \p equalLoc will not be stored.
  PatternBindingDecl(DeclKind kind, DeclContext *declContext, Pattern *pattern,
                     SourceLoc equalLoc = SourceLoc(), Expr *init = nullptr)
      : Decl(kind, declContext), patternAndHasInit(pattern, init != nullptr) {
    if (hasInitializer()) {
      *getDerivedTrailingObjects<SourceLoc>() = equalLoc;
      *getDerivedTrailingObjects<Expr *>() = init;
    }
  }

public:
  /// \returns true if we have an initializer
  bool hasInitializer() const { return patternAndHasInit.getInt(); }

  /// \returns the Pattern
  Pattern *getPattern() const { return patternAndHasInit.getPointer(); }

  /// \returns the SourceLoc of the equal sign if we have an initializer,
  /// SourceLoc() if we don't.
  SourceLoc getEqualLoc() const {
    return hasInitializer() ? *getDerivedTrailingObjects<SourceLoc>()
                            : SourceLoc();
  }

  /// \returns the initializer if present, nullptr otherwise.
  Expr *getInitializer() const {
    return hasInitializer() ? *getDerivedTrailingObjects<Expr *>() : nullptr;
  }

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::First_PatternBinding &&
           decl->getKind() <= DeclKind::Last_PatternBinding;
  }
}; // namespace sora

/// Represents a "let" declaration.
/// A "let" declaration consists of the "let" keyword, a pattern and
/// an optional initializer.
class LetDecl final : public PatternBindingDecl,
                      private PatternBindingDeclTrailingObjects<LetDecl> {
  friend PatternBindingDeclTrailingObjects<LetDecl>;
  friend PatternBindingDecl;

  SourceLoc letLoc;

  LetDecl(DeclContext *declContext, SourceLoc letLoc, Pattern *pattern,
          SourceLoc equalLoc, Expr *init)
      : PatternBindingDecl(DeclKind::Let, declContext, pattern, equalLoc, init),
        letLoc(letLoc) {}

public:
  /// Creates a "let" declaration.
  /// If it has an initializer, \p init must not be nullptr.
  /// If it doesn't have one, both equalLoc and init can be left empty.
  /// Please note that if init is nullptr, \p equalLoc will not be stored.
  static LetDecl *create(ASTContext &ctxt, DeclContext *declContext,
                         SourceLoc letLoc, Pattern *pattern,
                         SourceLoc equalLoc = SourceLoc(),
                         Expr *init = nullptr);

  SourceLoc getLetLoc() const { return letLoc; }

  SourceLoc getBegLoc() const;
  SourceLoc getEndLoc() const;

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Let;
  }
};

template <typename Type> Type *PatternBindingDecl::getDerivedTrailingObjects() {
  if (auto let = dyn_cast<LetDecl>(this))
    return let->getTrailingObjects<Type>();
  llvm_unreachable("Unhandled PatternBindingDecl?");
}
} // namespace sora