//===--- Decl.hpp - Declarations ASTs ---------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
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

  DeclKind kind;

protected:
  // Children should be able to use placement new, as it is needed for children
  // with trailing objects.
  void *operator new(size_t, void *mem) noexcept {
    assert(mem);
    return mem;
  }

  Decl(DeclKind kind) : kind(kind) {}

public:
  // Publicly allow allocation of declaration using the ASTContext.
  void *operator new(size_t size, ASTContext &ctxt,
                     unsigned align = alignof(Decl));

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const;
  /// \returns the preffered SourceLoc for diagnostics. This is defaults to
  /// getBegLoc but nodes can override it as they please.
  SourceLoc getLoc() const;
  /// \returns the full range of this declaration
  SourceRange getSourceRange() const;

  /// \return the kind of declaration this is
  DeclKind getKind() const { return kind; }
};

/// Base class for declarations that declare a value of some type with a given
/// name.
///
/// As an implementation detail, the type is not stored in the ValueDecl, but
/// inside the derived classes. This works in the same way as "getBegLoc" for
/// instance. This is done so we don't waste space storing a copy of the type in
/// this class. e.g. VarDecl needs to store a TypeLoc, and we don't want to
/// store the type in the TypeLoc AND in the ValueDecl, it'd waste space for
/// nothing.
class ValueDecl : public Decl {
  SourceLoc identifierLoc;
  Identifier identifier;

protected:
  ValueDecl(DeclKind kind, SourceLoc identifierLoc, Identifier identifier)
      : Decl(kind), identifierLoc(identifierLoc), identifier(identifier) {}

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
/// represents the "x". "let x" as a whole would be represented by LetDecl.
/// This also means that getBegLoc/getEndLoc/getRange will only return
/// the loc of the identifier, ignoring the TypeLoc entirely.
///
/// Please note that the TypeLoc won't have a TypeRepr* if the type wasn't
/// explicitely written down. However, the TypeLoc should always have a valid
/// Type after semantic analysis.
/// \verbatim
///   let x : i32 // has valid TypeRepr*
///   let (y, x) : (i32, i32) // has valid TypeRepr* (of the
/// \endverbatim
class VarDecl final : public ValueDecl {
  TypeLoc tyLoc;

public:
  VarDecl(SourceLoc identifierLoc, Identifier identifier)
      : ValueDecl(DeclKind::Var, identifierLoc, identifier) {}

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
  ParamDecl(SourceLoc identifierLoc, Identifier identifier, SourceLoc colonLoc,
            TypeLoc typeLoc)
      : ValueDecl(DeclKind::Param, identifierLoc, identifier), tyLoc(typeLoc),
        colonLoc(colonLoc) {}

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

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const { return getIdentifierLoc(); }
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const {
    /// When TypeReprs are up and running, use tyLoc.getEndLoc()
    return SourceLoc();
  }

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Param;
  }
};

/// Represents a list of Parameter Declarations.
/// This list cannot be changed once created (you can't add/remove parameter or
/// change the pointer's value, but you can change the ParamDecl themselves)
class ParamList final : private llvm::TrailingObjects<ParamList, ParamDecl *> {
  friend llvm::TrailingObjects<ParamList, ParamDecl *>;

  void *operator new(size_t size) throw() = delete;
  void operator delete(void *mem) throw() = delete;
  void *operator new(size_t size, void *mem) throw() {
    assert(mem);
    return mem;
  }

  size_t numTrailingObjects(OverloadToken<ParamDecl *>) { return numParams; }

  SourceLoc lParenLoc, rParenloc;
  unsigned numParams = 0;

  ParamList(SourceLoc lParenLoc, ArrayRef<ParamDecl *> params,
            SourceLoc rParenloc);

public:
  using iterator = ArrayRef<ParamDecl *>::iterator;

  /// Creates a parameter list
  ParamList *create(ASTContext &ctxt, SourceLoc lParenLoc,
                    ArrayRef<ParamDecl *> params, SourceLoc rParenLoc);

  /// Creates a empty parameter list
  ParamList *createEmpty(ASTContext &ctxt, SourceLoc lParenLoc,
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

  SourceLoc getBegLoc() const { return lParenLoc; }
  SourceLoc getEndLoc() const { return rParenloc; }
  SourceRange getSourceRange() const { return {lParenLoc, rParenloc}; }
};

/// "func" declarations - function declarations
class FuncDecl final : public ValueDecl {
  SourceLoc funcLoc;
  ParamList *params = nullptr;
  BlockStmt *body = nullptr;
  Type type;

public:
  FuncDecl(SourceLoc funcLoc, SourceLoc identLoc, Identifier ident)
      : ValueDecl(DeclKind::Func, identLoc, ident), funcLoc(funcLoc) {}

  BlockStmt *getBody() const { return body; }
  void setBody(BlockStmt *body) { this->body = body; }

  ParamList *getParams() const { return params; }
  void setParams(ParamList *params) { this->params = params; }

  /// \returns the type this value has (the type of the function)
  Type getValueType() const { return type; }

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const;

  static bool classof(const Decl *decl) {
    return decl->getKind() == DeclKind::Func;
  }
};
} // namespace sora