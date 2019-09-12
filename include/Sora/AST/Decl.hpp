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
#include "Sora/Common/SourceLoc.hpp"
#include <cassert>
#include <stdint.h>

namespace sora {
class ASTContext;
class PatternBindingDecl;

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
class VarDecl final : public ValueDecl {
  TypeLoc tyLoc;

public:
  TypeLoc &getTypeLoc() { return tyLoc; }
  TypeLoc getTypeLoc() const { return tyLoc; }

  /// \returns the type this value has (the type of the variable)
  Type getValueType() const { return tyLoc.getType(); }

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const { return getIdentifierLoc(); }
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const { return getIdentifierLoc(); }

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::Var;
  }
};

/// Function parameter declarations
class ParamDecl final : public ValueDecl {
  TypeLoc tyLoc;
  SourceLoc colonLoc;

public:
  TypeLoc &getTypeLoc() { return tyLoc; }
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
    return decl->getKind() >= DeclKind::Param;
  }
};

/// "func" declarations - function declarations
class FuncDecl final : public ValueDecl {
  Type type;

public:
  /// \returns the type this value has (the type of the function)
  Type getValueType() const { return type; }

  /// \returns the SourceLoc of the first token of the declaration
  SourceLoc getBegLoc() const {
    /* TODO */
    return SourceLoc();
  }
  /// \returns the SourceLoc of the last token of the declaration
  SourceLoc getEndLoc() const {
    /* TODO */
    return SourceLoc();
  }

  static bool classof(const Decl *decl) {
    return decl->getKind() >= DeclKind::Func;
  }
};
} // namespace sora