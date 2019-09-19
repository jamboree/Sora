//===--- SourceFile.hpp - Source File AST -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class Decl;

/// Represents a single source file, and keeps track of its name and members.
class alignas(SourceFileAlignement) SourceFile final {
  Identifier identifier;
  SmallVector<Decl *, 4> members;
  ASTContext &ctxt;

public:
  /// \param ctxt the ASTContext in which members of this source file are
  ///             allocated.
  /// \param identifier the identifier (name) of this source file
  SourceFile(ASTContext &ctxt, Identifier identifier)
      : identifier(identifier), ctxt(ctxt) {}

  /// \returns the identifier (name) of this source file
  Identifier getIdentifier() const { return identifier; }
  /// \returns the members of this source file
  ArrayRef<Decl *> getMembers() const { return members; }
  /// Adds a member to this source file
  void addMember(Decl *decl) { return members.push_back(decl); }
  /// \returns the ASTContext in which members of this SourceFile are allocated.
  ASTContext &getASTContext() const { return ctxt; }
};
} // namespace sora