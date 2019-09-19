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

/// Represents a source file.
class alignas(SourceFileAlignement) SourceFile final {
  Identifier identifier;
  SmallVector<Decl *, 4> members;

public:
  /// \param astContext the ASTContext in which the members of this source file
  /// are allocated.
  /// \param identifier the identifier (name) of this source file
  SourceFile(ASTContext &astContext, Identifier identifier)
      : identifier(identifier), astContext(astContext) {}

  /// \returns the identifier (name) of this source file
  Identifier getIdentifier() const { return identifier; }
  /// \returns the members of this source file
  ArrayRef<Decl *> getMembers() const { return members; }
  /// Adds a member to this source file
  void addMember(Decl *decl) { return members.push_back(decl); }
  /// The ASTContext in which the members of this source file
  /// are allocated.
  ASTContext &astContext;
};
} // namespace sora