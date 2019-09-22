//===--- SourceFile.hpp - Source File AST -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTAlignement.hpp"
#include "Sora/AST/DeclContext.hpp"
#include "Sora/AST/Identifier.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class ASTWalker;
class FuncDecl;

/// Represents a source file.
class alignas(SourceFileAlignement) SourceFile final : public DeclContext {
  Identifier identifier;
  SmallVector<FuncDecl *, 4> functions;
  BufferID bufferID;

public:
  /// \param astContext the ASTContext in which the members of this source file
  /// are allocated.
  /// \param identifier the identifier (name) of this source file
  SourceFile(BufferID bufferID, ASTContext &astContext, DeclContext *parent,
             Identifier identifier)
      : DeclContext(DeclContextKind::SourceFile, parent),
        identifier(identifier), bufferID(bufferID), astContext(astContext) {}

  /// \returns the identifier (name) of this source file
  Identifier getIdentifier() const { return identifier; }
  /// \returns the functions (members) of this source file
  ArrayRef<FuncDecl *> getFunctions() const { return functions; }
  /// Adds a function to this source file
  void addFunction(FuncDecl *fn) { return functions.push_back(fn); }
  /// \returns the buffer id of this SourceFile
  BufferID getBufferID() const { return bufferID; }

  /// Traverse this SourceFile using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);

  static bool classof(const DeclContext *dc) {
    return dc->getDeclContextKind() == DeclContextKind::SourceFile;
  }

  /// The ASTContext in which the members of this source file
  /// are allocated.
  ASTContext &astContext;
};
} // namespace sora