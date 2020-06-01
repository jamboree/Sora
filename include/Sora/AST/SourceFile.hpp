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
class ASTContext;
class ValueDecl;
class SourceFileScope;

/// Represents a source file.
class alignas(SourceFileAlignement) SourceFile final : public DeclContext {
  SmallVector<ValueDecl *, 4> members;
  SourceFileScope *fileScope = nullptr;
  BufferID bufferID;

  SourceFile(ASTContext &astContext, BufferID bufferID, DeclContext *parent)
      : DeclContext(DeclContextKind::SourceFile, parent), bufferID(bufferID),
        astContext(astContext) {}

  SourceFile(const SourceFile &) = delete;
  SourceFile &operator=(const SourceFile &) = delete;

public:
  /// \param ctxt the ASTContext in which the members of this source file
  /// are allocated (and also the one that'll be used to allocate this
  /// SourceFile's memory)
  /// \param bufferID the BufferID of this SourceFile in the SourceManager
  /// \param parent the parent DeclContext of this SourceFile. Can be nullptr.
  static SourceFile *create(ASTContext &ctxt, BufferID bufferID,
                            DeclContext *parent);

  /// \returns the number of members in this source file
  size_t getNumMembers() const { return members.size(); }
  /// \returns the members of this source file
  ArrayRef<ValueDecl *> getMembers() const { return members; }
  /// Adds a member to this source file
  void addMember(ValueDecl *decl) { return members.push_back(decl); }
  /// \returns the BufferID of this SourceFile
  BufferID getBufferID() const { return bufferID; }
  /// \returns the buffer identifier of this Sourcefile
  StringRef getBufferName() const;
  /// \returns true if this SourceFile is empty
  bool empty() const { return members.empty(); }

  /// \returns true if \p loc belongs to this file
  bool contains(SourceLoc loc) const;

  /// Dumps this source file to \p out
  void dump(raw_ostream &out, unsigned indent = 2) const;
  /// Dumps this source file to llvm::dbgs(), using default options.
  void dump() const;

  /// \returns whether this SourceFile has a ASTScope describing it.
  bool hasScopeMap() const { return fileScope != nullptr; }
  /// \param canLazilyBuild if true, the SourceFileScope will be constructed
  /// (but not expanded) if needed. Default is true.
  /// \returns this file's SourceFileScope (scope map)
  SourceFileScope *getScopeMap(bool canLazilyBuild = true);

  /// Traverse this SourceFile using \p walker.
  /// \returns true if the walk completed successfully, false if it ended
  /// prematurely.
  bool walk(ASTWalker &walker);
  bool walk(ASTWalker &&walker) { return walk(walker); }

  /// \returns the SourceLoc of the first token in the SourceFile.
  SourceLoc getBegLoc() const;
  /// \returns the SourceLoc of the last token in the SourceFile
  SourceLoc getEndLoc() const;
  /// \returns the SourceRange of the SourceFile.
  SourceRange getSourceRange() const;

  static bool classof(const DeclContext *dc) {
    return dc->getDeclContextKind() == DeclContextKind::SourceFile;
  }

  /// The ASTContext in which the members of this source file
  /// are allocated.
  ASTContext &astContext;
};
} // namespace sora