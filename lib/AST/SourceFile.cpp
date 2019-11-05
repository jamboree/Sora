//===--- SourceFile.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"

using namespace sora;

SourceFile *SourceFile::create(ASTContext &ctxt, BufferID bufferID,
                               DeclContext *parent) {
  // Allocate the memory for this SF in the ASTContext & create it.
  void *mem = ctxt.allocate<SourceFile>();
  auto *sf = new (mem) SourceFile(ctxt, bufferID, parent);
  // Register a cleanup for the SF since it contains non trivially-destructible
  // objects.
  ctxt.addDestructorCleanup(*sf);
  return sf;
}

SourceFile *DeclContext::getAsSourceFile() {
  return dyn_cast<SourceFile>(this);
}

bool SourceFile::walk(ASTWalker &walker) {
  for (Decl *decl : members) {
    if (!decl->walk(walker))
      return false;
  }
  return true;
}

SourceLoc SourceFile::getBegLoc() const {
  if (empty())
    return {};
  return members.front()->getBegLoc();
}

SourceLoc SourceFile::getEndLoc() const {
  if (empty())
    return {};
  return members.back()->getEndLoc();
}

SourceRange SourceFile::getSourceRange() const {
  return {getBegLoc(), getEndLoc()};
}