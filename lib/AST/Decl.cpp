//===--- Decl.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Decl.hpp"
#include "ASTNodeLoc.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"

using namespace sora;

/// Check that all declarations are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define DECL(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Decl>::value,               \
                #ID "Decl is not trivially destructible.");
#include "Sora/AST/DeclNodes.def"

void *Decl::operator new(size_t size, ASTContext &ctxt, unsigned align) {
  return ctxt.allocate(size, align, ASTAllocatorKind::Permanent);
}

SourceFile &Decl::getSourceFile() const {
  const Decl *cur = this;
  // Keep going while we have a valid parent.
  while (DeclParent parent = cur->getParent()) {
    // If we have a SourceFile as parent, we can stop.
    if (SourceFile *sf = parent.dyn_cast<SourceFile *>())
      return *sf;
    // If we don't have a SourceFile as parent, it has to be a Decl.
    cur = parent.get<Decl *>();
  }
  llvm_unreachable("ill-formed declaration parent chain");
}

ASTContext &Decl::getASTContext() const { return getSourceFile().astContext; }

DiagnosticEngine &Decl::getDiagnosticEngine() const {
  return getASTContext().diagEngine;
}

bool Decl::isLocal() const {
  if (Decl *decl = getParent().dyn_cast<Decl *>())
    return isa<FuncDecl>(decl);
  return false;
}

SourceLoc Decl::getBegLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getBegLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceLoc Decl::getEndLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getEndLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceLoc Decl::getLoc() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getLoc(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

SourceRange Decl::getSourceRange() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return ASTNodeLoc<Decl, ID##Decl>::getSourceRange(cast<ID##Decl>(this));
#include "Sora/AST/DeclNodes.def"
  }
}

Type ValueDecl::getValueType() const {
  switch (getKind()) {
  default:
    llvm_unreachable("unknown ValueDecl kind");
#define VALUE_DECL(ID, PARENT)                                                 \
  case DeclKind::ID:                                                           \
    return cast<ID##Decl>(this)->getValueType();
#include "Sora/AST/DeclNodes.def"
  }
}

ParamList::ParamList(SourceLoc lParenLoc, ArrayRef<ParamDecl *> params,
                     SourceLoc rParenloc)
    : lParenLoc(lParenLoc), rParenloc(rParenloc), numParams(params.size()) {
  std::uninitialized_copy(params.begin(), params.end(),
                          getTrailingObjects<ParamDecl *>());
}

ParamList *ParamList::create(ASTContext &ctxt, SourceLoc lParenLoc,
                             ArrayRef<ParamDecl *> params,
                             SourceLoc rParenLoc) {
  // Need manual memory allocation here because of trailing objects.
  auto size = totalSizeToAlloc<ParamDecl *>(params.size());
  void *mem = ctxt.allocate(size, alignof(ParamList));
  return new (mem) ParamList(lParenLoc, params, rParenLoc);
}

SourceLoc ParamDecl::getBegLoc() const { return getIdentifierLoc(); }

SourceLoc ParamDecl::getEndLoc() const { return tyLoc.getEndLoc(); }

SourceLoc FuncDecl::getBegLoc() const { return funcLoc; }

SourceLoc FuncDecl::getEndLoc() const { return body->getEndLoc(); }

SourceLoc LetDecl::getBegLoc() const { return letLoc; }

SourceLoc LetDecl::getEndLoc() const {
  if (hasInitializer())
    return getInitializer()->getEndLoc();
  return getPattern()->getEndLoc();
}