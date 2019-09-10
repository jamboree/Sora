//===--- ASTNode.cpp --------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTNode.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Stmt.hpp"
#include "llvm/Support/Error.h"

using namespace sora;

SourceRange ASTNode::getSourceRange() const {
  if (is<Expr *>())
    return get<Expr *>()->getSourceRange();
  if (is<Decl *>())
    return get<Decl *>()->getSourceRange();
  if (is<Stmt *>())
    return get<Stmt *>()->getSourceRange();
  llvm_unreachable("unknown node");
}

SourceLoc ASTNode::getBegLoc() const {
  if (is<Expr *>())
    return get<Expr *>()->getBegLoc();
  if (is<Decl *>())
    return get<Decl *>()->getBegLoc();
  if (is<Stmt *>())
    return get<Stmt *>()->getBegLoc();
  llvm_unreachable("unknown node");
}

SourceLoc ASTNode::getEndLoc() const {
  if (is<Expr *>())
    return get<Expr *>()->getEndLoc();
  if (is<Decl *>())
    return get<Decl *>()->getEndLoc();
  if (is<Stmt *>())
    return get<Stmt *>()->getEndLoc();
  llvm_unreachable("unknown node");
}