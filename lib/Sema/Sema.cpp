//===--- Sema.cpp -----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Sema/Sema.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"

using namespace sora;

Sema::Sema(ASTContext &ctxt) : ctxt(ctxt), diagEngine(ctxt.diagEngine) {}

//===----------------------------------------------------------------------===//
// This is currently just for testing purposes and will be refactored later.
// We're just going to perform name binding on every UnresolvedDeclRef in
// the source file.
//===----------------------------------------------------------------------===//

void Sema::performSema(SourceFile &file) {
  for (ValueDecl *decl : file.getMembers())
    typecheckDecl(decl);
  typecheckFunctionBodies();
}