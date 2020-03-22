//===--- TypeChecker.cpp ----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/EntryPoints.hpp"

using namespace sora;

//===- TypeChecker --------------------------------------------------------===//

TypeChecker::TypeChecker(ASTContext &ctxt)
    : ctxt(ctxt), diagEngine(ctxt.diagEngine) {}

//===- performSema --------------------------------------------------------===//

void sora::performSema(SourceFile &file) {
  TypeChecker tc(file.astContext);
  // type-check the declarations inside the file
  for (ValueDecl *decl : file.getMembers())
    tc.typecheckDecl(decl);
  // type-check the function bodies
  tc.typecheckDefinedFunctions();
}