//===--- SourceFile.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/SourceFile.hpp"

using namespace sora;

SourceFile *DeclContext::getAsSourceFile() {
  return dyn_cast<SourceFile>(this);
}