//===--- IRGenType.cpp - Type IR Generation ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/Type.hpp"

using namespace sora;

//===- IRGen --------------------------------------------------------------===//

mlir::Type IRGen::getMLIRType(Type type) {
  llvm_unreachable("getMLIRType - unimplemented");
}