//===--- IRGenPattern.cpp - Pattern IR Generation ---------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "IRGen.hpp"

#include "Sora/AST/Pattern.hpp"

using namespace sora;

//===- IRGen --------------------------------------------------------------===//

mlir::Location IRGen::getNodeLoc(Pattern *pattern) {
  return mlir::OpaqueLoc::get(pattern, getFileLineColLoc(pattern->getLoc()));
}
