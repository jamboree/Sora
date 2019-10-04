//===--- OperatorKinds.cpp --------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/OperatorKinds.hpp"
#include "llvm/Support/Error.h"

using namespace sora;

BinaryOperatorKind sora::getOpForCompoundAssignementOp(BinaryOperatorKind op) {
  switch (op) {
  default:
    llvm_unreachable("Not a Compound Assignement Operator!");
#define COMPOUND_ASSIGN_OP(ID, SPELLING, OP)                                   \
  case BinaryOperatorKind::ID:                                                 \
    return BinaryOperatorKind::OP;
#include "Sora/AST/OperatorKinds.def"
  }
}

const char *sora::getSpelling(BinaryOperatorKind op) {
  switch (op) {
#define BINARY_OP(ID, SPELLING)                                                \
  case BinaryOperatorKind::ID:                                                 \
    return SPELLING;
#include "Sora/AST/OperatorKinds.def"
  }
  llvm_unreachable("Unknown BinaryOperatorKind");
}

const char *sora::to_string(BinaryOperatorKind op) {
  switch (op) {
#define BINARY_OP(ID, SPELLING)                                                \
  case BinaryOperatorKind::ID:                                                 \
    return #ID;
#include "Sora/AST/OperatorKinds.def"
  }
  llvm_unreachable("Unknown BinaryOperatorKind");
}

const char *sora::getSpelling(UnaryOperatorKind op) {
  switch (op) {
#define UNARY_OP(ID, SPELLING)                                                 \
  case UnaryOperatorKind::ID:                                                  \
    return SPELLING;
#include "Sora/AST/OperatorKinds.def"
  }
  llvm_unreachable("Unknown UnaryOperatorKind");
}

const char *sora::to_string(UnaryOperatorKind op) {
  switch (op) {
#define UNARY_OP(ID, SPELLING)                                                 \
  case UnaryOperatorKind::ID:                                                  \
    return #ID;
#include "Sora/AST/OperatorKinds.def"
  }
  llvm_unreachable("Unknown UnaryOperatorKind");
}