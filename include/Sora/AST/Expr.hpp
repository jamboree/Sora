//===--- Expr.hpp - Expression ASTs -----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

namespace sora {
enum class ExprKind : uint8_t {
#define EXPR(KIND, PARENT) KIND,
#define EXPR_RANGE(KIND, FIRST, LAST) First_##KIND = FIRST, Last_##KIND = LAST,
#include "Sora/AST/ExprNodes.def"
};
}