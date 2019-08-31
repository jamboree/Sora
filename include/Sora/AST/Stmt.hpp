//===--- Stmt.hpp - Statement ASTs -----------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

namespace sora {
enum class StmtKind : uint8_t {
#define STMT(KIND, PARENT) KIND,
#include "Sora/AST/StmtNodes.def"
};
}