//===--- TypeRepr.hpp - Type Representation ASTs ----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// TypeReprs are, literally, Type Representations. They represent types as
// they were written by the user. TypeReprs aren't types, but they are converted
// to types during semantic analysis.
//
// This is used to have a representation of types that are source-accurate and
// include source location information.
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

namespace sora {
enum class TypeReprKind : uint8_t {
#define TYPEREPR(KIND, PARENT) KIND,
#include "Sora/AST/TypeReprNodes.def"
};
}