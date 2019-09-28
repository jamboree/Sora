//===--- DiagnosticsParser.hpp - Parser Diagnostics -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Diagnostics/DiagnosticsCommon.hpp"

namespace sora {
namespace diag {
#define DIAG(KIND, ID, TEXT, SIGNATURE)                                        \
  extern detail::TypedDiagHelper<void SIGNATURE>::type ID;
#include "DiagnosticsParser.def"
} // namespace diag
} // namespace sora
