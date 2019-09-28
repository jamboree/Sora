//===--- DiagnosticsCommon.hpp - Common Diagnostics -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Diagnostics/DiagnosticEngine.hpp"

namespace sora {
namespace detail {
template <typename Ty> struct TypedDiagHelper;
template <typename... Args> struct TypedDiagHelper<void(Args...)> {
  using type = TypedDiag<Args...>;
};
} // namespace detail
namespace diag {
#define DIAG(KIND, ID, TEXT, SIGNATURE)                                        \
  extern detail::TypedDiagHelper<void SIGNATURE>::type ID;
#include "DiagnosticsCommon.def"
} // namespace diag
} // namespace sora
