//===--- TypeChecker.hpp - Sora Language Semantic Analysis -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

namespace sora {
class SourceFile;

/// Performs Semantic Analysis on \p sf
///
/// Memory will be allocated using the SourceFile's ASTContext, and Diagnostics
/// will be emitted using the ASTContext's DiagnosticEngine.
void performSema(SourceFile &sf);
} // namespace sora