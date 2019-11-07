//===--- Sema.hpp - Sora Language Semantic Analysis -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Diagnostics/DiagnosticEngine.hpp"

namespace sora {
class ASTContext;
class SourceFile;

/// Main interface of Sora's semantic analyzer.
class Sema final {
  Sema(const Sema &) = delete;
  Sema &operator=(const Sema &) = delete;

public:
  Sema(ASTContext &ctxt);

  /// Main entry point of semantic analysis: performs semantic analysis on \p
  /// file.
  void performSema(SourceFile &file);

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    assert(loc && "Sema can't emit diagnostics without valid SourceLocs");
    return diagEngine.diagnose<Args...>(loc, diag, args...);
  }

  ASTContext &ctxt;
  DiagnosticEngine &diagEngine;
};
} // namespace sora
