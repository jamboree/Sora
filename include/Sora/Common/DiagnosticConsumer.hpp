//===--- DiagnosticConsumer.hpp - Diagnostic Consumers ----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Contains the DiagnosticConsumer abstract class and a few implementations
// of this class.
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <stdint.h>
#include <string>

namespace sora {
class SourceManager;

/// Kinds (severities) of diagnostics.
enum class DiagnosticKind : uint8_t { Remark, Note, Warning, Error };

/// A Diagnostic "Fix-It", a remplacement of a piece of text with another.
class FixIt {
  std::string text;
  CharSourceRange charRange;

public:
  FixIt(const std::string &text, CharSourceRange charRange)
      : text(text), charRange(charRange) {}

  /// \returns the character range of the piece of text targeted by this fix-it
  CharSourceRange getCharRange() const { return charRange; }
  /// \returns the text that should replace the old range of text
  StringRef getText() const { return text; }
};

/// A finished, fully-formed diagnostic.
struct Diagnostic {
  /// Creates a Diagnostic object
  /// \param message The Diagnostic's Message
  /// \param kind The Kind of Diagnostic this is
  /// \param loc The Location of the Diagnostic
  /// \param fixits Fix-its for this Diagnostic
  /// \param ranges Additional highlighted ranges
  Diagnostic(StringRef message, DiagnosticKind kind, SourceLoc loc,
             ArrayRef<CharSourceRange> ranges = {}, ArrayRef<FixIt> fixits = {})
      : message(message), kind(kind), loc(loc), ranges(ranges), fixits(fixits) {
  }

  /// The Diagnostic's Message
  StringRef message;
  /// The Kind of Diagnostic this is
  DiagnosticKind kind;
  /// The Location of the Diagnostic
  SourceLoc loc;
  /// Additional highlighted ranges
  ArrayRef<CharSourceRange> ranges;
  /// Fix-its for this Diagnostic
  ArrayRef<FixIt> fixits;
};

/// Base class for all diagnostic consumers
class DiagnosticConsumer {
public:
  virtual ~DiagnosticConsumer() = default;
  /// Handles a diagnostic.
  virtual void handle(SourceManager &srcMgr, const Diagnostic &diagnostic) = 0;
};

/// A DiagnosticConsumer that pretty-prints diagnostics to a raw_ostream.
class PrintingDiagnosticConsumer : public DiagnosticConsumer {
  /// Handles a "simple" diagnostic that does not have source location.
  void handleSimpleDiagnostic(const Diagnostic &diagnostic, bool showColors);

public:
  PrintingDiagnosticConsumer(raw_ostream &out) : out(out) {}

  /// the output stream
  raw_ostream &out;

  void handle(SourceManager &srcMgr, const Diagnostic &diagnostic) override;
};

/// A DiagnosticConsumer that forwards diagnostic handling to a function
class ForwardingDiagnosticConsumer : public DiagnosticConsumer {
public:
  using HandlerFunction =
      std::function<void(SourceManager &, const Diagnostic &)>;

  ForwardingDiagnosticConsumer(HandlerFunction func) : func(func) {}

  /// Handles a diagnostic.
  void handle(SourceManager &srcMgr, const Diagnostic &diagnostic) override {
    func(srcMgr, diagnostic);
  }

private:
  HandlerFunction func;
};
} // namespace sora