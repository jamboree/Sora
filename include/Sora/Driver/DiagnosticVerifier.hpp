//===--- DiagnosticVerifier.hpp - Diagnostic Verification -------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the interface of the "Diagnostic Verifier", which is a
// special DiagnosticConsumer that implements the -verify feature of the
// compiler.
//
// The -verify feature is an important testing feature that allows tests
// to check that a given diagnostic was correctly emitted.
//
// The grammar of a "verify" line is roughly
//  verify-line = "expect-" (number "-")? kind offset? number? "{{" text "}}"
//  kind = "remark" | "note" | "warning" | "error"
//  offset = '@' ('+' | '-') number
//
// Example usages:
//    expect-error: foo // expects an error with message "foo" at that line
//    expect-3-error: foo // expects 3 instances of an error with message "foo"
//                        // at that line
//    expect-warning@+1: foo // expects a warning with message "foo" at the next
//                           // line
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/DiagnosticConsumer.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/StringMap.h"
#include <list>
#include <memory>

namespace sora {
struct Diagnostic;

/// The DiagnosticVerifier, which implements the -verify feature of the driver.
class DiagnosticVerifier final : public DiagnosticConsumer {
  std::unique_ptr<DiagnosticConsumer> consumer;
  SourceManager &srcMgr;
  bool unexpectedDiagsEmitted = false;

  /// An expected diagnostic's data. This does not contain the message.
  struct ExpectedDiagnosticData {
    ExpectedDiagnosticData(DiagnosticKind kind, BufferID buffer, unsigned line)
        : kind(kind), buffer(buffer), line(line) {}
    DiagnosticKind kind;
    BufferID buffer;
    unsigned line;
  };

  using ExpectedDiagnosticDataSet = std::list<ExpectedDiagnosticData>;

  /// The set of expected diagnostics.
  /// The string (message) is the key.
  llvm::StringMap<ExpectedDiagnosticDataSet> expectedDiagnostics;

  std::string getNoteUnemitted(StringRef message,
                               ExpectedDiagnosticData &data) const;

public:
  /// \param out the output stream where the DiagnosticVerifier will emit its
  /// error messages.
  /// \param srcMgr the SourceManager in which the files that will be treated by
  /// the DiagnosticVerifier are stored. This must be the same SourceManager
  /// used by the parent DiagnosticEngine.
  /// \param consumer the DiagnosticConsumer that will consume unexpected
  /// diagnostics.
  DiagnosticVerifier(raw_ostream &out, SourceManager &srcMgr,
                     std::unique_ptr<DiagnosticConsumer> consumer)
      : consumer(std::move(consumer)), srcMgr(srcMgr), out(out) {}

  /// Parses all of the "expect" lines in \p buff
  /// This prints error when error occurs.
  /// \returns true if the file was parsed successfully.
  bool parseFile(BufferID buffer);

  /// Handles a diagnostic. If the diagnostic was not expected by the verifier,
  /// it'll be forwarded to the consumer.
  void handle(SourceManager &srcMgr, const Diagnostic &diagnostic) override;

  /// Finishes verification
  /// \returns true if the verification succeeded.
  bool finish() const;

  DiagnosticConsumer *getConsumer() const { return consumer.get(); }
  std::unique_ptr<DiagnosticConsumer> takeConsumer() {
    return std::move(consumer);
  }
  void setConsumer(std::unique_ptr<DiagnosticConsumer> consumer) {
    this->consumer = std::move(consumer);
  }

  raw_ostream &out;
};
} // namespace sora
