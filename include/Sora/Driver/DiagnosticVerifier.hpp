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
//  verify-line = "expect-" severity offset? ':' text
//  severiy = "remark" | "note" | "warning" | "error"
//  offset = ('+' | '-') number
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/Common/DiagnosticConsumer.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/StringMap.h"
#include <memory>

namespace sora {
struct Diagnostic;

/// The DiagnosticVerifier, which implements the -verify feature of the driver.
class DiagnosticVerifier final : public DiagnosticConsumer {
  std::unique_ptr<DiagnosticConsumer> consumer;
  SourceManager &srcMgr;
  bool success = true;

  /// An expected diagnostic's data. This does not contain the message.
  struct ExpectedDiagnosticData {
    ExpectedDiagnosticData(DiagnosticKind kind, BufferID buffer, unsigned line)
        : kind(kind), buffer(buffer), line(line) {}
    DiagnosticKind kind;
    BufferID buffer;
    unsigned line;
  };

  /// The set of expected diagnostics.
  /// The string (message) is the key.
  llvm::StringMap<ExpectedDiagnosticData> expectedDiagnostics;

public:
  /// \param out the output stream where the DiagnosticVerifier will emit its
  /// error messages.
  /// \param srcMgr the SourceManager in which the files that will be treated by
  /// the DiagnosticVerifier are stored. This must be the same SourceManager
  /// used by the parent DiagnosticEngine.
  /// \param consumer the DiagnosticConsumer that will consume unverified
  /// diagnostics.
  DiagnosticVerifier(raw_ostream &out, SourceManager &srcMgr,
                     std::unique_ptr<DiagnosticConsumer> consumer)
      : out(out), srcMgr(srcMgr), consumer(std::move(consumer)) {}

  /// Parses all of the "expect" lines in \p buff
  /// \returns true if the file was parsed successfully.
  bool parseFile(BufferID buffer);

  /// Handles a diagnostic. If the diagnostic was not expected by the verifier,
  /// it'll be forwarded to the consumer.
  void handle(SourceManager &srcMgr, const Diagnostic &diagnostic) override;

  /// \returns true if the verification succeeded.
  bool isSuccess() const { return success; }

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
