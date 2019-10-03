//===--- DiagnosticEngine.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticsCommon.hpp"
#include "Sora/Lexer/Lexer.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

namespace sora {
// Define the DiagID enum.
enum class DiagID : uint32_t {
#define DIAG(KIND, ID, TEXT, SIGNATURE) ID,
#include "Sora/Diagnostics/DiagnosticsAll.def"
};
// Define & Initialize all diagnostic objects
namespace diag {
#define DIAG(KIND, ID, TEXT, SIGNATURE)                                        \
  detail::TypedDiagHelper<void SIGNATURE>::type ID = {DiagID::ID};
#include "Sora/Diagnostics/DiagnosticsAll.def"
} // namespace diag
} // namespace sora

namespace {
/// Struct containing the information about a diagnostic.
struct DiagnosticData {
  const char *string;
  DiagnosticKind kind;
};

/// Array containing information about every diagnostic.
const constexpr DiagnosticData diagnosticData[] = {
#define DIAG(KIND, ID, STRING, SIGNATURE) {STRING, DiagnosticKind::KIND},
#include "Sora/Diagnostics/DiagnosticsAll.def"
};

/// \returns the raw, unformatted diagnostic string of a diagnostic
StringRef getRawDiagnosticString(DiagID id) {
  return StringRef(diagnosticData[(uint32_t)id].string);
}

/// \returns the default kind of a diagnostic
DiagnosticKind getDefaultDiagnosticKind(DiagID id) {
  return diagnosticData[(uint32_t)id].kind;
}
} // namespace

RawDiagnostic &InFlightDiagnostic::getRawDiagnostic() {
  assert(isActive() && "Diagnostic isn't active!");
  return diagEngine->activeDiagnostic.getValue();
}

InFlightDiagnostic::~InFlightDiagnostic() {
  if (diagEngine) {
    diagEngine->emit();
    diagEngine = nullptr;
  }
}

void InFlightDiagnostic::abort() {
  assert(isActive() && "cannot abort an already inactive diagnostic");
  diagEngine->abort();
  diagEngine = nullptr;
}

InFlightDiagnostic &InFlightDiagnostic::highlightChars(CharSourceRange range) {
  assert(isActive() && "cannot modify an inactive diagnostic");
  assert(range && "range is invalid");
  getRawDiagnostic().addRange(range);
  return *this;
}

InFlightDiagnostic &InFlightDiagnostic::highlight(SourceRange range) {
  assert(isActive() && "cannot modify an inactive diagnostic");
  assert(range && "range is invalid");
  return highlightChars(Lexer::toCharSourceRange(diagEngine->srcMgr, range));
}

InFlightDiagnostic &InFlightDiagnostic::fixitInsert(SourceLoc loc,
                                                    StringRef text) {
  fixitReplace(CharSourceRange(diagEngine->srcMgr, loc, loc), text);
  return *this;
}

InFlightDiagnostic &InFlightDiagnostic::fixitReplace(CharSourceRange range,
                                                     StringRef text) {
  assert(isActive() && "cannot modify an inactive diagnostic");
  getRawDiagnostic().addFixit(FixIt(text, range));
  return *this;
}

void DiagnosticEngine::initBitfields() {
  errorOccured = false;
  warningsAreErrors = false;
  ignoreAll = false;
}

SourceLoc DiagnosticEngine::getLocForDiag(BufferID buffer) const {
  return srcMgr.getBufferCharSourceRange(buffer).getBegin();
}

Optional<DiagnosticKind> DiagnosticEngine::getDiagnosticKind(DiagID id) {
  // If all diagnostics are to be ignored, don't even bother.
  if (ignoreAll)
    return None;
  // Find the default kind of this diagnostic.
  auto kind = getDefaultDiagnosticKind(id);
  ;
  // Promote to error if needed.
  if (warningsAreErrors && (kind == DiagnosticKind::Warning))
    return DiagnosticKind::Error;
  return kind;
}

void DiagnosticEngine::actOnDiagnosticEmission(DiagnosticKind kind) {
  if (kind == DiagnosticKind::Error)
    errorOccured = true;
}

namespace {
/// Finds all occurences of '%' + \p index in the string and replace them
/// with \p arg
///
/// e.g. if \p index is 2, this function will replace every "%2" inside
/// the string with \p arg
void replaceArgument(std::string &str, std::size_t index,
                     const std::string &arg) {
  // Generate the "placeholder" string.
  std::string placeholder = "%";
  placeholder += std::to_string(index);
#ifndef NDEBUG
  // In debug mode, check that we have replaced at least one occurence of
  // the argument, so we're sure every argument is used.
  unsigned numOccurencesReplaced = 0;
#endif
  std::size_t lastOccurence = str.find(placeholder, 0);
  // Loop until we replaced everything
  while (lastOccurence != std::string::npos) {
#ifndef NDEBUG
    ++numOccurencesReplaced;
#endif
    // Replace
    str.replace(lastOccurence, placeholder.size(), arg);
    // Search again?
    lastOccurence = str.find(placeholder, lastOccurence + 1);
  }
  assert(numOccurencesReplaced &&
         "no placeholder for this diagnostic argument (unused arg?)");
}

/// \returns the diagnostic string for \p id, formatted with \p providers
std::string
getFormattedDiagnosticString(DiagID id,
                             ArrayRef<DiagnosticArgumentProvider> providers) {
  std::string str = getRawDiagnosticString(id);
  for (std::size_t k = 0, size = providers.size(); k < size; ++k)
    replaceArgument(str, k, providers[k]());
  return str;
}
} // namespace

void DiagnosticEngine::emit() {
  assert(activeDiagnostic.hasValue() && "No active diagnostic!");
  assert(consumer && "can't emit a diagnostic without a consumer!");

  // Fetch the Diagnostic Kind, if it's null, abort the diag.
  RawDiagnostic &rawDiag = *activeDiagnostic;
  auto optDiagKind = getDiagnosticKind(rawDiag.getDiagID());
  if (!optDiagKind.hasValue()) {
    abort();
    return;
  }

  // Format the diagnsotic
  std::string diagStr = getFormattedDiagnosticString(rawDiag.getDiagID(),
                                                     rawDiag.getArgProviders());

  // Create the diagnostic object
  Diagnostic diag(diagStr, optDiagKind.getValue(), rawDiag.getLoc(),
                  rawDiag.getRanges(), rawDiag.getFixits());

  // Feed it to the consumer
  consumer->handle(srcMgr, diag);
  activeDiagnostic.reset();
}

void DiagnosticEngine::abort() { activeDiagnostic.reset(); }
