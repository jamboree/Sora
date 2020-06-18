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
#define SORA_FWD_DECL(X) X;
#include "Sora/Diagnostics/DiagnosticsAll.def"

// Define all diagnostic objects.
namespace diag {
#define DIAG(KIND, ID, TEXT, SIGNATURE)                                        \
  detail::TypedDiagHelper<void SIGNATURE>::type ID = {DiagnosticKind::KIND,    \
                                                      TEXT};
#include "Sora/Diagnostics/DiagnosticsAll.def"
} // namespace diag
} // namespace sora

DiagnosticEngine::DiagnosticData &InFlightDiagnostic::getDiagnosticData() {
  assert(isActive() && "Diagnostic isn't active!");
  return diagEngine->activeDiagnostic.getValue();
}

CharSourceRange InFlightDiagnostic::toCharSourceRange(SourceRange range) const {
  assert(diagEngine && "needs a DiagnosticEngine for this");
  return Lexer::toCharSourceRange(diagEngine->srcMgr, range);
}
bool InFlightDiagnostic::canAddInfo() const {
  return isActive() && getDiagnosticData().loc.isValid();
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
  assert(canAddInfo() && "diagnostic is inactive or doesn't have a loc");
  assert(range && "range is invalid");
  getDiagnosticData().ranges.push_back(range);
  return *this;
}

InFlightDiagnostic &InFlightDiagnostic::highlight(SourceRange range) {
  assert(canAddInfo() && "diagnostic is inactive or doesn't have a loc");
  assert(range && "range is invalid");
  return highlightChars(toCharSourceRange(range));
}

InFlightDiagnostic &InFlightDiagnostic::fixitInsert(SourceLoc loc,
                                                    StringRef text) {
  assert(canAddInfo() && "diagnostic is inactive or doesn't have a loc");
  fixitReplace(CharSourceRange(diagEngine->srcMgr, loc, loc), text);
  return *this;
}

InFlightDiagnostic &InFlightDiagnostic::fixitInsertAfter(SourceLoc loc,
                                                         StringRef text) {
  assert(canAddInfo() && "diagnostic is inactive or doesn't have a loc");
  loc = Lexer::getLocPastTheEndOfTokenAtLoc(diagEngine->srcMgr, loc);
  return fixitInsert(loc, text);
}

InFlightDiagnostic &InFlightDiagnostic::fixitReplace(CharSourceRange range,
                                                     StringRef text) {
  assert(canAddInfo() && "diagnostic is inactive or doesn't have a loc");
  getDiagnosticData().fixits.push_back(FixIt(text.str(), range));
  return *this;
}

InFlightDiagnostic &InFlightDiagnostic::fixitRemove(SourceRange range) {
  assert(canAddInfo() && "diagnostic is inactive or doesn't have a loc");
  CharSourceRange charRange = toCharSourceRange(range);
  // When we're removing something that has :
  //    - a space ' ' after it
  //    - a whitespace before it
  // Remove the extra space after the range to keep the source consistent.
  const SourceManager &srcMgr = diagEngine->srcMgr;
  BufferID buffer = srcMgr.findBufferContainingLoc(range.begin);
  assert(buffer && "bogus range?");
  CharSourceRange fileRange = srcMgr.getBufferCharSourceRange(buffer);
  assert(fileRange.contains(charRange));

  bool hasWhitespaceBefore = false;
  if (charRange.getBegin() == fileRange.getBegin())
    hasWhitespaceBefore = true;
  else
    hasWhitespaceBefore = isspace(*(charRange.getBegin().getPointer() - 1));

  if (hasWhitespaceBefore) {
    bool hasSpaceAfter = false;
    if (charRange.getEnd() == fileRange.getEnd())
      hasSpaceAfter = false;
    else
      hasSpaceAfter = (*charRange.getEnd().getPointer() == ' ');

    if (hasSpaceAfter)
      charRange =
          CharSourceRange(charRange.getBegin(), charRange.getByteLength() + 1);
  }

  return fixitReplace(charRange, {});
}

void DiagnosticEngine::initBitfields() {
  errorOccured = false;
  warningsAreErrors = false;
  ignoreAll = false;
}

SourceLoc DiagnosticEngine::getLocForDiag(BufferID buffer) const {
  return srcMgr.getBufferCharSourceRange(buffer).getBegin();
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

} // namespace

void DiagnosticEngine::emit() {
  assert(activeDiagnostic.hasValue() && "No active diagnostic!");

  if (ignoreAll) {
    abort();
    return;
  }

  DiagnosticData &diagData = *activeDiagnostic;

  // Promote the diagnostic to an error if needed.
  DiagnosticKind kind = diagData.kind;
  if (warningsAreErrors && (kind == DiagnosticKind::Warning))
    kind = DiagnosticKind::Error;

  // Format the diagnsotic
  std::string diagStr =
      formatDiagnosticString(diagData.str, diagData.argProviders);

  // Create the diagnostic object
  Diagnostic diag(diagStr, kind, diagData.loc, diagData.ranges,
                  diagData.fixits);

  // Feed it to the consumer if there's one
  if (consumer)
    consumer->handle(srcMgr, diag);
  actOnDiagnosticEmission(diag.kind);
  activeDiagnostic.reset();
}

void DiagnosticEngine::abort() { activeDiagnostic.reset(); }

/// \returns the diagnostic string for \p id, formatted with \p providers
std::string DiagnosticEngine::formatDiagnosticString(
    const char *diagStr, ArrayRef<DiagnosticEngine::ArgProviderFn> providers) {
  std::string str = diagStr;
  for (std::size_t k = 0, size = providers.size(); k < size; ++k)
    replaceArgument(str, k, providers[k]());
  return str;
}