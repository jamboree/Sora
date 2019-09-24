//===--- DiagnosticVerifier.cpp ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Driver/DiagnosticVerifier.hpp"
#include "llvm/Support/ErrorHandling.h"

using namespace sora;
namespace {
constexpr char prefix[] = "expect-";
constexpr size_t prefixLen = sizeof(prefix) - 1;

/// Attempts to remove \p expected from \p str, and returns true if it was
/// correctly removed.
bool tryConsume(StringRef &str, StringRef expected) {
  if (str.startswith(expected)) {
    str = str.substr(expected.size());
    return true;
  }
  return false;
}
} // namespace

bool DiagnosticVerifier::parseFile(BufferID buffer) {
  assert(buffer && "invalid buffer");
  StringRef file = srcMgr.getBufferStr(buffer);
  assert(file && "empty/invalid file?");

  for (size_t match = file.find(prefix); match != StringRef::npos;
       match = file.find(prefix, match + 1)) {
    // Extract the string and remove the prefix.
    StringRef str = file.substr(match + prefixLen);
    auto getCurLoc = [&]() { return SourceLoc::fromPointer(str.data()); };

    SourceLoc matchBegLoc = getCurLoc();

    // Get the kind of the diagnostic & consume it
    DiagnosticKind kind;
    if (tryConsume(str, "remark"))
      kind = DiagnosticKind::Remark;
    else if (tryConsume(str, "note"))
      kind = DiagnosticKind::Note;
    else if (tryConsume(str, "warning"))
      kind = DiagnosticKind::Warning;
    else if (tryConsume(str, "error"))
      kind = DiagnosticKind::Error;
    else
      continue;

    unsigned offset = 0;
    // parse + - or : 
    return false;
  }
}

void DiagnosticVerifier::handle(SourceManager &srcMgr,
                                const Diagnostic &diagnostic) {
  assert((&(this->srcMgr) == &srcMgr) &&
         "The SourceManager used by the DiagnosticEngine is different from the "
         "one used by the DiagnosticVerifier!");
}
