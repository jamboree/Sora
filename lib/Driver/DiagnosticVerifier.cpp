//===--- DiagnosticVerifier.cpp ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Driver/DiagnosticVerifier.hpp"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

using namespace sora;

using DiagKind = llvm::SourceMgr::DiagKind;

namespace {

/// Attempts to remove \p expected from \p str, and returns true if it was
/// correctly removed.
bool tryConsume(StringRef &str, StringRef expected) {
  if (str.startswith(expected)) {
    str = str.substr(expected.size());
    return true;
  }
  return false;
}

/// Attempts to consume a number from the input
Optional<size_t> tryConsumeNumber(StringRef &str) {
  size_t k = 0;
  while ((k < str.size()) && isdigit(str[k]))
    ++k;
  // No digit
  if (k == 0)
    return None;
  // Got digits, k is the position of the first char that isn't a digit.
  StringRef numberStr = str.substr(0, k);
  // Remove the number from str
  str = str.substr(k);
  // Parse the number & return it.
  size_t result;
  if (numberStr.getAsInteger(10, result))
    llvm_unreachable(
        "StringRef::getAsInteger() failed parsing a string with only digits?");
  return result;
}
} // namespace

std::string
DiagnosticVerifier::getNoteUnemitted(StringRef message,
                                     ExpectedDiagnosticData &data) const {
  std::string str;
  llvm::raw_string_ostream rso(str);
  switch (data.kind) {
  case DiagnosticKind::Remark:
    rso << "remark";
    break;
  case DiagnosticKind::Note:
    rso << "note";
    break;
  case DiagnosticKind::Warning:
    rso << "warning";
    break;
  case DiagnosticKind::Error:
    rso << "error";
    break;
  }
  rso << " '" << message << "' expected at line " << data.line << " of '"
      << srcMgr.getBufferName(data.buffer) << "' was not emitted";
  return rso.str();
}

bool DiagnosticVerifier::parseFile(BufferID buffer) {
  constexpr char prefix[] = "expect-";
  constexpr size_t prefixLen = sizeof(prefix) - 1;

  bool parsingSuccessful = true;
  auto error = [&](SourceLoc loc, StringRef message) {
    srcMgr.llvmSourceMgr.PrintMessage(out, loc.getSMLoc(), DiagKind::DK_Error,
                                      message);
    parsingSuccessful = false;
  };

  assert(buffer && "invalid buffer");
  StringRef file = srcMgr.getBufferStr(buffer);

  unsigned lastLine = srcMgr.findLineNumber(SourceLoc::fromPointer(file.end()));

  for (size_t matchPos = file.find(prefix); matchPos != StringRef::npos;
       matchPos = file.find(prefix, matchPos + 1)) {
    // Create a string that starts at the beginning of the line
    StringRef str = file.substr(matchPos);

    auto getCurLoc = [&]() {
      return SourceLoc::fromPointer(str.data());
    };
    SourceLoc matchBegLoc = getCurLoc();

    // Consume the prefix
    str = str.substr(prefixLen);

    // (number "-") ?
    size_t diagCount = 1;
    if (auto result = tryConsumeNumber(str)) {
      if (!tryConsume(str, "-")) {
        error(getCurLoc(), "expected '-'");
        continue;
      }
      diagCount = *result;
      if (diagCount <= 1) {
        error(matchBegLoc, "expected diagnostic count must be greater than 1");
        continue;
      }
    }

    // kind = "remark" | "note" | "warning" | "error"
    DiagnosticKind kind;
    if (tryConsume(str, "remark"))
      kind = DiagnosticKind::Remark;
    else if (tryConsume(str, "note"))
      kind = DiagnosticKind::Note;
    else if (tryConsume(str, "warning"))
      kind = DiagnosticKind::Warning;
    else if (tryConsume(str, "error"))
      kind = DiagnosticKind::Error;
    else {
      error(getCurLoc(),
            "expected number, 'remark', 'note', 'warning' or 'error'");
      continue;
    }

    int offset = 0;
    // offset = '@' ('+' | '-') number
    if (tryConsume(str, "@")) {
      // Factor is set to 1 for positive offsets, -1 for negative ones.
      int factor = 0;
      if (tryConsume(str, "+"))
        factor = 1;
      else if (tryConsume(str, "-"))
        factor = -1;
      else {
        // no + or -, diagnose and ignore this "expect-" line.
        error(getCurLoc(), "expected '+' or '-'");
        continue;
      }

      SourceLoc numBegLoc = getCurLoc();
      if (auto result = tryConsumeNumber(str)) {
        if (*result == 0) {
          error(numBegLoc, "offset number can't be zero");
          continue;
        }
        offset = *result * factor;
      }
    }

    if (!tryConsume(str, ":")) {
      error(getCurLoc(), "expected ':' or '@'");
      continue;
    }

    // The rest of the string up until the newline or the end of the file is our
    // diagnostic string.
    StringRef diagStr = str;
    size_t end = diagStr.find("\n");
    if (end != StringRef::npos) {
      if (str[end - 1] == '\r')
        --end;
      diagStr = diagStr.substr(0, end);
    }
    diagStr = diagStr.trim();

    // Calculate the line number
    unsigned line = srcMgr.findLineNumber(matchBegLoc);

    // Check that we don't have a negative offset that'd overflow. (e.g. line is
    // 5, but offset is -6)
    if ((offset < 0) && (line < (unsigned)-offset)) {
      error(matchBegLoc,
            "cannot expect a diagnostic at a negative line number");
      continue;
    }
    line += offset;

    // Check if the line number is valid
    if (line == 0) {
      error(matchBegLoc, "cannot expect a diagnostic at line 0");
      continue;
    }
    if (line > lastLine) {
      auto str = llvm::formatv("diagnostic is expected at line {0} but the "
                               "file's last line is line {1}",
                               line, lastLine);
      error(matchBegLoc, str.str());
    }

    while (diagCount--)
      expectedDiagnostics[diagStr].push_back({kind, buffer, line});
  }
  return parsingSuccessful;
}

void DiagnosticVerifier::handle(const SourceManager &srcMgr,
                                const Diagnostic &diagnostic) {
  assert((&(this->srcMgr) == &srcMgr) &&
         "The SourceManager used by the DiagnosticEngine is different from the "
         "one used by the DiagnosticVerifier!");
  auto fail = [&]() {
    unexpectedDiagsEmitted = true;
    if (consumer)
      consumer->handle(srcMgr, diagnostic);
  };

  // Did we expect this diagnostic?
  auto entry = expectedDiagnostics.find(diagnostic.message);
  if (entry == expectedDiagnostics.end() || entry->second.empty())
    return fail();

  // Look for the correct one in the buffer
  ExpectedDiagnosticDataSet &expectedDiags = entry->second;
  for (auto it = expectedDiags.begin(); it != expectedDiags.end(); ++it) {
    // Compare kinds
    if (it->kind != diagnostic.kind)
      continue;
    // Compare locs
    auto locBuff = srcMgr.findBufferContainingLoc(diagnostic.loc);
    if (it->buffer != locBuff)
      continue;
    // Compare lines
    if (it->line != srcMgr.findLineNumber(diagnostic.loc))
      continue;

    // It's a match, delete this entry from the set of expected diagnostics and
    // return.
    expectedDiags.erase(it);
    if (expectedDiags.size() == 0)
      expectedDiagnostics.erase(entry);
    return;
  }
  return fail();
}

bool DiagnosticVerifier::finish() const {
  std::vector<std::string> unemittedDiagNotes;
  for (auto &entry : expectedDiagnostics) {
    auto &set = entry.second;
    if (set.empty())
      continue;

    for (auto unemitted : set)
      unemittedDiagNotes.push_back(getNoteUnemitted(entry.first(), unemitted));
  }
  if (unemittedDiagNotes.empty())
    return !unexpectedDiagsEmitted;

  std::string err = llvm::formatv(
      "verification failed: {0} diagnostics were expected but not emitted",
      unemittedDiagNotes.size());
  srcMgr.llvmSourceMgr.PrintMessage(out, llvm::SMLoc(), DiagKind::DK_Error,
                                    err);
  for (auto &note : unemittedDiagNotes)
    srcMgr.llvmSourceMgr.PrintMessage(out, llvm::SMLoc(), DiagKind::DK_Note,
                                      note);
  return false;
}