//===--- DiagnosticConsumer.cpp ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/DiagnosticConsumer.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

using namespace sora;

namespace {
llvm::SourceMgr::DiagKind getDKasLLVM(DiagnosticKind kind) {
  using DiagKind = llvm::SourceMgr::DiagKind;
  switch (kind) {
  case DiagnosticKind::Remark:
    return DiagKind::DK_Note;
  case DiagnosticKind::Note:
    return DiagKind::DK_Note;
  case DiagnosticKind::Warning:
    return DiagKind::DK_Warning;
  case DiagnosticKind::Error:
    return DiagKind::DK_Error;
  default:
    llvm_unreachable("unknown DiagnosticKind");
  }
}
} // namespace

PrintingDiagnosticConsumer::PrintingDiagnosticConsumer() : out(llvm::outs()) {}

void PrintingDiagnosticConsumer::handle(SourceManager &srcMgr,
                                        const Diagnostic &diagnostic) {
  // Get the SMRanges
  SmallVector<llvm::SMRange, 2> ranges;
  for (auto range : diagnostic.ranges)
    ranges.push_back(range.getSMRange());
  // Get the Fixits
  SmallVector<llvm::SMFixIt, 4> fixits;
  for (auto fixit : diagnostic.fixits)
    fixits.push_back({fixit.getCharRange().getSMRange(), fixit.getText()});
  // Get the message
  auto msg = srcMgr.llvmSourceMgr.GetMessage(
      diagnostic.loc.getSMLoc(),    // diagnostic location
      getDKasLLVM(diagnostic.kind), // diagnostic kind
      diagnostic.message,           // diagnostic message
      ranges,                       // additional ranges
      fixits                        // fix-its
  );
  // Emit the message
  srcMgr.llvmSourceMgr.PrintMessage(out, msg, /*showColors*/ true);
}
