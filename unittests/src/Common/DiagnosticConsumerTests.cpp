//===--- DiagnosticConsumerTests.cpp ----------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/DiagnosticConsumer.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace sora;

TEST(PrintingDiagnosticConsumerTest, handle) {
  StringRef str = "The Lazy Brown Fox Jumps Over The Lazy Dog";

  auto buff = llvm::MemoryBuffer::getMemBuffer(str, "some_file.sora");
  str = buff->getBuffer();
  SourceManager srcMgr;
  srcMgr.giveBuffer(std::move(buff));

  SourceLoc loc(llvm::SMLoc::getFromPointer(str.data() + 4));
  SourceLoc beg(llvm::SMLoc::getFromPointer(str.data() + 5));
  SourceLoc end(llvm::SMLoc::getFromPointer(str.data() + 7));

  CharSourceRange additionalRange(beg, 3);
  CharSourceRange wordRange(loc, 4);
  FixIt fixit("Hyperactive", wordRange);

  Diagnostic diag("I'm not lazy!", DiagnosticKind::Error, SourceLoc(loc),
                  wordRange, fixit);

  std::string output;
  llvm::raw_string_ostream stream(output);

  PrintingDiagnosticConsumer pdc(stream);
  pdc.handle(srcMgr, diag);

  stream.str();

  EXPECT_EQ(output, "some_file.sora:1:5: error: I'm not lazy!\n"
                    "The Lazy Brown Fox Jumps Over The Lazy Dog\n"
                    "    ^~~~\n"
                    "    Hyperactive\n");
}

TEST(PrintingDiagnosticConsumerTest, handleSimple) {
  SourceManager srcMgr;
  
  std::string output;
  llvm::raw_string_ostream stream(output);

  Diagnostic diag("I'm not lazy!", DiagnosticKind::Error, {});

  PrintingDiagnosticConsumer pdc(stream);
  pdc.handle(srcMgr, diag);
  EXPECT_EQ(stream.str(), "error: I'm not lazy!\n");
}