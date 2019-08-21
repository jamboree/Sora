﻿//===--- DiagnosticEngineTests.cpp ------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/DiagnosticsCommon.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace sora;

TEST(DiagnosticEngine, diagnose) {
  StringRef str = "The Lazy Brown Fox Jumps Over The Lazy Dog";
  StringRef name = "some_file.sora";

  auto buff = llvm::MemoryBuffer::getMemBuffer(str, name);
  str = buff->getBuffer();
  SourceManager srcMgr;
  srcMgr.giveBuffer(std::move(buff));

  SourceLoc loc = SourceLoc::fromPointer(str.data() + 4);
  SourceLoc ins = SourceLoc::fromPointer(str.data() + 3);
  SourceLoc beg = SourceLoc::fromPointer(str.data() + 5);
  SourceLoc end = SourceLoc::fromPointer(str.data() + 7);

  CharSourceRange additionalRange(beg, 3);
  CharSourceRange wordRange(loc, 4);

  std::string output;
  llvm::raw_string_ostream stream(output);

  DiagnosticEngine diagEngine(srcMgr, stream);
  diagEngine.diagnose(loc, diag::hello_world, "Pierre", 4)
      .highlightChars(additionalRange)
      .fixitInsert(ins, "Incredibely")
      .fixitReplace(wordRange, "Hyperactive");

  stream.str();

  ASSERT_NE(output.size(), 0) << "No Output.";
  EXPECT_EQ(output, "some_file.sora:1:5: note: Hello, Pierre! I've greeted you "
                    "4 times already!\n"
                    "The Lazy Brown Fox Jumps Over The Lazy Dog\n"
                    "    ^~~~\n"
                    "   Incredibely Hyperactive\n");
}