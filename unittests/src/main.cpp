//===--- main.cpp - Entry point of the Sora Unit Test Executable *- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Common/InitLLVM.hpp"
#include "gtest/gtest.h"
#include <cstdio>

int main(int argc, char **argv) {
  PROGRAM_START(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
