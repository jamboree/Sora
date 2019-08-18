//===--- main.cpp - Entry point of the Sora Unit Test Executable *- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include <cstdio>

// Check if we can leak-check using _Crt leak-checking tools
#if defined(_MSC_VER) && !defined(_NDEBUG)
#define CAN_LEAK_CHECK_ON_MSVC 1
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
#include <stdlib.h>
#else
#define CAN_LEAK_CHECK_ON_MSVC 0
#endif

int main(int argc, char **argv) {
// Enable leak checking under MSVC.
#if CAN_LEAK_CHECK_ON_MSVC
  _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif
  ::testing::InitGoogleTest(&argc, argv);
  auto result = RUN_ALL_TESTS();
#ifndef NDEBUG
  std::getchar();
#endif
  return result;
}
