//===--- InitLLVM.hpp -  LLVM Initialization helpers ------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/InitLLVM.h"

/// PROGRAM_START should be called in the "main" function of every binary
#define PROGRAM_START(argc, argv) llvm::InitLLVM _llvm_init_(argc, argv)