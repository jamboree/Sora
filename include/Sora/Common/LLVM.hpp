//===--- LLVM.hpp - LLVM Forward Declarations and Imports -------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Forward declarations & imports for common LLVM types and functions that we
// want to use unqualified.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/Casting.h" // can't forward declare easily
#include "llvm/ADT/None.h" // can't forward declare at all

namespace llvm {
class raw_ostream;
class StringRef;
class APInt;
class APFloat;
template <typename T> class SmallVectorImpl;
template <typename T, unsigned N> class SmallVector;
template <typename T> class ArrayRef;
template <typename T> class Optional;
template <typename T> class MutableArrayRef;
} // namespace llvm

namespace sora {
// Casting
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::isa;

// Other
using llvm::raw_ostream;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using llvm::None;
using llvm::Optional;
using llvm::ArrayRef;
using llvm::MutableArrayRef;

// APInt & APFloat
using llvm::APInt;
using llvm::APFloat;
} // namespace sora