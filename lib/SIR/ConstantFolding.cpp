//===--- ConstantFolding.cpp ------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//
/// Implementation of the "fold" methods of the SIR Dialect Operations.
//
//===----------------------------------------------------------------------===//

#include "Sora/SIR/Dialect.hpp"
#include "mlir/IR/OpDefinition.h"

using namespace sora;
using namespace sora::sir;

// TODO