//===--- Type.hpp - Type Object ---------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// This file contains the "Type" object which is a wrapper around TypeBase
// pointers. This is needed to disable direct type pointer comparison, which
// can be deceiving due to the canonical/non-canonical type distinction.
//===----------------------------------------------------------------------===//

#pragma once

namespace sora {

}