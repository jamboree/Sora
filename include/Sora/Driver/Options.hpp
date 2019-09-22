//===--- Options.hpp - Compiler Driver Options ------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Command-line Options
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>

namespace llvm {
namespace opt {
class OptTable;
}
} // namespace llvm

namespace sora {
// TODO: Find better names for this. I must have C enums for llvm's OptTable,
// but this naming conventionn (borrowed from Swift) kinda sucks.
namespace opt {
/// Option flags, must not overlap with llvm::opt::DriverFlag.
/* UNUSED FOR NOW
enum SoraFlags : unsigned {
  /// Options that affect the Sora front-end, this includes the Driver, Lexer,
  /// Parser, etc. up to CodeGen.
  FrontendOption = (1 << 4)
};
*/
enum OptionID : unsigned {
  OPT_INVALID,
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Sora/Driver/Options.inc"
#undef OPTION
};
} // namespace opt

/// \returns a Sora command-line option table (llvm::opt::OptTable)
std::unique_ptr<llvm::opt::OptTable> createSoraOptTable();

} // namespace sora