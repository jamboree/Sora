//===--- Options.cpp ---------------------------------------------*- C++-*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Driver/Options.hpp"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"

using namespace sora::opt;
using namespace llvm::opt;

namespace {
// Create the prefixes
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Sora/Driver/Options.inc"
#undef PREFIX

// Create the option infos
const llvm::opt::OptTable::Info optionsInfo[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {PREFIX,      NAME,      HELPTEXT,                                           \
   METAVAR,     OPT_##ID,  llvm::opt::Option::KIND##Class,                     \
   PARAM,       FLAGS,     OPT_##GROUP,                                        \
   OPT_##ALIAS, ALIASARGS, VALUES},
#include "Sora/Driver/Options.inc"
#undef OPTION
};

class SoraOptTable : public llvm::opt::OptTable {
public:
  SoraOptTable() : OptTable(optionsInfo) {}
};

} // namespace

std::unique_ptr<llvm::opt::OptTable> sora::createSoraOptTable() {
  return std::unique_ptr<llvm::opt::OptTable>(new SoraOptTable());
}