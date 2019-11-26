//===--- Identifier.cpp -----------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Identifier.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

StringRef Identifier::str() const { return StringRef(value); }

bool Identifier::operator==(StringRef other) const { return str() == other; }

bool Identifier::operator!=(StringRef other) const { return str() != other; }

bool Identifier::operator<(StringRef other) const { return str() <= other; }

llvm::raw_ostream &sora::operator<<(llvm::raw_ostream &out, Identifier ident) {
  out << ident.str();
  return out;
}
