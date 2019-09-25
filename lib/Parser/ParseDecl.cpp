//===--- ParseDecl.cpp - Declaration Parsing Impl. --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/Parser/Parser.hpp"

using namespace sora;

bool Parser::isStartOfDecl() const {
  switch (tok.getKind()) {
  case TokenKind::LetKw:
  case TokenKind::FuncKw:
    return true;
  default:
    return false;
  }
}
