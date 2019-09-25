//===--- ParseType.cpp - Type Parsing Impl. ---------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Parser/Parser.hpp"

using namespace sora;

/*
type = identifier
     | tuple-type
     | array-type
     | reference-or-pointer-type
 

array-type = '[' type (';' expr)? ']'
tuple-type = '(' type (',' type)* ')'
reference-or-pointer-type = ('&' | '*') "mut"? type
*/
ParserResult<TypeRepr> Parser::parseType(std::function<void()> onNoType) {
  onNoType();
  return nullptr;
}