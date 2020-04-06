//===--- OperatorKinds.hpp - Kinds of Operators ----------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>
#include <string>

namespace sora {
template <typename Ty> struct DiagnosticArgument;

/// Kinds of Binary Operators
enum class BinaryOperatorKind : uint8_t {
#define BINARY_OP(ID, SPELLING) ID,
#include "Sora/AST/OperatorKinds.def"
};

/// \returns the spelling of a binary operator (e.g. "+" for Add)
const char *getSpelling(BinaryOperatorKind op);

/// \returns the operator kind as a string (e.g. "Add")
const char *to_string(BinaryOperatorKind op);

// Make BinaryOperatorKind usable in Diagnostic Arguments
template <> struct DiagnosticArgument<BinaryOperatorKind> {
  static std::string format(BinaryOperatorKind op) { return getSpelling(op); }
};

// Binary Operator Classification

/// \returns true if \p op is + or -
inline bool isAdditiveOp(BinaryOperatorKind op) {
  return (op == BinaryOperatorKind::Add) || (op == BinaryOperatorKind::Sub);
}
/// \returns true if \p op is * / or %
inline bool isMultiplicativeOp(BinaryOperatorKind op) {
  return (op >= BinaryOperatorKind::Mul) && (op <= BinaryOperatorKind::Rem);
}
/// \returns true if \p op is << or >>
inline bool isShiftOp(BinaryOperatorKind op) {
  return (op == BinaryOperatorKind::Shl) || (op == BinaryOperatorKind::Shr);
}
/// \returns true if \p op is | & or ^
inline bool isBitwiseOp(BinaryOperatorKind op) {
  return (op >= BinaryOperatorKind::And) && (op <= BinaryOperatorKind::XOr);
}
/// \returns true if \p op is == or !=
inline bool isEqualityOp(BinaryOperatorKind op) {
  return (op == BinaryOperatorKind::Eq) || (op == BinaryOperatorKind::NEq);
}
/// \returns true if \p op is < <= > or >=
inline bool isRelationalOp(BinaryOperatorKind op) {
  return (op >= BinaryOperatorKind::LT) && (op <= BinaryOperatorKind::GE);
}
/// \returns true if \p op is || or &&
inline bool isLogicalOp(BinaryOperatorKind op) {
  return (op == BinaryOperatorKind::LOr) || (op == BinaryOperatorKind::LAnd);
}
/// \returns true if \p op is any assignement operator
inline bool isAssignementOp(BinaryOperatorKind op) {
  return (op >= BinaryOperatorKind::Assign);
}
/// \returns true if \p op is a compound assignement operator
inline bool isCompoundAssignementOp(BinaryOperatorKind op) {
  return (op > BinaryOperatorKind::Assign);
}
/// \returns the operator of a compound assignement. e.g. for AddAssign this
/// returns Add.
BinaryOperatorKind getOpForCompoundAssignementOp(BinaryOperatorKind op);

/// Kinds of Unary Operators
enum class UnaryOperatorKind : uint8_t {
#define UNARY_OP(ID, SPELLING) ID,
#include "Sora/AST/OperatorKinds.def"
};

/// \returns the spelling of an unary operator (e.g. "+" for Plus)
const char *getSpelling(UnaryOperatorKind op);

/// \returns the operator kind as a string (e.g. "Plus")
const char *to_string(UnaryOperatorKind op);

// Make UnaryOperatorKind usable in Diagnostic Arguments
template <> struct DiagnosticArgument<UnaryOperatorKind> {
  static std::string format(UnaryOperatorKind op) { return getSpelling(op); }
};

} // namespace sora