//===--- ASTContextTests.cpp ------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/OperatorKinds.hpp"
#include "gtest/gtest.h"

using namespace sora;

TEST(BinaryOperatorKindTest, getOpForCompoundAssignementOp) {
  using Op = BinaryOperatorKind;
#define CHECK(COMPOUND_OP, EXPECTED)                                           \
  EXPECT_EQ(getOpForCompoundAssignementOp(COMPOUND_OP), EXPECTED)
  CHECK(Op::AddAssign, Op::Add);
  CHECK(Op::SubAssign, Op::Sub);
  CHECK(Op::MulAssign, Op::Mul);
  CHECK(Op::DivAssign, Op::Div);
  CHECK(Op::ModAssign, Op::Mod);
  CHECK(Op::ShlAssign, Op::Shl);
  CHECK(Op::ShrAssign, Op::Shr);
  CHECK(Op::AndAssign, Op::And);
  CHECK(Op::OrAssign, Op::Or);
  CHECK(Op::XOrAssign, Op::XOr);
#undef CHECK
}

TEST(BinaryOperatorKindTest, classificationAndSpelling) {
  using Op = BinaryOperatorKind;

#define OP(ID, SPELL, ADD, MUL, SH, BIT, EQ, REL, LOGIC, ASS, CASS)            \
  EXPECT_STRCASEEQ(getSpelling(ID), SPELL);                                    \
  EXPECT_EQ(isAdditiveOp(ID), ADD);                                            \
  EXPECT_EQ(isMultiplicativeOp(ID), MUL);                                      \
  EXPECT_EQ(isShiftOp(ID), SH);                                                \
  EXPECT_EQ(isBitwiseOp(ID), BIT);                                             \
  EXPECT_EQ(isEqualityOp(ID), EQ);                                             \
  EXPECT_EQ(isRelationalOp(ID), REL);                                          \
  EXPECT_EQ(isLogicalOp(ID), LOGIC);                                           \
  EXPECT_EQ(isAssignementOp(ID), ASS);                                         \
  EXPECT_EQ(isCompoundAssignementOp(ID), CASS)

  OP(Op::Add, "+", 1, 0, 0, 0, 0, 0, 0, 0, 0);
  OP(Op::Sub, "-", 1, 0, 0, 0, 0, 0, 0, 0, 0);
  OP(Op::Mul, "*", 0, 1, 0, 0, 0, 0, 0, 0, 0);
  OP(Op::Div, "/", 0, 1, 0, 0, 0, 0, 0, 0, 0);
  OP(Op::Mod, "%", 0, 1, 0, 0, 0, 0, 0, 0, 0);
  OP(Op::Shl, "<<", 0, 0, 1, 0, 0, 0, 0, 0, 0);
  OP(Op::Shr, ">>", 0, 0, 1, 0, 0, 0, 0, 0, 0);
  OP(Op::And, "&", 0, 0, 0, 1, 0, 0, 0, 0, 0);
  OP(Op::Or, "|", 0, 0, 0, 1, 0, 0, 0, 0, 0);
  OP(Op::XOr, "^", 0, 0, 0, 1, 0, 0, 0, 0, 0);
  OP(Op::Eq, "==", 0, 0, 0, 0, 1, 0, 0, 0, 0);
  OP(Op::NEq, "!=", 0, 0, 0, 0, 1, 0, 0, 0, 0);
  OP(Op::LT, "<", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(Op::LE, "<=", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(Op::GT, ">", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(Op::GE, ">=", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(Op::LAnd, "&&", 0, 0, 0, 0, 0, 0, 1, 0, 0);
  OP(Op::LOr, "||", 0, 0, 0, 0, 0, 0, 1, 0, 0);
  OP(Op::Assign, "=", 0, 0, 0, 0, 0, 0, 0, 1, 0);
  OP(Op::AddAssign, "+=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::SubAssign, "-=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::MulAssign, "*=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::DivAssign, "/=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::ModAssign, "%=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::ShlAssign, "<<=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::ShrAssign, ">>=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::AndAssign, "&=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::OrAssign, "|=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(Op::XOrAssign, "^=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
#undef OP
}

TEST(UnaryOperatorKindTest, spelling) {
  using Op = UnaryOperatorKind;
  EXPECT_STRCASEEQ(getSpelling(Op::Plus), "+");
  EXPECT_STRCASEEQ(getSpelling(Op::Minus), "-");
  EXPECT_STRCASEEQ(getSpelling(Op::LNot), "!");
  EXPECT_STRCASEEQ(getSpelling(Op::Not), "~");
  EXPECT_STRCASEEQ(getSpelling(Op::Deref), "*");
  EXPECT_STRCASEEQ(getSpelling(Op::AddressOf), "&");
}