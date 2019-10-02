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
  ASSERT_EQ(getOpForCompoundAssignementOp(COMPOUND_OP), EXPECTED)
  CHECK(Op::AddAssign, Op::Add);
  CHECK(Op::SubAssign, Op::Sub);
  CHECK(Op::MulAssign, Op::Mul);
  CHECK(Op::DivAssign, Op::Div);
  CHECK(Op::RemAssign, Op::Rem);
  CHECK(Op::ShlAssign, Op::Shl);
  CHECK(Op::ShrAssign, Op::Shr);
  CHECK(Op::AndAssign, Op::And);
  CHECK(Op::OrAssign, Op::Or);
  CHECK(Op::XOrAssign, Op::XOr);
  CHECK(Op::NullCoalesceAssign, Op::NullCoalesce);
#undef CHECK
}

TEST(BinaryOperatorKindTest, classificationSpellingAndToString) {
  using Op = BinaryOperatorKind;

#define OP(ID, SPELL, ADD, MUL, SH, BIT, EQ, REL, LOGIC, ASS, CASS)            \
  ASSERT_STRCASEEQ(getSpelling(Op::ID), SPELL);                                \
  ASSERT_STRCASEEQ(to_string(Op::ID), #ID);                                    \
  ASSERT_EQ(isAdditiveOp(Op::ID), ADD);                                        \
  ASSERT_EQ(isMultiplicativeOp(Op::ID), MUL);                                  \
  ASSERT_EQ(isShiftOp(Op::ID), SH);                                            \
  ASSERT_EQ(isBitwiseOp(Op::ID), BIT);                                         \
  ASSERT_EQ(isEqualityOp(Op::ID), EQ);                                         \
  ASSERT_EQ(isRelationalOp(Op::ID), REL);                                      \
  ASSERT_EQ(isLogicalOp(Op::ID), LOGIC);                                       \
  ASSERT_EQ(isAssignementOp(Op::ID), ASS);                                     \
  ASSERT_EQ(isCompoundAssignementOp(Op::ID), CASS)

  OP(Add, "+", 1, 0, 0, 0, 0, 0, 0, 0, 0);
  OP(Sub, "-", 1, 0, 0, 0, 0, 0, 0, 0, 0);
  OP(Mul, "*", 0, 1, 0, 0, 0, 0, 0, 0, 0);
  OP(Div, "/", 0, 1, 0, 0, 0, 0, 0, 0, 0);
  OP(Rem, "%", 0, 1, 0, 0, 0, 0, 0, 0, 0);
  OP(Shl, "<<", 0, 0, 1, 0, 0, 0, 0, 0, 0);
  OP(Shr, ">>", 0, 0, 1, 0, 0, 0, 0, 0, 0);
  OP(And, "&", 0, 0, 0, 1, 0, 0, 0, 0, 0);
  OP(Or, "|", 0, 0, 0, 1, 0, 0, 0, 0, 0);
  OP(XOr, "^", 0, 0, 0, 1, 0, 0, 0, 0, 0);
  OP(Eq, "==", 0, 0, 0, 0, 1, 0, 0, 0, 0);
  OP(NEq, "!=", 0, 0, 0, 0, 1, 0, 0, 0, 0);
  OP(LT, "<", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(LE, "<=", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(GT, ">", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(GE, ">=", 0, 0, 0, 0, 0, 1, 0, 0, 0);
  OP(LAnd, "&&", 0, 0, 0, 0, 0, 0, 1, 0, 0);
  OP(LOr, "||", 0, 0, 0, 0, 0, 0, 1, 0, 0);
  OP(NullCoalesce, "??", 0, 0, 0, 0, 0, 0, 0, 0, 0);
  OP(Assign, "=", 0, 0, 0, 0, 0, 0, 0, 1, 0);
  OP(AddAssign, "+=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(SubAssign, "-=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(MulAssign, "*=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(DivAssign, "/=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(RemAssign, "%=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(ShlAssign, "<<=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(ShrAssign, ">>=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(AndAssign, "&=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(OrAssign, "|=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(XOrAssign, "^=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
  OP(NullCoalesceAssign, "?\?=", 0, 0, 0, 0, 0, 0, 0, 1, 1);
#undef OP
}

TEST(UnaryOperatorKindTest, spellingAndToString) {
  using Op = UnaryOperatorKind;
#define OP(KIND, SPELLING)                                                     \
  ASSERT_STRCASEEQ(getSpelling(Op::KIND), SPELLING);                           \
  ASSERT_STRCASEEQ(to_string(Op::KIND), #KIND);

  OP(Plus, "+");
  OP(Minus, "-");
  OP(LNot, "!");
  OP(Not, "~");
  OP(Deref, "*");
  OP(AddressOf, "&");

#undef OP
}
