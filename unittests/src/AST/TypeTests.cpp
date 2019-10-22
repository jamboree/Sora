//===--- TypeTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "llvm/ADT/Triple.h"
#include "gtest/gtest.h"

using namespace sora;

namespace {
class TypeTest : public ::testing::Test {
protected:
  TypeTest() {
    refType = ReferenceType::get(*ctxt, ctxt->i32Type, false);
    maybeType = MaybeType::get(*ctxt, ctxt->i32Type);
    lvalueType = LValueType::get(*ctxt, ctxt->i32Type);
  }

  IntegerType *getSignedInt(IntegerWidth width) {
    return IntegerType::getSigned(*ctxt, width);
  }

  IntegerType *getUnsignedInt(IntegerWidth width) {
    return IntegerType::getUnsigned(*ctxt, width);
  }

  Type refType;
  Type maybeType;
  Type lvalueType;

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};
};
} // namespace

TEST_F(TypeTest, rtti) {
  EXPECT_TRUE(ctxt->i32Type->is<IntegerType>());
  EXPECT_TRUE(ctxt->i32Type->is<BuiltinType>());

  EXPECT_TRUE(ctxt->f32Type->is<FloatType>());
  EXPECT_TRUE(ctxt->f32Type->is<BuiltinType>());

  EXPECT_TRUE(ctxt->voidType->is<VoidType>());
  EXPECT_TRUE(ctxt->voidType->is<BuiltinType>());

  EXPECT_TRUE(ctxt->errorType->is<ErrorType>());

  EXPECT_TRUE(refType->is<ReferenceType>());

  EXPECT_TRUE(maybeType->is<MaybeType>());

  EXPECT_TRUE(lvalueType->is<LValueType>());
}

TEST_F(TypeTest, ASTContextSingletons) {
  EXPECT_EQ(ctxt->i8Type.getPtr(), getSignedInt(IntegerWidth::fixed(8)));
  EXPECT_EQ(ctxt->i16Type.getPtr(), getSignedInt(IntegerWidth::fixed(16)));
  EXPECT_EQ(ctxt->i32Type.getPtr(), getSignedInt(IntegerWidth::fixed(32)));
  EXPECT_EQ(ctxt->i64Type.getPtr(), getSignedInt(IntegerWidth::fixed(64)));
  EXPECT_EQ(ctxt->isizeType.getPtr(),
            getSignedInt(IntegerWidth::pointer(ctxt->getTargetTriple())));

  EXPECT_EQ(ctxt->u8Type.getPtr(), getUnsignedInt(IntegerWidth::fixed(8)));
  EXPECT_EQ(ctxt->u16Type.getPtr(), getUnsignedInt(IntegerWidth::fixed(16)));
  EXPECT_EQ(ctxt->u32Type.getPtr(), getUnsignedInt(IntegerWidth::fixed(32)));
  EXPECT_EQ(ctxt->u64Type.getPtr(), getUnsignedInt(IntegerWidth::fixed(64)));
  EXPECT_EQ(ctxt->usizeType.getPtr(),
            getUnsignedInt(IntegerWidth::pointer(ctxt->getTargetTriple())));

  EXPECT_EQ(ctxt->f32Type.getPtr(), FloatType::get(*ctxt, FloatKind::IEEE32));
  EXPECT_EQ(ctxt->f64Type.getPtr(), FloatType::get(*ctxt, FloatKind::IEEE64));
}

TEST_F(TypeTest, integerTypes) {
#define CHECK(MEMBER, WIDTH, ISSIGNED)                                         \
  EXPECT_EQ(ctxt->MEMBER->castTo<IntegerType>()->getWidth(), WIDTH);           \
  EXPECT_EQ(ctxt->MEMBER->castTo<IntegerType>()->isSigned(), ISSIGNED)

  CHECK(i8Type, IntegerWidth::fixed(8), true);
  CHECK(i16Type, IntegerWidth::fixed(16), true);
  CHECK(i32Type, IntegerWidth::fixed(32), true);
  CHECK(i64Type, IntegerWidth::fixed(64), true);
  CHECK(isizeType, IntegerWidth::pointer(ctxt->getTargetTriple()), true);

  CHECK(u8Type, IntegerWidth::fixed(8), false);
  CHECK(u16Type, IntegerWidth::fixed(16), false);
  CHECK(u32Type, IntegerWidth::fixed(32), false);
  CHECK(u64Type, IntegerWidth::fixed(64), false);
  CHECK(usizeType, IntegerWidth::pointer(ctxt->getTargetTriple()), false);

#undef CHECK
}