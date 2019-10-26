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
    tcArena.emplace(*ctxt);

    refType = ReferenceType::get(ctxt->i32Type, false);
    maybeType = MaybeType::get(ctxt->i32Type);
    lvalueType = LValueType::get(ctxt->i32Type);
    tupleType = TupleType::getEmpty(*ctxt);
    tyVarType = TypeVariableType::create(*ctxt, 0);
    fnType = FunctionType::get({}, refType);
  }

  ~TypeTest() {
    tcArena.reset();
    ctxt.reset();
  }

  IntegerType *getSignedInt(IntegerWidth width) {
    return IntegerType::getSigned(*ctxt, width);
  }

  IntegerType *getUnsignedInt(IntegerWidth width) {
    return IntegerType::getUnsigned(*ctxt, width);
  }

  Optional<TypeCheckerArenaRAII> tcArena;

  Type refType;
  Type maybeType;
  Type lvalueType;
  Type tupleType;
  Type tyVarType;
  Type fnType;

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

  EXPECT_TRUE(tupleType->is<TupleType>());

  EXPECT_TRUE(tyVarType->is<TypeVariableType>());

  EXPECT_TRUE(fnType->is<FunctionType>());
}

TEST_F(TypeTest, typeProperties) {
  EXPECT_FALSE(ctxt->errorType->hasTypeVariable());
  EXPECT_TRUE(ctxt->errorType->hasErrorType());

  EXPECT_TRUE(tyVarType->hasTypeVariable());
  EXPECT_FALSE(tyVarType->hasErrorType());
}

TEST_F(TypeTest, TupleType) {
  Type i32 = ctxt->i32Type;
  Type emptyTuple = TupleType::getEmpty(*ctxt);
  EXPECT_EQ(TupleType::get(*ctxt, {}).getPtr(), emptyTuple.getPtr());
  EXPECT_EQ(TupleType::get(*ctxt, {i32}).getPtr(), i32.getPtr());
  EXPECT_EQ(TupleType::get(*ctxt, {i32, i32}).getPtr(),
            TupleType::get(*ctxt, {i32, i32}).getPtr());
}

// Test for the propagation of TypeProperties on "simple" wrapper types:
// LValueType, ReferenceType, MaybeType
TEST_F(TypeTest, simpleTypePropertiesPropagation) {
#define CHECK(CREATE, HAS_TV, HAS_ERR)                                         \
  {                                                                            \
    auto ty = CREATE;                                                          \
    EXPECT_EQ(ty->hasTypeVariable(), HAS_TV);                                  \
    EXPECT_EQ(ty->hasErrorType(), HAS_ERR);                                    \
  }

  // i32: no properties set
  CHECK(LValueType::get(ctxt->i32Type), false, false);
  CHECK(ReferenceType::get(ctxt->i32Type, false), false, false);
  CHECK(MaybeType::get(ctxt->i32Type), false, false);

  // TypeVariableType
  ASSERT_TRUE(tyVarType->hasTypeVariable());
  CHECK(LValueType::get(tyVarType), true, false);
  CHECK(ReferenceType::get(tyVarType, false), true, false);
  CHECK(MaybeType::get(tyVarType), true, false);

  // ErrorType
  ASSERT_TRUE(ctxt->errorType->hasErrorType());
  CHECK(LValueType::get(ctxt->errorType), false, true);
  CHECK(ReferenceType::get(ctxt->errorType, false), false, true);
  CHECK(MaybeType::get(ctxt->errorType), false, true);

#undef CHECK
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

TEST_F(TypeTest, canonicalTypes_alwaysCanonicalTypes) {
#define CHECK_ALWAYS_CANONICAL(T)                                              \
  {                                                                            \
    auto can = T->getCanonicalType();                                          \
    EXPECT_EQ(can.getPtr(), T.getPtr());                                       \
    ASSERT_TRUE(can->isCanonical());                                           \
    ASSERT_TRUE(T->isCanonical());                                             \
  }
  CHECK_ALWAYS_CANONICAL(ctxt->f32Type)
  CHECK_ALWAYS_CANONICAL(ctxt->f64Type)
  CHECK_ALWAYS_CANONICAL(ctxt->i8Type)
  CHECK_ALWAYS_CANONICAL(ctxt->i16Type)
  CHECK_ALWAYS_CANONICAL(ctxt->i32Type)
  CHECK_ALWAYS_CANONICAL(ctxt->i64Type)
  CHECK_ALWAYS_CANONICAL(ctxt->isizeType)
  CHECK_ALWAYS_CANONICAL(ctxt->u8Type)
  CHECK_ALWAYS_CANONICAL(ctxt->u16Type)
  CHECK_ALWAYS_CANONICAL(ctxt->u32Type)
  CHECK_ALWAYS_CANONICAL(ctxt->u64Type)
  CHECK_ALWAYS_CANONICAL(ctxt->usizeType)
  CHECK_ALWAYS_CANONICAL(ctxt->voidType)
  CHECK_ALWAYS_CANONICAL(ctxt->errorType)
  Type tyVar = TypeVariableType::create(*ctxt, 0);
  CHECK_ALWAYS_CANONICAL(tyVar)
#undef CHECK_ALWAYS_CANONICAL
}