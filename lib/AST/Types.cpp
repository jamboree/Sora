//===--- Types.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Types.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "llvm/ADT/APInt.h"

using namespace sora;

SourceRange TypeLoc::getSourceRange() const {
  return tyRepr ? tyRepr->getSourceRange() : SourceRange();
}

SourceLoc TypeLoc::getBegLoc() const {
  return tyRepr ? tyRepr->getBegLoc() : SourceLoc();
}

SourceLoc TypeLoc::getLoc() const {
  return tyRepr ? tyRepr->getLoc() : SourceLoc();
}

SourceLoc TypeLoc::getEndLoc() const {
  return tyRepr ? tyRepr->getEndLoc() : SourceLoc();
}

void *operator new(size_t size, ASTContext &ctxt,
                   unsigned align = alignof(TypeBase)) {
  return ctxt.allocate(size, align);
}

APInt IntegerWidth::parse(StringRef str, int isNegative, unsigned radix,
                          Status *status) const {
  assert(isDenseMapSpecial() && "cant be used on DenseMap special value");
  APInt result;

  auto finish = [&](Status s) {
    if (status)
      *status = s;
    return result;
  };

  assert(radix && "Radix can't be zero!");
  // getAsInteger returns true on error
  if (str.getAsInteger(radix, result))
    return finish(Status::Error);

  // For arbitrary-precision integers, we don't have too much additional
  // processing to do, but we do need to take care of the sign bit.
  if (isArbitraryPrecision()) {
    // getAsInteger always return a non-negative value, but that value can be
    // considered negative if the bit that is used as the sign bit is 1. If
    // that's the case, zero-extend the value to give it a proper sign bit.
    // e.g. if it returns 1111 1111 for "255", it can be interpreted as -1.
    // We must make it 0 1111 1111 for it to be properly read as 255.
    if (result.isNegative())
      result.zext(result.getBitWidth() + 1);
    assert(!result.isNegative() &&
           "Value still negative even after zero-extension");

    // Now we can safely negate
    if (isNegative) {
      result.negate();
      assert(result.isNegative() || result.isNullValue());
    }

    // Truncate, so we only use the minimum amount of bits needed to accurately
    // represent this value.
    unsigned neededBits = result.getMinSignedBits();
    if (result.getBitWidth() > neededBits)
      result = result.trunc(neededBits);

    return finish(Status::Ok);
  }

  // For fixed-width/pointer-sized integers, we need to be careful about
  // overflowing, so we have more processing to do.
  unsigned maxWidth = getMaxWidth();

  bool overflowed = (result.getActiveBits() > maxWidth);
  result = result.zextOrTrunc(maxWidth);

  Status s = overflowed ? Status::Overflow : Status::Ok;

  if (isNegative) {
    result.negate();
    if (!result.isNegative())
      s = Status::Error;
  }

  assert(result.getBitWidth() == maxWidth);

  return finish(s);
}