//===--- Source.cpp ---------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
// Implementation of the SourceLoc, SourceRange, CharSourceRange and
// SourceManager classes.
//===----------------------------------------------------------------------===//

#include "Sora/Common/SourceLoc.hpp"
#include "Sora/Common/SourceManager.hpp"

using namespace sora;

size_t SourceManager::getDistanceInBytes(SourceLoc beg, SourceLoc end) {
  assert(beg && end && "invalid locs!");
  const char *begPtr = beg.getPointer();
  const char *endPtr = end.getPointer();
#ifndef NDEBUG
  unsigned bufferID = llvmSourceMgr.FindBufferContainingLoc(beg.value);
  assert(bufferID && "SourceLoc doesn't belong in any buffer!");
  auto buff = llvmSourceMgr.getMemoryBuffer(bufferID);
  assert((endPtr >= buff->getBufferStart()) &&
         (endPtr <= buff->getBufferEnd()) &&
         "beg & end aren't from the same buffer!");
#endif
  assert((endPtr >= begPtr) && "beg > end!");
  return endPtr - begPtr;
}

BufferID SourceManager::giveBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer) {
  unsigned bufferID =
      llvmSourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  return BufferID(bufferID);
}

StringRef SourceManager::getBufferStr(BufferID id) const {
  assert(id && "invalid buffer id");
  return llvmSourceMgr.getMemoryBuffer(id.value)->getBuffer();
}

CharSourceRange::CharSourceRange(SourceManager &srcMgr, SourceLoc begin,
                                 SourceLoc end)
    : begin(begin) {
  byteLength = srcMgr.getDistanceInBytes(begin, end);
}

StringRef CharSourceRange::str() const {
  return StringRef(begin.value.getPointer(), byteLength);
}