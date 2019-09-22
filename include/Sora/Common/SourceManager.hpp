//===--- SourceManager.hpp - Source Files Management ------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/SourceMgr.h"
#include <memory>
#include <utility>

namespace sora {

/// Wrapper around an unsigned integer, used to represent a Buffer ID.
class BufferID {
  unsigned value = 0;

  friend class SourceManager;
  BufferID(unsigned value) : value(value) {}

public:
  /// Creates a null BufferID
  BufferID() = default;

  /// \returns the raw value of the BufferID;
  unsigned getRawValue() const { return value; }

  /// \returns true if this is a null buffer id
  bool isNull() { return value == 0; }
  /// \returns true if this buffer id isn't null.
  explicit operator bool() { return !isNull(); }
};

/// Manages & owns source buffers
class SourceManager {
public:
  SourceManager() = default;

  // The SourceManager is non-copyable.
  SourceManager(const SourceManager &) = delete;
  SourceManager &operator=(const SourceManager &) = delete;

  /// The underlying LLVM Source Manager, which owns the source buffers.
  /// Use it wisely. Use SourceManager's method whenever possible instead
  /// of using it.
  llvm::SourceMgr llvmSourceMgr;

  /// \returns the distance in bytes between \p beg and \p end
  size_t getDistanceInBytes(SourceLoc beg, SourceLoc end);

  /// Gives a buffer to this SourceManager, returning a BufferID for that
  /// buffer and taking ownership of it.
  BufferID giveBuffer(std::unique_ptr<llvm::MemoryBuffer> buffer);

  /// \returns the string of the buffer with id \p id
  StringRef getBufferStr(BufferID id) const;

  /// \returns the BufferID of the buffer that contains \p loc
  BufferID findBufferContainingLoc(SourceLoc loc) const {
    return llvmSourceMgr.FindBufferContainingLoc(loc.value);
  }

  /// \returns the line and column represented by \p loc.
  /// If \p id is valid, \p loc must come from that source buffer.
  std::pair<unsigned, unsigned>
  getLineAndColumn(SourceLoc loc, BufferID id = BufferID()) const {
    assert(loc && "loc cannot be invalid");
    return llvmSourceMgr.getLineAndColumn(loc.value, id.value);
  }

  /// \returns the identifier of the MemoryBuffer with \p id
  StringRef getBufferIdentifier(BufferID id) const {
    assert(id && "id cannot be invalid!");
    return llvmSourceMgr.getMemoryBuffer(id.value)->getBufferIdentifier();
  }
};

} // namespace sora