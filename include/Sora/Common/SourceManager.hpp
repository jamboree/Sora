//===--- SourceManager.hpp - Source Files Management ------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace sora {

/// Wrapper around an unsigned integer, used to represent a Buffer ID.
class BufferID {
  unsigned value = 0;

  friend class SourceManager;
  BufferID(unsigned value) : value(value) {}

public:
  /// Creates a null BufferID
  BufferID() = default;

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
};

} // namespace sora