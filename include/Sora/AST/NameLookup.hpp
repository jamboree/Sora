//===--- NameLookup.hpp - AST Name Lookup Entry Points ----------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Identifier.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceLoc.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class SourceFile;
class SourceLoc;
class ValueDecl;

/// Class used to configure and execute an unqualified lookup inside a source
/// file. This class also represents the result of the lookup!
///
/// An important thing to note is that name lookup DOES NOT look after built-in
/// types. It just cares about the user-defined names.
/// FIXME: Should it also care about built-in types? Maybe.
///
/// FIXME: Currently, this class doesn't offer much. In the future, it should
/// contain helpers to configure the lookup, e.g. to only see types, only see
/// local things, etc.
class UnqualifiedLookup final {
  void lookupImpl(SourceLoc loc, Identifier ident);

  llvm::DenseSet<ValueDecl *> ignoredDecls;

  /// Adds \p decls to the list of results.
  /// \returns true if at least one result was added.
  bool addResults(ArrayRef<ValueDecl *> decls) {
    bool added = false;
    for (ValueDecl *decl : decls) {
      if (ignoredDecls.find(decl) == ignoredDecls.end()) {
        results.push_back(decl);
        added = true;
      }
    }
    return added;
  }

public:
  UnqualifiedLookup(SourceFile &sourceFile) : sourceFile(sourceFile) {}

  /// Adds \p decls to the list of decls that should be ignored during the
  /// lookup.
  UnqualifiedLookup ignore(ArrayRef<ValueDecl *> decls) {
    ignoredDecls.insert(decls.begin(), decls.end());
    return *this;
  }

  /// Lookup for decls with name \p ident in \p loc
  void performLookup(SourceLoc loc, Identifier ident) {
    assert(ident && "identifier is invalid!");
    lookupImpl(loc, ident);
  }

  /// Finds every decl visible at \p loc
  void findDeclsAt(SourceLoc loc) { lookupImpl(loc, Identifier()); }

  /// \returns whether the set of results is empty
  bool isEmpty() const { return results.empty(); }
  /// \returns whether the set of results contains a single result
  bool isUnique() const { return results.size() == 1; }
  /// \returns the single result of the lookup, or nullptr if there are 0 or 2+
  /// results.
  ValueDecl *getUniqueResult() const {
    return (results.size() == 1) ? results[0] : nullptr;
  }

  /// The SourceFile in which we are looking
  SourceFile &sourceFile;
  /// The list of results
  SmallVector<ValueDecl *, 8> results;
};
} // namespace sora