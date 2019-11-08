//===--- Sema.hpp - Sora Language Semantic Analysis -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Diagnostics/DiagnosticEngine.hpp"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace sora {
class ASTContext;
class Decl;
class DeclContext;
class Expr;
class FuncDecl;
class Pattern;
class Stmt;
class SourceFile;
class TypeLoc;
class TypeRepr;
class ValueDecl;

/// Main interface of Sora's semantic analyzer.
class Sema final {
  Sema(const Sema &) = delete;
  Sema &operator=(const Sema &) = delete;

  /// Entry point to typecheck function bodies
  void typecheckFunctionBodies() const;

  /// Declaration typechecking entry point
  ///
  /// NOTE: This doesn't check
  void typecheckDecl(Decl *decl);

  /// Typechecks the body of the functions that have been defined.
  void typecheckFunctionBodies();

  /// Expression typechecking entry point
  /// \param expr the expression to typecheck
  /// \param dc th DeclContext in which this Expr lives. Cannot be null.
  /// \param ofType If valid, the expected type of the expression.
  Expr *typecheckExpr(Expr *expr, DeclContext *dc, Type ofType = Type());

  /// Statement typechecking entry point
  /// \param stmt the statement to typecheck
  /// \param dc th DeclContext in which this Stmt lives. Cannot be null.
  void typecheckStmt(Stmt *stmt, DeclContext *dc);

  /// Pattern typechecking entry point
  void typecheckPattern(Pattern *pat);

  /// Resolves a TypeLoc, giving it a type from its TypeRepr.
  /// This can only be used if the TypeLoc has a TypeRepr.
  void resolveTypeLoc(TypeLoc &tyLoc);

  // Implementation classes
  class DeclChecker;
  class StmtChecker;

  /// The list of functions that have been defined.
  /// This is kept here because typecheckDecl doesn't check the body of the
  /// function - it's checked later. See \p typecheckFunctionBodies()
  std::vector<FuncDecl *> definedFunctions;

  /// The set of declarations that are ignored during lookup
  SmallVector<ValueDecl *, 8> ignoredDecls;

public:
  Sema(ASTContext &ctxt);

  /// Main entry point of semantic analysis: performs semantic analysis on \p
  /// file.
  void performSema(SourceFile &file);

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    assert(loc && "Sema can't emit diagnostics without valid SourceLocs");
    return diagEngine.diagnose<Args...>(loc, diag, args...);
  }

  ASTContext &ctxt;
  DiagnosticEngine &diagEngine;
};
} // namespace sora
