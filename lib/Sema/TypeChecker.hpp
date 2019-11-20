//===--- TypeChecker.hpp - Sora Language Semantic Analysis ------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Diagnostics/DiagnosticsSema.hpp"
#include "llvm/ADT/SmallVector.h"

namespace sora {
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

/// The Sora type-checker, which performs semantic analysis of the Sora AST.
class TypeChecker final {
  TypeChecker(const TypeChecker &) = delete;
  TypeChecker &operator=(const TypeChecker &) = delete;

public:
  TypeChecker(ASTContext &ctxt);

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
           typename detail::PassArgument<Args>::type... args) {
    assert(loc &&
           "TypeChecker can't emit diagnostics without valid SourceLocs");
    return diagEngine.diagnose<Args...>(loc, diag, args...);
  }

  /// Checks if a ValueDecl is a legal declaration, and not an invalid declaration.
  /// \verbatim
  ///   func foo() {} // this foo is valid, it's the first one
  ///   func foo() {} // this one isn't
  /// \endverbatim
  void checkForRedeclaration(ValueDecl *decl);

  /// Declaration typechecking entry point
  ///
  /// NOTE: This doesn't check
  void typecheckDecl(Decl *decl);

  /// Typechecks the body of the functions in \c definedFunctions
  void typecheckDefinedFunctions();

  /// Typechecks the body of \p func. Does nothing if \p func doesn't have a
  /// body.
  void typecheckFunctionBody(FuncDecl *func);

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
  /// \param tyLoc the TypeLoc to resolve (must have a TypeRepr* but no Type)
  /// \param file the file in which this TypeLoc lives
  void resolveTypeLoc(TypeLoc &tyLoc, SourceFile &file);

  /// The list of non-local functions that have a body (=have been defined).
  /// See \p typecheckFunctionBodies()
  SmallVector<FuncDecl *, 4> definedFunctions;

  /// The set of declarations that are ignored during lookup
  SmallVector<ValueDecl *, 8> ignoredDecls;

  ASTContext &ctxt;
  DiagnosticEngine &diagEngine;
};
} // namespace sora
