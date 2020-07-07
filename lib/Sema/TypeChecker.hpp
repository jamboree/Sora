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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace sora {
class ConstraintSystem;
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

  void typecheckDecl(Decl *decl);

  /// Entry point for typechecking "let" declarations used as conditions.
  void typecheckLetCondition(LetDecl *decl);

  /// Typechecking entry point for expressions used as conditions.
  /// \param expr the expression to typecheck
  /// \param dc th DeclContext in which this Expr lives. Cannot be null.
  /// \returns \p expr or the expr that should replace it in the tree.
  Expr *typecheckBooleanCondition(Expr *expr, DeclContext *dc);

  /// Typechecks the body of the functions in \c definedFunctions
  void typecheckDefinedFunctions();

  /// Typechecks the body of \p func. Does nothing if \p func doesn't have a
  /// body.
  void typecheckFunctionBody(FuncDecl *func);

  /// Expression typechecking entry point.
  /// \param cs the ConstraintSystem to use
  /// \param expr the expression to typecheck
  /// \param dc the DeclContext in which this Expr lives. Cannot be null.
  /// \param ofType (Optional) The expected type of the expression. If non-null,
  /// implicit conversions may be added to \p expr to convert it to \p ofType
  /// \param onUnificationFailure Called if the Expr's type can't unify with \p
  /// ofType. The first argument is the type of the Expr, the second is \p
  /// ofType.
  /// \returns \p expr or the expr that should replace it in the tree.
  Expr *typecheckExpr(
      ConstraintSystem &cs, Expr *expr, DeclContext *dc, Type ofType = Type(),
      llvm::function_ref<void(Type, Type)> onUnificationFailure = nullptr);

  /// Same as above, except this creates a fresh ConstraintSystem for the
  /// expression.
  Expr *typecheckExpr(
      Expr *expr, DeclContext *dc, Type ofType = Type(),
      llvm::function_ref<void(Type, Type)> onUnificationFailure = nullptr);

  void typecheckPattern(Pattern *pat, DeclContext *dc,
                        bool canEmitInferenceErrors = true);

  /// Type-checks a pattern and its initializer together.
  /// \returns \p init or the expr that should replace it in the tree.
  /// \param onUnificationFailure Called if the Expr's type can't
  /// unify with \p pat's type. The first argument is the type of \p init
  /// (simplified), the second is the type of \p pat (not simplified)
  Expr *typecheckPatternAndInitializer(
      Pattern *pat, Expr *init, DeclContext *dc,
      llvm::function_ref<void(Type, Type)> onUnificationFailure = nullptr);

  /// Statement typechecking entry point
  /// \param stmt the statement to typecheck
  /// \param dc th DeclContext in which this Stmt lives. Cannot be null.
  void typecheckStmt(Stmt *stmt, DeclContext *dc);

  /// Attempts to make cs.unify(toType, expr->getType()) work by adding implicit
  /// conversions around or in \p expr.
  /// \param ctxt the ASTContext
  /// \param toType the destination type - it can not contain LValues.
  /// \param expr the expression - it must not have LValues outside of
  ///        LoadExprs!
  /// \returns the expr that should take \p expr's place in the AST.
  Expr *tryCoerceExpr(ConstraintSystem &cs, Expr *expr, Type toType);

  /// Resolves a TypeLoc's type from its TypeRepr.
  /// \param tyLoc the TypeLoc to resolve (must have a TypeRepr* but no Type)
  /// \param file the file in which this TypeLoc lives
  void resolveTypeLoc(TypeLoc &tyLoc, SourceFile &file);

  /// Resolves a TypeRepr \p tyRepr
  /// \param tyRepr the TypeRepr to resolve
  /// \param file the file in which this TypeRepr lives
  /// \returns the type of \p tyRepr, or ErrorType on failure.
  Type resolveTypeRepr(TypeRepr *tyRepr, SourceFile &file);

  /// \returns true if \p from can be explicitly converted to \p to, or if \p
  /// from can unify with \p to.
  /// \param to must not contain an error type or
  /// type variable.
  /// \param from must not contain an error type.
  /// NOTE: This does NOT unify the types
  bool canExplicitlyCast(const ConstraintSystem &cs, Type from, Type to);

  /// \returns true if \p from can be implicitly converted to \p to, or if \p
  /// can unify with \p to.
  /// \param to must not contain an error type
  /// \param from must not contain an error type.
  /// NOTE: This does NOT unify the types
  bool canImplicitlyCast(const ConstraintSystem &cs, Type from, Type to);

  /// \returns whether we can emit a diagnostic involving \p type
  static bool canDiagnose(Type type);

  /// \returns whether we can emit a diagnostic involving \p expr
  static bool canDiagnose(Expr *expr);

  /// The list of non-local functions that have a body (=have been defined).
  /// See \p typecheckFunctionBodies()
  SmallVector<FuncDecl *, 4> definedFunctions;

  ASTContext &ctxt;
  DiagnosticEngine &diagEngine;
};

/// A small common base between AST "Checkers" (ExprChecker, DeclChecker,
/// etc.) that provides some basic functionalities.
class ASTCheckerBase {
public:
  ASTCheckerBase(TypeChecker &tc)
      : tc(tc), ctxt(tc.ctxt), diagEngine(tc.diagEngine) {}

  bool canDiagnose(Expr *expr) { return TypeChecker::canDiagnose(expr); }
  bool canDiagnose(Type type) { return TypeChecker::canDiagnose(type); }

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, const TypedDiag<Args...> &diag,
           typename detail::PassArgument<Args>::type... args) {
    assert(loc &&
           "TypeChecker can't emit diagnostics without valid SourceLocs");
    return diagEngine.diagnose<Args...>(loc, diag, args...);
  }

  TypeChecker &tc;
  ASTContext &ctxt;
  DiagnosticEngine &diagEngine;
};
} // namespace sora
