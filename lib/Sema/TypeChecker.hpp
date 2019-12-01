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

  /// Performs expression checking on \p expr using the ConstraintSystem \p cs.
  /// \returns \p expr or the expression that should replace \p expr in the
  /// tree. Never nullptr. 
  ///
  /// After expression checking finishes, \c
  /// performExprCheckingEpilogue must be called to replace type variables with
  /// their substitutions & diagnose inference errors.
  ///
  /// Note that if the type of the expression contains unbound type variables,
  /// you can use unify(expr->getType, someType) to bind it to something before
  /// calling \c performExprCheckingEpilogue
  Expr *performExprChecking(ConstraintSystem &cs, Expr *expr, DeclContext *dc);

  /// Performs expression checking epilogue on \p expr using the
  /// ConstraintSystem \p cs. This will replace type variables with their
  /// substitutions & diagnose inference errors.
  /// \returns \p expr or the expression that should replace \p expr in the
  /// tree. Never nullptr.
  Expr *performExprCheckingEpilogue(ConstraintSystem &cs, Expr *expr);

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

  /// Checks if a single ValueDecl is a legal declaration, and not an invalid
  /// redeclaration.
  /// This sets the decl's 'isIllegalRedeclaration' flag.
  /// This will not check the decl if it's already marked as being an illegal
  /// redeclaration.
  /// \verbatim
  ///   func foo() {} // this foo is valid, it's the first one
  ///   func foo() {} // this one isn't
  /// \endverbatim
  void checkIsIllegalRedeclaration(ValueDecl *decl);

  /// Checks that a declaration list doesn't bind an identifier more than once.
  ///
  /// First, this checks that an identifier isn't bound more than once in
  /// \p decls. Else, \p noteFirst is called for the first binding, and
  /// \p diagnoseDuplicateBinding for each duplicate binding.
  ///
  /// This also sets the 'isIllegalRedeclaration' flag on each duplicate
  /// binding.
  ///
  /// Finally, note that diagnoseDuplicateBinding/noteFirst are called in
  /// groups. For example, in 'let (a, b, a, b)', they're both called for 'a'
  /// first and then for 'b' (or for 'b' then for 'a' - that ordering isn't
  /// guaranteed)
  static void checkForDuplicateBindingsInList(
      ArrayRef<ValueDecl *> decls,
      llvm::function_ref<void(ValueDecl *)> diagnoseDuplicateBinding,
      llvm::function_ref<void(ValueDecl *)> noteFirstBinding);

  /// Declaration typechecking entry point
  void typecheckDecl(Decl *decl);

  /// Typechecks the body of the functions in \c definedFunctions
  void typecheckDefinedFunctions();

  /// Typechecks the body of \p func. Does nothing if \p func doesn't have a
  /// body.
  void typecheckFunctionBody(FuncDecl *func);

  /// Expression typechecking entry point.
  /// This will create a ConstraintSystem for the expression.
  /// \param expr the expression to typecheck
  /// \param dc the DeclContext in which this Expr lives. Cannot be null.
  /// \param ofType If valid, the expected type of the expression.
  /// \returns \p expr or the expr that should replace it in the tree.
  Expr *typecheckExpr(Expr *expr, DeclContext *dc, Type ofType = Type());

  /// Typechecking entry point for expressions used as conditions.
  /// \param expr the expression to typecheck
  /// \param dc th DeclContext in which this Expr lives. Cannot be null.
  /// \returns \p expr or the expr that should replace it in the tree.
  Expr *typecheckCondition(Expr *expr, DeclContext *dc) {
    /// FIXME: Handle these differently
    return typecheckExpr(expr, dc);
  }

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

  ASTContext &ctxt;
  DiagnosticEngine &diagEngine;
};

/// A small common base between AST "Checkers" (ExprChecker, DeclChecker,
/// etc.) that provides some basic functionalities.
class ASTChecker {
public:
  ASTChecker(TypeChecker &tc)
      : tc(tc), ctxt(tc.ctxt), diagEngine(tc.diagEngine) {}

  /// Emits a diagnostic at \p loc
  template <typename... Args>
  InFlightDiagnostic
  diagnose(SourceLoc loc, TypedDiag<Args...> diag,
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
