//===--- TypeCheckPattern.cpp -----------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//
//  Pattern Semantic Analysis
//===----------------------------------------------------------------------===//

#include "TypeChecker.hpp"

#include "ConstraintSystem.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Pattern.hpp"

using namespace sora;

//===- Helpers ------------------------------------------------------------===//

static void setTuplePatternType(ASTContext &ctxt, TuplePattern *pattern) {
  // The type of this pattern is an tuple of its element's types.
  assert(pattern->getNumElements() != 1 &&
         "Single Element Tuples Shouldn't Exist!");
  SmallVector<Type, 8> eltsTypes;
  for (Pattern *elt : pattern->getElements())
    eltsTypes.push_back(elt->getType());
  pattern->setType(TupleType::get(ctxt, eltsTypes));
}

static void setMaybeValuePatternType(MaybeValuePattern *pattern) {
  // The pattern has a "maybe T" type where T is the type of the subpattern.
  pattern->setType(MaybeType::get(pattern->getSubPattern()->getType()));
}

//===- PatternChecker -----------------------------------------------------===//

namespace {
/// This class handles the bulk of pattern type checking.
///
/// The walk is done in post-order (children are visited first).
///
/// Note that pattern type checking is quite similar to expression type checking
/// (it also uses type variables & a constraint system).
class PatternChecker : public ASTCheckerBase,
                       public ASTWalker,
                       public PatternVisitor<PatternChecker> {
public:
  /// The constraint system for this pattern
  ConstraintSystem &cs;
  /// The DeclContext in which this pattern appears
  DeclContext *dc;

  PatternChecker(TypeChecker &tc, ConstraintSystem &cs, DeclContext *dc)
      : ASTCheckerBase(tc), cs(cs), dc(dc) {}

  SourceFile &getSourceFile() const {
    assert(dc && "no DeclContext?");
    SourceFile *sf = dc->getParentSourceFile();
    assert(sf && "no Source File?");
    return *sf;
  }

  bool walkToPatternPost(Pattern *pattern) override {
    assert(pattern && "pattern is null");
    visit(pattern);
    assert(pattern->hasType() && "pattern is still untyped after visit");
    return true;
  }

  void visitVarPattern(VarPattern *pattern);
  void visitDiscardPattern(DiscardPattern *pattern);
  void visitTransparentPattern(TransparentPattern *pattern){
      /* no-op, type is automatically fetched from the subpattern */
  };
  void visitTuplePattern(TuplePattern *pattern);
  void visitTypedPattern(TypedPattern *pattern);
  void visitMaybeValuePattern(MaybeValuePattern *pattern);
};

void PatternChecker::visitVarPattern(VarPattern *pattern) {
  // Assign a type variable to this pattern so its type can be inferred.
  pattern->setType(cs.createGeneralTypeVariable());
}

void PatternChecker::visitDiscardPattern(DiscardPattern *pattern) {
  // Assign a type variable to this pattern so its type can be inferred.
  pattern->setType(cs.createGeneralTypeVariable());
}

void PatternChecker::visitTuplePattern(TuplePattern *pattern) {
  setTuplePatternType(ctxt, pattern);
}

void PatternChecker::visitTypedPattern(TypedPattern *pattern) {
  Type subPatType = pattern->getSubPattern()->getType();

  TypeLoc &typeLoc = pattern->getTypeLoc();
  tc.resolveTypeLoc(typeLoc, getSourceFile());

  // If the type couldn't be resolved successfully, bind every type variable to
  // the error type and stop.
  Type type = pattern->getType();
  if (type->hasErrorType()) {
    cs.bindAllToErrorType(subPatType);
    return;
  }

  // Else, unify. If unification succeeds, we're done.
  if (cs.unify(subPatType, type))
    return;

  // Diagnose the error, highlighting the subpattern & typerepr fully.
  diagnose(typeLoc.getLoc(), diag::cannot_convert_pattern, subPatType, type)
      .highlight(typeLoc.getSourceRange())
      .highlight(pattern->getSubPattern()->getSourceRange());

  // Bind everything to the error type so we don't complain about this again.
  cs.bindAllToErrorType(subPatType);
}

void PatternChecker::visitMaybeValuePattern(MaybeValuePattern *pattern) {
  setMaybeValuePatternType(pattern);
}

//===- PatternCheckerEpilogue ---------------------------------------------===//
// This class performs the epilogue of pattern type-checking. The visit methods
// will simplify the types of the patterns, or recompute them when needed. This
// will also trigger type-checking of VarDecls and set their types.
//
// Note that inference errors are only emitted for the Discard and Var patterns.
//===----------------------------------------------------------------------===//

class PatternCheckerEpilogue : public ASTCheckerBase,
                               public ASTWalker,
                               public PatternVisitor<PatternCheckerEpilogue> {
public:
  ConstraintSystem &cs;
  const bool canEmitInferenceErrors;

  PatternCheckerEpilogue(TypeChecker &tc, ConstraintSystem &cs,
                         bool canEmitInferenceErrors)
      : ASTCheckerBase(tc), cs(cs),
        canEmitInferenceErrors(canEmitInferenceErrors) {}

  /// Simplifies \p type, returning true on success and false when an inference
  /// error occured. The type is simplified in-place.
  bool simplifyPatternType(Type &type) {
    bool wasDiagnosable = canDiagnose(type);
    bool isAmbiguous = false;
    type = cs.simplify(type, &isAmbiguous);
    return !isAmbiguous || !(wasDiagnosable && canEmitInferenceErrors);
  }

  void diagnoseAmbiguousPatternType(Pattern *pattern, StringRef patternName) {
    diagnose(pattern->getLoc(),
             diag::type_of_pattern_is_ambiguous_without_more_ctxt, patternName);
    diagnose(pattern->getLoc(), diag::add_type_annot_to_give_pattern_a_type,
             patternName)
        .fixitInsertAfter(pattern->getEndLoc(), ": <type>");
  }

  void visitVarPattern(VarPattern *pattern) {
    Type type = pattern->getType();
    if (simplifyPatternType(type)) {
      pattern->setType(type);
      return;
    }

    diagnoseAmbiguousPatternType(
        pattern, pattern->getVarDecl()->getIdentifier().c_str());
    pattern->setType(ctxt.errorType);
  }
  void visitDiscardPattern(DiscardPattern *pattern) {
    Type type = pattern->getType();
    if (simplifyPatternType(type)) {
      pattern->setType(type);
      return;
    }

    diagnoseAmbiguousPatternType(pattern, "_");
    pattern->setType(ctxt.errorType);
  }

  void visitTuplePattern(TuplePattern *pattern) {
    setTuplePatternType(ctxt, pattern);
  }

  void visitTransparentPattern(TransparentPattern *pattern){
      /* no-op, type will be automatically updated when the subpattern is */
  };

  void visitTypedPattern(TypedPattern *pattern) {
    llvm_unreachable(
        "TypedPatterns should never have TypeVariables as their type!");
  }

  void visitMaybeValuePattern(MaybeValuePattern *pattern) {
    setMaybeValuePatternType(pattern);
  }

  bool walkToPatternPost(Pattern *pattern) override {
    if (pattern->getType()->hasTypeVariable())
      visit(pattern);

    if (VarPattern *varPat = dyn_cast<VarPattern>(pattern)) {
      VarDecl *varDecl = varPat->getVarDecl();
      // Set the type of the VarDecls inside VarPatterns.
      varDecl->setValueType(varPat->getType());
      // Check the VarDecl
      tc.typecheckDecl(varDecl);
    }

    return true;
  }
};

} // namespace

//===- TypeChecker --------------------------------------------------------===//

void TypeChecker::typecheckPattern(Pattern *pat, DeclContext *dc,
                                   bool canEmitInferenceErrors) {
  assert(pat && dc);
  // Create a constraint system for this pattern
  ConstraintSystem cs(*this);

  // Check the pattern
  pat->walk(PatternChecker(*this, cs, dc));

  // Perform the epilogue
  pat->walk(PatternCheckerEpilogue(*this, cs, canEmitInferenceErrors));
}

Expr *TypeChecker::typecheckPatternAndInitializer(
    Pattern *pat, Expr *init, DeclContext *dc,
    llvm::function_ref<void(Type, Type)> onUnificationFailure) {
  assert(pat && init && dc);
  // Create a constraint system for this pattern
  ConstraintSystem cs(*this);

  // Check the pattern
  pat->walk(PatternChecker(*this, cs, dc));

  // Fully check the expr, unifying it with the type of the pattern
  bool emitInferenceErrors = true;
  init = typecheckExpr(cs, init, dc, pat->getType(), [&](Type from, Type to) {
    if (onUnificationFailure)
      onUnificationFailure(from, to);
    // Don't emit inference errors in the pattern
    emitInferenceErrors = false;
  });

  // Perform the epilogue
  pat->walk(PatternCheckerEpilogue(*this, cs, emitInferenceErrors));

  return init;
}