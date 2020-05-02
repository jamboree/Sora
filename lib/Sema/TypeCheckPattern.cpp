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
  void visitMutPattern(MutPattern *pattern);
  void visitParenPattern(ParenPattern *pattern);
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

void PatternChecker::visitMutPattern(MutPattern *pattern) {
  // This pattern is transparent: its type is simply the type of its subpattern.
  pattern->setType(pattern->getSubPattern()->getType());
}

void PatternChecker::visitParenPattern(ParenPattern *pattern) {
  // This pattern is transparent: its type is simply the type of its subpattern.
  pattern->setType(pattern->getSubPattern()->getType());
}

void PatternChecker::visitTuplePattern(TuplePattern *pattern) {
  // The type of this pattern is an tuple of its element's types.
  assert(pattern->getNumElements() != 1 &&
         "Single Element Tuples Shouldn't Exist!");
  SmallVector<Type, 8> eltsTypes;
  for (Pattern *elt : pattern->getElements())
    eltsTypes.push_back(elt->getType());
  pattern->setType(TupleType::get(ctxt, eltsTypes));
}

void PatternChecker::visitTypedPattern(TypedPattern *pattern) {
  Type subPatType = pattern->getSubPattern()->getType();

  Type type = tc.resolveTypeRepr(pattern->getTypeRepr(), getSourceFile());
  pattern->setType(type);

  // If the type couldn't be resolved successfully, bind every type variable to
  // the error type and stop.
  if (type->hasErrorType()) {
    cs.bindAllToErrorType(subPatType);
    return;
  }

  // Else, unify. If unification succeeds, we're done.
  if (cs.unify(subPatType, type))
    return;

  // Diagnose the error, highlighting the subpattern & typerepr fully.
  diagnose(pattern->getTypeRepr()->getLoc(), diag::cannot_convert_pattern,
           subPatType, type)
      .highlight(pattern->getTypeRepr()->getSourceRange())
      .highlight(pattern->getSubPattern()->getSourceRange());

  // Bind everything to the error type so we don't complain about this again.
  cs.bindAllToErrorType(subPatType);
}

void PatternChecker::visitMaybeValuePattern(MaybeValuePattern *pattern) {
  // The pattern has a "maybe T" type where T is the type of the subpattern.
  pattern->setType(MaybeType::get(pattern->getSubPattern()->getType()));
}

//===- PatternCheckerEpilogue ---------------------------------------------===//

class PatternCheckerEpilogue : public ASTCheckerBase, public ASTWalker {
public:
  ConstraintSystem &cs;
  const bool canEmitInferenceErrors;

  PatternCheckerEpilogue(TypeChecker &tc, ConstraintSystem &cs,
                         bool canEmitInferenceErrors)
      : ASTCheckerBase(tc), cs(cs),
        canEmitInferenceErrors(canEmitInferenceErrors) {}

  void simplifyTypeOfPattern(Pattern *pattern) {
    Type type = pattern->getType();
    assert(type && "untyped pattern");
    if (!type->hasTypeVariable())
      return;

    bool wasDiagnosable = canDiagnose(type);
    bool isAmbiguous = false;
    type = cs.simplifyType(type, &isAmbiguous);
    pattern->setType(type);

    // Return if there was no error, if we can't diagnose this.
    if (!isAmbiguous || !(wasDiagnosable && canEmitInferenceErrors))
      return;

    // We only complain about inference errors on VarPattern & DiscardPatterns.
    if (!isa<VarPattern>(pattern) && !isa<DiscardPattern>(pattern))
      return;

    StringRef patternName;
    if (isa<DiscardPattern>(pattern))
      patternName = "_";
    else
      patternName = cast<VarPattern>(pattern)->getIdentifier().c_str();

    diagnose(pattern->getLoc(),
             diag::type_of_pattern_is_ambiguous_without_more_ctxt, patternName);
    diagnose(pattern->getLoc(), diag::add_type_annot_to_give_pattern_a_type,
             patternName)
        .fixitInsertAfter(pattern->getEndLoc(), ": <type>");
  }

  bool walkToPatternPost(Pattern *pattern) override {
    simplifyTypeOfPattern(pattern);

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