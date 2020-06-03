//===--- SIRGenPattern.cpp - Pattern SIR Generation -------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "SIRGen.hpp"

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Pattern.hpp"

using namespace sora;

//===- PatternGenerator ---------------------------------------------------===//

namespace {
class PatternGenerator
    : public SIRGeneratorBase,
      public PatternVisitor<PatternGenerator, void, mlir::Value> {
public:
  PatternGenerator(SIRGen &sirGen, mlir::OpBuilder &builder)
      : SIRGeneratorBase(sirGen), builder(builder) {}

  mlir::OpBuilder &builder;

  void visitVarPattern(VarPattern *pattern, mlir::Value value);
  void visitDiscardPattern(DiscardPattern *pattern, mlir::Value value);
  void visitMutPattern(MutPattern *pattern, mlir::Value value);
  void visitParenPattern(ParenPattern *pattern, mlir::Value value);
  void visitTuplePattern(TuplePattern *pattern, mlir::Value value);
  void visitTypedPattern(TypedPattern *pattern, mlir::Value value);
  void visitMaybeValuePattern(MaybeValuePattern *pattern, mlir::Value value);
};

void PatternGenerator::visitVarPattern(VarPattern *pattern, mlir::Value value) {
  // Generate the VarDecl using 'value' as the initial value.
  // If the value is null, this will generate a default initializer.
  sirGen.genVarDecl(builder, pattern->getVarDecl(), value);
}

void PatternGenerator::visitDiscardPattern(DiscardPattern *, mlir::Value) {
  // No-op, it just discards the value as the name implies.
}

void PatternGenerator::visitMutPattern(MutPattern *pattern, mlir::Value value) {
  // This pattern is "transparent". Just emit its subpattern.
  visit(pattern->getSubPattern(), value);
}

void PatternGenerator::visitParenPattern(ParenPattern *pattern,
                                         mlir::Value value) {
  // This pattern is "transparent". Just emit its subpattern.
  visit(pattern->getSubPattern(), value);
}

void PatternGenerator::visitTuplePattern(TuplePattern *pattern,
                                         mlir::Value value) {
  // If this is an empty pattern, don't do anything, the value is just
  // discarded.
  if (pattern->isEmpty())
    return;

  // If we don't have an initial value, then this is easy: just generate the
  // elements.
  if (!value) {
    for (Pattern *elt : pattern->getElements())
      visit(elt, {});
    return;
  }

  // If we have an initial value, generate a destructure_tuple for it, and
  // generate each element of the TuplePattern with the corresponding tuple
  // value as initial value.
  assert(value.getType().isa<mlir::TupleType>() && "Value is not a tuple!");

  sir::DestructureTupleOp destructuredTuple =
      builder.create<sir::DestructureTupleOp>(getNodeLoc(pattern), value);

  mlir::ResultRange destructuredTupleValues = destructuredTuple.getResults();
  ArrayRef<Pattern *> patternElts = pattern->getElements();

  assert(destructuredTupleValues.size() == patternElts.size() &&
         "Illegal structure!");

  for (size_t k = 0; k < patternElts.size(); ++k)
    visit(patternElts[k], destructuredTupleValues[k]);
}

void PatternGenerator::visitTypedPattern(TypedPattern *pattern,
                                         mlir::Value value) {
  // This pattern is "transparent". Just emit its subpattern.
  visit(pattern->getSubPattern(), value);
}

void PatternGenerator::visitMaybeValuePattern(MaybeValuePattern *pattern,
                                              mlir::Value value) {
  // These should only be present at the top level a LetDecl's pattern when it
  // is used as a condition, and need special handling.
  llvm_unreachable("Generation of MaybeValuePatterns through the "
                   "PatternGenerator is not supported!");
}

} // namespace

//===- SIRGen -------------------------------------------------------------===//

void SIRGen::genPattern(mlir::OpBuilder &builder, Pattern *pattern,
                        mlir::Value value) {
  PatternGenerator(*this, builder).visit(pattern, value);
}

mlir::Location SIRGen::getNodeLoc(Pattern *pattern) {
  return mlir::OpaqueLoc::get(pattern, getFileLineColLoc(pattern->getLoc()));
}