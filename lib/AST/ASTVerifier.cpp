//===--- ASTVerifier.cpp - AST Invariants Verifier --------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/ASTWalker.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/Types.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/EntryPoints.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

//===- ASTVerifier Impl ---------------------------------------------------===//
//
// There are 2 verifier implementations:
//  - The BasicVerifierImpl, which checks parsed ASTs.
//  - The CheckedVerifierImpl, which checks type-checked ASTs.
//
// The ASTVerifier always calls the BasicVerifier, but it only calls the
// CheckedVerifier when requested to do so.
//
// Please note that the ASTVerifier is pretty much a constant work in progress -
// things are added to it when needed.

namespace {
class ASTVerifier;

using VerifiedNode =
    llvm::PointerUnion<Decl *, Expr *, Pattern *, Stmt *, TypeRepr *>;

class ASTVerifierImplBase {
public:
  ASTVerifier &verifier;

  ASTVerifierImplBase(ASTVerifier &verifier) : verifier(verifier) {}

  SmallVectorImpl<VerifiedNode> &getNodeStack();

  /// \returns the last verified node
  VerifiedNode &getLastVerifiedNode() { return getNodeStack().back(); }

  /// \returns the top-level node of some kind by walking the stack of visited
  /// nodes, and stopping and return the last node of type Ty. Returns nullptr
  /// if the last verified node isn't of type Ty.
  template <typename Ty> Ty *getTopLevelNode() {
    SmallVectorImpl<VerifiedNode> &nodeStack = getNodeStack();
    Ty *last = nullptr;
    for (auto it = nodeStack.rbegin(); it != nodeStack.rend(); ++it) {
      Ty *cur = it->dyn_cast<Ty *>();
      if (!cur)
        break;
      last = cur;
    }
    return last;
  }
};

/// RAII object to report AST Verification failures, which exits the compiler
/// upon destruction.
///
/// VerifierImpls should use the << operator on this object to print additional
/// information. They don't need to dump the node that has failed verification -
/// this class will handle it.
class ASTVerificationFailure {
  void dump(VerifiedNode node) const {
    if (Decl *decl = node.dyn_cast<Decl *>())
      return decl->dump(llvm::dbgs());
    if (Expr *expr = node.dyn_cast<Expr *>())
      return expr->dump(llvm::dbgs(), &srcMgr);
    if (Pattern *pattern = node.dyn_cast<Pattern *>())
      return pattern->dump(llvm::dbgs(), &srcMgr);
    if (Stmt *stmt = node.dyn_cast<Stmt *>())
      return stmt->dump(llvm::dbgs(), &srcMgr);
    if (TypeRepr *tyRepr = node.dyn_cast<TypeRepr *>())
      return tyRepr->dump(llvm::dbgs(), &srcMgr);
    llvm_unreachable("Unknown VerifiedNode kind");
  }

  SourceLoc getLoc(VerifiedNode node) const {
    if (Decl *decl = node.dyn_cast<Decl *>())
      return decl->getLoc();
    if (Expr *expr = node.dyn_cast<Expr *>())
      return expr->getLoc();
    if (Pattern *pattern = node.dyn_cast<Pattern *>())
      return pattern->getLoc();
    if (Stmt *stmt = node.dyn_cast<Stmt *>())
      return stmt->getLoc();
    if (TypeRepr *tyRepr = node.dyn_cast<TypeRepr *>())
      return tyRepr->getLoc();
    llvm_unreachable("Unknown VerifiedNode kind");
  }

public:
  ASTVerificationFailure(ASTVerifierImplBase &verifierImpl, VerifiedNode node,
                         unsigned line);

  // Prints something to llvm::dbgs().
  template <typename Ty> ASTVerificationFailure &operator<<(Ty &&val) {
    llvm::dbgs() << std::forward<Ty>(val);
    return *this;
  }

  ~ASTVerificationFailure();

  ASTVerifierImplBase &verifierImpl;
  ASTVerifier &verifier;
  const SourceManager &srcMgr;
  VerifiedNode node;
};

#define CREATE_FAILURE(NODE) ASTVerificationFailure(*this, NODE, __LINE__)

/// Verifies parsed ASTs. This shouldn't rely on any checking done by the
/// Semantic Analyzer (Sema).
class BasicVerifierImpl : public ASTVerifierImplBase,
                          public SimpleASTVisitor<BasicVerifierImpl> {
public:
  BasicVerifierImpl(ASTVerifier &verifier) : ASTVerifierImplBase(verifier) {}

  void visitExpr(Expr *) {}

  void visitDecl(Decl *) {}

  void visitMaybeValuePattern(MaybeValuePattern *pattern);
  void visitPattern(Pattern *) {}

  void visitTypeRepr(TypeRepr *) {}

  void visitStmt(Stmt *) {}
};

/// Verifies type-checked ASTs. This checks complete ASTs, and can rely on
/// checking done by Sema.
class CheckedVerifierImpl : public ASTVerifierImplBase,
                            public SimpleASTVisitor<CheckedVerifierImpl> {
  using Base = SimpleASTVisitor<CheckedVerifierImpl>;

public:
  CheckedVerifierImpl(ASTVerifier &verifier) : ASTVerifierImplBase(verifier) {}

  //===- Entry Points -----------------------------------------------------===//

  void visit(Decl *decl) {
    if (ValueDecl *vd = dyn_cast<ValueDecl>(decl))
      checkValueDeclCommon(vd);
    Base::visit(decl);
  }

  void visit(Expr *expr) {
    checkExprCommon(expr);
    Base::visit(expr);
  }

  void visit(Pattern *pattern) {
    checkPatternCommon(pattern);
    Base::visit(pattern);
  }

  void visit(Stmt *stmt) { Base::visit(stmt); }

  void visit(TypeRepr *tyRepr) { Base::visit(tyRepr); }

  //===- Visit Methods ----------------------------------------------------===//

  void visitDiscardExpr(DiscardExpr *expr);
  void visitDeclRefExpr(DeclRefExpr *expr);
  void visitTupleElementExpr(TupleElementExpr *expr);
  void visitLoadExpr(LoadExpr *expr);
  void visitUnaryExpr(UnaryExpr *expr);
  void visitExpr(Expr *) {}

  void visitDecl(Decl *) {}

  void visitMaybeValuePattern(MaybeValuePattern *pattern);
  void visitPattern(Pattern *) {}

  void visitTypeRepr(TypeRepr *) {}

  void visitStmt(Stmt *) {}

  //===- Common Checks ----------------------------------------------------===//

  void checkPatternCommon(Pattern *pattern) {
    Type type = pattern->getType();
    if (type.isNull())
      CREATE_FAILURE(pattern) << "Pattern does not have a type!";
    if (type->hasLValue())
      CREATE_FAILURE(pattern) << "Pattern type cannot contain LValues!";
    if (type->hasTypeVariable())
      CREATE_FAILURE(pattern) << "Pattern type cannot contain Type Variables!";
    if (type->hasErrorType())
      CREATE_FAILURE(pattern) << "Pattern type cannot contain Error Types!";
  }

  void checkExprCommon(Expr *expr) {
    Type type = expr->getType();
    if (type.isNull())
      CREATE_FAILURE(expr) << "Expression does not have a type!";
    if (type->hasTypeVariable())
      CREATE_FAILURE(expr) << "Expression type cannot contain Type Variables!";
    if (type->hasErrorType())
      CREATE_FAILURE(expr) << "Expression type cannot contain Error Types!";

    if (type->is<LValueType>()) {
      ExprKind kind = expr->getKind();

      auto isOk = [&] {
        if (kind == ExprKind::DeclRef || kind == ExprKind::Discard ||
            kind == ExprKind::TupleElement || kind == ExprKind::Paren)
          return true;

        if (UnaryExpr *unary = dyn_cast<UnaryExpr>(expr))
          return unary->getOpKind() == UnaryOperatorKind::Deref;
        return false;
      };

      // Only a few expressions should have LValue types. All other operations
      // should load their operands and not produce LValues.
      if (!isOk()) {
        CREATE_FAILURE(expr) << "Expression should not have an LValue type!";
      }
    }
  }

  void checkValueDeclCommon(ValueDecl *decl) {
    Type valueType = decl->getValueType();
    if (valueType.isNull())
      CREATE_FAILURE(decl) << "ValueDecl does not have a type!";
    if (valueType->hasLValue())
      CREATE_FAILURE(decl) << "ValueDecl type cannot contain LValues!";
    if (valueType->hasTypeVariable())
      CREATE_FAILURE(decl) << "ValueDecl type cannot contain Type Variables!";
    if (valueType->hasErrorType())
      CREATE_FAILURE(decl) << "ValueDecl type cannot contain Error Types!";
  }
};

/// The ASTVerifier implementation, which walks the AST in pre-order and calls
/// the verifiers.
/// This also keeps track of the "stack" of visited nodes. Nodes are added to
/// the stack after they're verified, and removed from the stack after visiting
/// their children.
class ASTVerifier : public ASTWalker {
public:
  ASTVerifier(ASTContext &ctxt, bool isChecked)
      : ctxt(ctxt), BasicVerifierImpl(*this) {
    if (isChecked)
      checkedImpl.emplace(*this);
  }

  ASTContext &ctxt;
  BasicVerifierImpl BasicVerifierImpl;
  Optional<CheckedVerifierImpl> checkedImpl;

  SmallVector<VerifiedNode, 16> nodeStack;

  bool isCheckedAST() const { return (bool)checkedImpl; }

  template <typename Ty> void visit(Ty val) {
    BasicVerifierImpl.visit(std::forward<Ty>(val));
    if (checkedImpl)
      checkedImpl->visit(std::forward<Ty>(val));
  }

  template <typename Ty> void pushToStack(Ty val) {
    nodeStack.emplace_back(val);
  }

  template <typename Ty> void popFromStack(Ty val) {
    assert(nodeStack.back().is<Ty>() && (nodeStack.back().get<Ty>() == val) &&
           "Incorrect stack structure!");
    nodeStack.pop_back();
  }

  Action walkToDeclPre(Decl *decl) override {
    visit(decl);
    pushToStack(decl);
    return Action::Continue;
  }

  bool walkToDeclPost(Decl *decl) override {
    popFromStack(decl);
    return true;
  }

  std::pair<Action, Expr *> walkToExprPre(Expr *expr) override {
    visit(expr);
    pushToStack(expr);
    return {Action::Continue, nullptr};
  }

  std::pair<bool, Expr *> walkToExprPost(Expr *expr) override {
    popFromStack(expr);
    return {true, nullptr};
  }

  Action walkToPatternPre(Pattern *pattern) override {
    visit(pattern);
    pushToStack(pattern);
    return Action::Continue;
  }

  bool walkToPatternPost(Pattern *pattern) override {
    popFromStack(pattern);
    return true;
  }

  Action walkToStmtPre(Stmt *stmt) override {
    visit(stmt);
    pushToStack(stmt);
    return Action::Continue;
  }

  bool walkToStmtPost(Stmt *stmt) override {
    popFromStack(stmt);
    return true;
  }

  Action walkToTypeReprPre(TypeRepr *tyRepr) override {
    visit(tyRepr);
    pushToStack(tyRepr);
    return Action::Continue;
  }

  bool walkToTypeReprPost(TypeRepr *tyRepr) override {
    popFromStack(tyRepr);
    return true;
  }
};

SmallVectorImpl<VerifiedNode> &ASTVerifierImplBase::getNodeStack() {
  return verifier.nodeStack;
}
//===- ASTVerificationFailure ---------------------------------------------===//

ASTVerificationFailure::ASTVerificationFailure(
    ASTVerifierImplBase &verifierImpl, VerifiedNode node, unsigned line)
    : verifierImpl(verifierImpl), verifier(verifierImpl.verifier),
      srcMgr(verifier.ctxt.srcMgr), node(node) {
  llvm::dbgs() << "===--- AST VERIFICATION FAILURE! ---===\n";
  llvm::dbgs() << "> caused by "
               << (verifier.isCheckedAST() ? "a type-checked" : "an unchecked")
               << " AST node at the following location: ";
  // getLoc(node).print(llvm::dbgs(), srcMgr, true);
  llvm::dbgs() << "\n> failure triggered in file '" << __FILE__ << "' at line "
               << line << "\n";
  llvm::dbgs() << "===---------------------------------===\n";
}

ASTVerificationFailure::~ASTVerificationFailure() {
  // Dump the node after everything, in case it's so badly ill-formed that the
  // AST dumper crashes.
  llvm::dbgs() << "\n===---       AST NODE DUMP       ---===\n";
  dump(node);
  llvm::dbgs() << "===---------------------------------===\n";
  llvm::report_fatal_error("AST Verification Failure: The AST is ill-formed!");
}

//===- BasicVerifierImpl --------------------------------------------------===//

void BasicVerifierImpl::visitMaybeValuePattern(MaybeValuePattern *pattern) {}

//===- CheckedVerifierImpl ------------------------------------------------===//

void CheckedVerifierImpl::visitDiscardExpr(DiscardExpr *expr) {
  if (!expr->getType()->is<LValueType>())
    CREATE_FAILURE(expr) << "DiscardExpr does not have an LValue Type";
}

void CheckedVerifierImpl::visitDeclRefExpr(DeclRefExpr *expr) {
  if (!expr->getType()->is<LValueType>())
    CREATE_FAILURE(expr) << "DeclRefExpr does not have an LValue Type";
}

void CheckedVerifierImpl::visitTupleElementExpr(TupleElementExpr *expr) {
  if (expr->getBase()->getType()->is<LValueType>() !=
      expr->getType()->is<LValueType>())
    CREATE_FAILURE(expr)
        << "TupleElementExpr's type must be an LValue when its base is!";
}

void CheckedVerifierImpl::visitLoadExpr(LoadExpr *expr) {
  Type type = expr->getType();
  Type subExprType = expr->getSubExpr()->getType();

  if (!subExprType)
    return;

  if (!subExprType->is<LValueType>())
    CREATE_FAILURE(expr)
        << "LoadExpr's subexpression does not have an LValue Type";

  if (type.getPtr() != subExprType->getRValueType().getPtr())
    CREATE_FAILURE(expr) << "LoadExpr type must me the exact same type as its "
                            "subexpression without the LValue!\n "
                         << "Expression type: '" << type
                         << "', Subexpression type: '" << subExprType
                         << "', Expected Type: '"
                         << subExprType->getRValueType() << "'";
}

void CheckedVerifierImpl::visitMaybeValuePattern(MaybeValuePattern *pattern) {
  if (Pattern *topLevelPattern = getTopLevelNode<Pattern>())
    CREATE_FAILURE(topLevelPattern)
        << "MaybeValuePattern can only be present at the top-level of a "
           "pattern, not as the children of another pattern!";

  Type type = pattern->getType();
  Type subPatTy = pattern->getSubPattern()->getType();

  if (!subPatTy)
    return;

  MaybeType *maybe = type->getAs<MaybeType>();
  if (!maybe)
    CREATE_FAILURE(pattern) << "MaybeValuePattern type must be a MaybeType!";

  if (maybe->getValueType().getPtr() != subPatTy.getPtr()) {
    CREATE_FAILURE(pattern)
        << "MaybeValuePattern type must me a MaybeType of its subpattern's "
           "type!\n "
        << "Pattern type: '" << type << "', Subpattern type: '" << subPatTy
        << "', Expected Type: '" << Type(MaybeType::get(subPatTy)) << "'";
  }
}

void CheckedVerifierImpl::visitUnaryExpr(UnaryExpr *expr) {
  if (expr->getOpKind() != UnaryOperatorKind::Deref)
    return;

  if (!expr->getType()->is<LValueType>())
    CREATE_FAILURE(expr) << "Dereference does not have an LValue Type";
}

} // namespace

//===- Entry Points -------------------------------------------------------===//

void sora::verify(SourceFile &sf, bool isChecked) {
#ifndef NDEBUG
  sf.walk(ASTVerifier(sf.astContext, isChecked));
#endif
}