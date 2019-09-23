//===--- DeclTests.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Common/DiagnosticEngine.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "gtest/gtest.h"

using namespace sora;

namespace {
const char *str = "Hello, World!";

class DeclTest : public ::testing::Test {
protected:
  DeclTest() {
    // Setup SourceLocs
    beg = SourceLoc::fromPointer(str);
    mid = SourceLoc::fromPointer(str + 5);
    end = SourceLoc::fromPointer(str + 10);
    // Setup nodes
    varDecl = new (*ctxt) VarDecl(nullptr, beg, {});
    TypeRepr *tyRepr = new (*ctxt) IdentifierTypeRepr(end, {});
    paramDecl = new (*ctxt) ParamDecl(nullptr, beg, {}, {}, {{}, tyRepr});
    funcDecl = new (*ctxt) FuncDecl(nullptr, beg, {}, {}, nullptr, {});
    funcDecl->setBody(BlockStmt::createEmpty(*ctxt, mid, end));
    Pattern *pat = new (*ctxt) DiscardPattern(mid);
    Expr *init = new (*ctxt) DiscardExpr(end);
    letDecl = new (*ctxt) LetDecl(nullptr, beg, pat, {}, init);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, mid, end;

  VarDecl *varDecl;
  ParamDecl *paramDecl;
  FuncDecl *funcDecl;
  LetDecl *letDecl;
};
} // namespace

TEST_F(DeclTest, rtti) {
  // VarDecl
  EXPECT_TRUE(isa<VarDecl>((Decl*)varDecl));
  EXPECT_TRUE(isa<ValueDecl>((Decl *)varDecl));

  // ParamDecl
  EXPECT_TRUE(isa<ParamDecl>((Decl *)paramDecl));
  EXPECT_TRUE(isa<ValueDecl>((Decl *)paramDecl));

  // FuncDecl
  EXPECT_TRUE(isa<FuncDecl>((Decl *)funcDecl));
  EXPECT_TRUE(isa<ValueDecl>((Decl *)funcDecl));

  // LetDecl
  EXPECT_TRUE(isa<LetDecl>((Decl *)letDecl));
  EXPECT_TRUE(isa<PatternBindingDecl>((Decl *)letDecl));
}

TEST_F(DeclTest, getSourceRange) {
  Decl *cur;
  // VarDecl
  cur = varDecl;
  EXPECT_EQ(beg, cur->getBegLoc());
  EXPECT_EQ(beg, cur->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), cur->getSourceRange());

  // ParamDecl
  cur = paramDecl;
  EXPECT_EQ(beg, cur->getBegLoc());
  EXPECT_EQ(end, cur->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), cur->getSourceRange());

  // FuncDecl
  cur = funcDecl;
  EXPECT_EQ(beg, cur->getBegLoc());
  EXPECT_EQ(end, cur->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), cur->getSourceRange());

  // LetDecl
  cur = letDecl;
  EXPECT_EQ(beg, cur->getBegLoc());
  EXPECT_EQ(end, cur->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), cur->getSourceRange());
  Expr *letInit = letDecl->getInitializer();
  letDecl->setInitializer(nullptr);
  EXPECT_EQ(beg, cur->getBegLoc());
  EXPECT_EQ(mid, cur->getEndLoc());
  EXPECT_EQ(SourceRange(beg, mid), cur->getSourceRange());
  letDecl->setInitializer(letInit);
}

/// Tests Decl::getSourceFile and related features (getParent, getASTContext,
/// getDiagnosticEngine)
TEST_F(DeclTest, getSourceFile) {
  SourceFile sf({}, *ctxt, nullptr, Identifier());
  // Let's create a simple tree:
  // SourceFile
  //    FuncDecl
  //      ParamDecl
  FuncDecl *fn = new (*ctxt) FuncDecl(&sf, {}, {}, {}, nullptr, {});
  ParamDecl *param = new (*ctxt) ParamDecl(fn, {}, {}, {}, {});
  fn->setParamList(ParamList::create(*ctxt, {}, param, {}));

  // Now, let's try to retrieve our ASTContext, SourceFile and DiagnosticEngine
  // from the param and fn.
  EXPECT_EQ(ctxt.get(), &param->getASTContext());
  EXPECT_EQ(ctxt.get(), &fn->getASTContext());

  EXPECT_EQ(&diagEng, &param->getDiagnosticEngine());
  EXPECT_EQ(&diagEng, &fn->getDiagnosticEngine());

  EXPECT_EQ(&sf, &param->getSourceFile());
  EXPECT_EQ(&sf, &fn->getSourceFile());
}