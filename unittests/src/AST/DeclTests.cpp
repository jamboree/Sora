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
    cast<FuncDecl>(funcDecl)->setBody(BlockStmt::createEmpty(*ctxt, mid, end));
    Pattern *pat = new (*ctxt) DiscardPattern(mid);
    Expr *init = new (*ctxt) DiscardExpr(end);
    letDecl = new (*ctxt) LetDecl(nullptr, beg, pat, {}, init);
  }

  SourceManager srcMgr;
  DiagnosticEngine diagEng{srcMgr, llvm::outs()};
  std::unique_ptr<ASTContext> ctxt{ASTContext::create(srcMgr, diagEng)};

  SourceLoc beg, mid, end;

  Decl *varDecl;
  Decl *paramDecl;
  Decl *funcDecl;
  Decl *letDecl;
};
} // namespace

TEST_F(DeclTest, rtti) {
  EXPECT_TRUE(isa<VarDecl>(varDecl));
  EXPECT_TRUE(isa<ValueDecl>(varDecl));
  EXPECT_TRUE(isa<ParamDecl>(paramDecl));
  EXPECT_TRUE(isa<ValueDecl>(paramDecl));
  EXPECT_TRUE(isa<FuncDecl>(funcDecl));
  EXPECT_TRUE(isa<ValueDecl>(funcDecl));
  EXPECT_TRUE(isa<LetDecl>(letDecl));
  EXPECT_TRUE(isa<PatternBindingDecl>(letDecl));
}

TEST_F(DeclTest, getSourceRange) {
  // VarDecl
  EXPECT_EQ(beg, varDecl->getBegLoc());
  EXPECT_EQ(beg, varDecl->getEndLoc());
  EXPECT_EQ(SourceRange(beg, beg), varDecl->getSourceRange());

  // ParamDecl
  EXPECT_EQ(beg, paramDecl->getBegLoc());
  EXPECT_EQ(end, paramDecl->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), paramDecl->getSourceRange());

  // FuncDecl
  EXPECT_EQ(beg, funcDecl->getBegLoc());
  EXPECT_EQ(end, funcDecl->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), funcDecl->getSourceRange());

  // LetDecl
  EXPECT_EQ(beg, letDecl->getBegLoc());
  EXPECT_EQ(end, letDecl->getEndLoc());
  EXPECT_EQ(SourceRange(beg, end), letDecl->getSourceRange());
  Expr *letInit = cast<LetDecl>(letDecl)->getInitializer();
  cast<LetDecl>(letDecl)->setInitializer(nullptr);
  EXPECT_EQ(beg, letDecl->getBegLoc());
  EXPECT_EQ(mid, letDecl->getEndLoc());
  EXPECT_EQ(SourceRange(beg, mid), letDecl->getSourceRange());
  cast<LetDecl>(letDecl)->setInitializer(letInit);
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