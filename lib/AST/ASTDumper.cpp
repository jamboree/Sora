//===--- ASTDumper.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/SourceFile.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/Common/LLVM.hpp"
#include "Sora/Common/SourceManager.hpp"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

namespace {
const char *getKindStr(DeclKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown DeclKind");
#define DECL(ID, PARENT)                                                       \
  case DeclKind::ID:                                                           \
    return #ID "Decl";
#include "Sora/AST/DeclNodes.def"
  }
}

const char *getKindStr(ExprKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown ExprKind");
#define EXPR(ID, PARENT)                                                       \
  case ExprKind::ID:                                                           \
    return #ID "Expr";
#include "Sora/AST/ExprNodes.def"
  }
}

const char *getKindStr(StmtKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown StmtKind");
#define STMT(ID, PARENT)                                                       \
  case StmtKind::ID:                                                           \
    return #ID "Stmt";
#include "Sora/AST/StmtNodes.def"
  }
}

const char *getKindStr(PatternKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown PatternKind");
#define PATTERN(ID, PARENT)                                                    \
  case PatternKind::ID:                                                        \
    return #ID "Pattern";
#include "Sora/AST/PatternNodes.def"
  }
}

const char *getKindStr(TypeReprKind kind) {
  switch (kind) {
  default:
    llvm_unreachable("unknown TypeReprKind");
#define TYPEREPR(ID, PARENT)                                                   \
  case TypeReprKind::ID:                                                       \
    return #ID "TypeRepr";
#include "Sora/AST/TypeReprNodes.def"
  }
}

using Colors = llvm::raw_ostream::Colors;

constexpr Colors nodeKindColor = Colors::WHITE;
constexpr Colors locColor = Colors::SAVEDCOLOR;

class Dumper : public SimpleASTVisitor<Dumper> {
  raw_ostream &out;
  const SourceManager &srcMgr;
  unsigned curIndent;
  const unsigned indentSize;
  bool enableColors = false;

  /// RAII class that increases/decreases indentation when it's
  /// constructed/destroyed.
  struct IncreaseIndentRAII {
    Dumper &dumper;

    IncreaseIndentRAII(Dumper &dumper) : dumper(dumper) {
      dumper.curIndent += dumper.indentSize;
    }

    ~IncreaseIndentRAII() { dumper.curIndent -= dumper.indentSize; }
  };

  /// Increases the indentation temporarily (until the returned object is
  /// destroyed)
  IncreaseIndentRAII increaseIndent() { return IncreaseIndentRAII(*this); }

  llvm::WithColor withColor(Colors color) {
    return llvm::WithColor(out, color, !enableColors);
  }

  /// Dumps basic information about a Decl.
  /// For ValueDecls, this dumps their type, identifier and identifier location.
  void dumpCommon(Decl *decl) {
    out.indent(curIndent);
    withColor(nodeKindColor) << getKindStr(decl->getKind());

    if (ValueDecl *vd = dyn_cast<ValueDecl>(decl)) {
      out << ' ';
      dumpType(vd->getValueType(), "type");
      out << ' ';
      dumpIdent(vd->getIdentifier(), "identifier");
      out << ' ';
      dumpLoc(vd->getIdentifierLoc(), "identifierLoc");
    }
  }

  void dumpCommon(ParamList *paramList) {
    out.indent(curIndent);
    withColor(nodeKindColor) << "ParamList";
  }

  void dumpCommon(Expr *expr) {
    out.indent(curIndent);
    withColor(nodeKindColor) << getKindStr(expr->getKind());
    if (expr->isImplicit())
      out << " implicit";
    out << ' ';
    dumpType(expr->getType(), "type");
  }

  void dumpCommon(Pattern *pattern) {
    out.indent(curIndent);
    withColor(nodeKindColor) << getKindStr(pattern->getKind());
  }

  void dumpCommon(Stmt *stmt) {
    out.indent(curIndent);
    withColor(nodeKindColor) << getKindStr(stmt->getKind());
  }

  void dumpCommon(TypeRepr *tyRepr) {
    out.indent(curIndent);
    withColor(nodeKindColor) << getKindStr(tyRepr->getKind());
  }

  /// For null nodes.
  void printNoNode() {
    out.indent(curIndent);
    withColor(nodeKindColor) << "<null node>\n";
  }

  void dumpLoc(SourceLoc loc, StringRef name) {
    auto out = withColor(locColor);
    out << name << '=';
    loc.print(out, srcMgr, false);
  }

  void dumpRange(SourceRange range, StringRef name) {
    auto out = withColor(locColor);
    out << name << '=';
    range.print(out, srcMgr, false);
  }

  void dumpIdent(Identifier ident, StringRef name) {
    out << name << "=" << (ident ? ident.str() : "<null identifier>");
  }

  void dumpType(Type type, StringRef name) {
    out << name << "=" << (type ? "?TODO?" : "<null type>");
  }

public:
  Dumper(raw_ostream &out, const SourceManager &srcMgr, unsigned indentSize)
      : out(out), srcMgr(srcMgr), curIndent(0), indentSize(indentSize) {
    enableColors = out.has_colors();
  }

  /// Override the visit method so it calls printNoNode() when the node is null.
  template <typename T> void visit(T node) {
    node ? ASTVisitor::visit(node) : printNoNode();
  }

  void visit(ParamList *paramList) {
    paramList ? visitParamList(paramList) : printNoNode();
  }

  void visit(StmtCondition cond) {
    !cond.isNull() ? visitStmtCondition(cond) : printNoNode();
  }

  /// Provide an alternative visit entry point that only visits the node if
  /// it's non-null, else it ignores it.
  template <typename T> void visitIf(T node) {
    if (node)
      ASTVisitor::visit(node);
  }
  //===--- Decl -----------------------------------------------------------===//

  void visitVarDecl(VarDecl *decl) {
    dumpCommon(decl);
    out << '\n';

    if (TypeRepr *tyRepr = decl->getTypeLoc().getTypeRepr()) {
      auto indent = increaseIndent();
      visit(tyRepr);
    }
  }

  void visitParamDecl(ParamDecl *decl) {
    dumpCommon(decl);
    out << '\n';

    auto indent = increaseIndent();
    visit(decl->getTypeLoc().getTypeRepr());
  }

  void visitParamList(ParamList *list) {
    dumpCommon(list);
    out << " numElements=" << list->getNumParams() << ' ';
    dumpLoc(list->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(list->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    for (auto elem : list->getParams())
      visit(elem);
  }

  void visitFuncDecl(FuncDecl *decl) {
    dumpCommon(decl);
    out << ' ';
    dumpLoc(decl->getFuncLoc(), "fnLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(decl->getParamList());
    visitIf(decl->getReturnTypeLoc().getTypeRepr());
    visit(decl->getBody());
  }

  void visitLetDecl(LetDecl *decl) {
    dumpCommon(decl);
    out << ' ';
    dumpLoc(decl->getLetLoc(), "letLoc");
    if (decl->hasInitializer()) {
      out << ' ';
      dumpLoc(decl->getEqualLoc(), "equalLoc");
    }
    out << '\n';

    auto indent = increaseIndent();
    visit(decl->getPattern());
    visitIf(decl->getInitializer());
  }

  //===--- Expr -----------------------------------------------------------===//

  void visitUnresolvedDeclRefExpr(UnresolvedDeclRefExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpIdent(expr->getIdentifier(), "ident");
    out << ' ';
    dumpLoc(expr->getLoc(), "loc");
    out << '\n';
  }

  void visitUnresolvedMemberRefExpr(UnresolvedMemberRefExpr *expr) {
    dumpCommon(expr);
    out << ' ' << (expr->isArrow() ? "arrow" : "dot") << ' ';
    dumpIdent(expr->getMemberIdentifier(), "memberIdent");
    out << ' ';
    dumpLoc(expr->getMemberIdentifierLoc(), "memberIdentLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getBase());
  }

  void visitDiscardExpr(DiscardExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getLoc(), "loc");
    out << '\n';
  }

  void visitIntegerLiteralExpr(IntegerLiteralExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getLoc(), "loc");
    out << " str='" << expr->getString() << "' rawValue=" << expr->getRawValue()
        << '\n';
  }

  void visitFloatLiteralExpr(FloatLiteralExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getLoc(), "loc");
    out << " str='" << expr->getString() << "'\n";
  }

  void visitBooleanLiteralExpr(BooleanLiteralExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getLoc(), "loc");
    out << " value=" << (expr->getValue() ? "true" : "false") << "\n";
  }

  void visitNullLiteralExpr(NullLiteralExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getLoc(), "loc");
    out << '\n';
  }

  void visitErrorExpr(ErrorExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpRange(expr->getSourceRange(), "range");
    out << '\n';
  }

  void visitTupleElementExpr(TupleElementExpr *expr) {
    dumpCommon(expr);
    out << ' ' << (expr->isArrow() ? "arrow" : "dot") << ' ';
    dumpLoc(expr->getOpLoc(), "opLoc");
    out << " index=" << expr->getIndex() << " ";
    dumpLoc(expr->getIndexLoc(), "indexLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getBase());
  }

  void visitTupleExpr(TupleExpr *expr) {
    dumpCommon(expr);
    out << " numElements=" << expr->getNumElements() << ' ';
    dumpLoc(expr->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(expr->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    for (auto elem : expr->getElements())
      visit(elem);
  }

  void visitParenExpr(ParenExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(expr->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getSubExpr());
  }

  void visitCallExpr(CallExpr *expr) {
    dumpCommon(expr);
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getFn());
    visit(expr->getArgs());
  }

  void visitConditionalExpr(ConditionalExpr *expr) {
    dumpCommon(expr);
    out << ' ';
    dumpLoc(expr->getQuestionLoc(), "questionLoc");
    out << ' ';
    dumpLoc(expr->getColonLoc(), "colonLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getCond());
    visit(expr->getThen());
    visit(expr->getElse());
  }

  void visitBinaryExpr(BinaryExpr *expr) {
    dumpCommon(expr);
    out << ' ' << expr->getOpSpelling() << " (" << expr->getOpKindStr() << ") ";
    dumpLoc(expr->getOpLoc(), "opLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getLHS());
    visit(expr->getRHS());
  }

  void visitUnaryExpr(UnaryExpr *expr) {
    dumpCommon(expr);
    out << ' ' << expr->getOpSpelling() << " (" << expr->getOpKindStr() << ") ";
    dumpLoc(expr->getOpLoc(), "opLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(expr->getSubExpr());
  }

  //===--- Pattern --------------------------------------------------------===//

  void visitVarPattern(VarPattern *pattern) {
    dumpCommon(pattern);
    out << ' ';
    dumpLoc(pattern->getLoc(), "loc");
    out << ' ';
    dumpIdent(pattern->getIdentifier(), "ident");
    out << '\n';
  }

  void visitDiscardPattern(DiscardPattern *pattern) {
    dumpCommon(pattern);
    out << ' ';
    dumpLoc(pattern->getLoc(), "loc");
    out << '\n';
  }

  void visitMutPattern(MutPattern *pattern) {
    dumpCommon(pattern);
    out << ' ';
    dumpLoc(pattern->getMutLoc(), "mutLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(pattern->getSubPattern());
  }

  void visitParenPattern(ParenPattern *pattern) {
    dumpCommon(pattern);
    out << ' ';
    dumpLoc(pattern->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(pattern->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(pattern->getSubPattern());
  }

  void visitTuplePattern(TuplePattern *pattern) {
    dumpCommon(pattern);
    out << " numElements=" << pattern->getNumElements() << ' ';
    dumpLoc(pattern->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(pattern->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    for (auto elem : pattern->getElements())
      visit(elem);
  }

  void visitTypedPattern(TypedPattern *pattern) {
    dumpCommon(pattern);
    out << '\n';

    auto indent = increaseIndent();
    visit(pattern->getSubPattern());
    visit(pattern->getTypeRepr());
  }

  //===--- SourceFile -----------------------------------------------------===//

  void visitSourceFile(const SourceFile &sf) {
    out.indent(curIndent);
    withColor(nodeKindColor) << "SourceFile";
    out << " numMembers=" << sf.getNumMembers() << "\n ";
    auto indent = increaseIndent();
    for (Decl *member : sf.getMembers())
      visit(member);
  }

  //===--- StmtCondition --------------------------------------------------===//

  void visitStmtCondition(StmtCondition cond) {
    out.indent(curIndent);
    withColor(nodeKindColor) << "StmtCondition";

    if (cond.isExpr())
      visit(cond.getExpr());
    else if (cond.isLetDecl())
      visit(cond.getLetDecl());
    else
      llvm_unreachable("unknown StmtCondition kind");
  }

  //===--- Stmt -----------------------------------------------------------===//

  void visitContinueStmt(ContinueStmt *stmt) {
    dumpCommon(stmt);
    out << ' ';
    dumpLoc(stmt->getLoc(), "loc");
    out << '\n';
  }

  void visitBreakStmt(BreakStmt *stmt) {
    dumpCommon(stmt);
    out << ' ';
    dumpLoc(stmt->getLoc(), "loc");
    out << '\n';
  }

  void visitReturnStmt(ReturnStmt *stmt) {
    dumpCommon(stmt);
    out << ' ';
    dumpLoc(stmt->getLoc(), "loc");
    out << '\n';
    if (!stmt->hasResult())
      return;

    if (stmt->hasResult()) {
      auto indent = increaseIndent();
      visit(stmt->getResult());
    }
  }

  void visitBlockStmt(BlockStmt *stmt) {
    dumpCommon(stmt);
    out << " numElement=" << stmt->getNumElements();
    out << ' ';
    dumpLoc(stmt->getLeftCurlyLoc(), "leftCurlyLoc");
    out << ' ';
    dumpLoc(stmt->getRightCurlyLoc(), "rightCurlyLoc");
    out << '\n';

    auto indent = increaseIndent();
    for (auto elem : stmt->getElements())
      visit(elem);
  }

  void visitIfStmt(IfStmt *stmt) {
    dumpCommon(stmt);
    out << ' ';
    dumpLoc(stmt->getIfLoc(), "ifLoc");
    if (SourceLoc loc = stmt->getElseLoc())
      dumpLoc(loc, "elseLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(stmt->getCond());
    visit(stmt->getThen());
    visitIf(stmt->getElse());
  }

  void visitWhileStmt(WhileStmt *stmt) {
    dumpCommon(stmt);
    out << ' ';
    dumpLoc(stmt->getWhileLoc(), "whileLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(stmt->getCond());
    visit(stmt->getBody());
  }

  //===--- TypeRepr -------------------------------------------------------===//

  void visitIdentifierTypeRepr(IdentifierTypeRepr *tyRepr) {
    dumpCommon(tyRepr);
    out << ' ';
    dumpLoc(tyRepr->getLoc(), "loc");
    out << '\n';
  }

  void visitParenTypeRepr(ParenTypeRepr *tyRepr) {
    dumpCommon(tyRepr);
    out << ' ';
    dumpLoc(tyRepr->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(tyRepr->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(tyRepr->getSubTypeRepr());
  }

  void visitTupleTypeRepr(TupleTypeRepr *tyRepr) {
    dumpCommon(tyRepr);
    out << " numElements=" << tyRepr->getNumElements() << ' ';
    dumpLoc(tyRepr->getLParenLoc(), "lParenLoc");
    out << ' ';
    dumpLoc(tyRepr->getRParenLoc(), "rParenLoc");
    out << '\n';

    auto indent = increaseIndent();
    for (auto elem : tyRepr->getElements())
      visit(elem);
  }

  void visitArrayTypeRepr(ArrayTypeRepr *tyRepr) {
    dumpCommon(tyRepr);
    out << ' ';
    dumpLoc(tyRepr->getRSquareLoc(), "lSquareLoc");
    out << ' ';
    dumpLoc(tyRepr->getLSquareLoc(), "rSquareLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(tyRepr->getSubTypeRepr());
    visit(tyRepr->getSizeExpr());
  }

  void visitReferenceTypeRepr(ReferenceTypeRepr *tyRepr) {
    dumpCommon(tyRepr);
    out << (tyRepr->hasMut() ? " mutable " : " immutable ");
    dumpLoc(tyRepr->getAmpLoc(), "ampLoc");
    if (tyRepr->hasMut()) {
      out << ' ';
      dumpLoc(tyRepr->getMutLoc(), "mutLoc");
    }
    out << '\n';

    auto indent = increaseIndent();
    visit(tyRepr->getSubTypeRepr());
  }

  void visitMaybeTypeRepr(MaybeTypeRepr *tyRepr) {
    dumpCommon(tyRepr);
    out << ' ';
    dumpLoc(tyRepr->getMaybeLoc(), "maybeLoc");
    out << '\n';

    auto indent = increaseIndent();
    visit(tyRepr->getSubTypeRepr());
  }
};
} // namespace

void Decl::dump(raw_ostream &out, unsigned indent) const {
  Dumper(out, getASTContext().srcMgr, indent).visit(const_cast<Decl *>(this));
}

void Expr::dump(raw_ostream &out, const SourceManager &srcMgr,
                unsigned indent) const {
  Dumper(out, srcMgr, indent).visit(const_cast<Expr *>(this));
}

void Pattern::dump(raw_ostream &out, const SourceManager &srcMgr,
                   unsigned indent) const {
  Dumper(out, srcMgr, indent).visit(const_cast<Pattern *>(this));
}

void TypeRepr::dump(raw_ostream &out, const SourceManager &srcMgr,
                    unsigned indent) const {
  Dumper(out, srcMgr, indent).visit(const_cast<TypeRepr *>(this));
}

void SourceFile::dump(raw_ostream &out, unsigned indent) const {
  Dumper(out, astContext.srcMgr, indent).visitSourceFile(*this);
}

void Stmt::dump(raw_ostream &out, const SourceManager &srcMgr,
                unsigned indent) const {
  Dumper(out, srcMgr, indent).visit(const_cast<Stmt *>(this));
}