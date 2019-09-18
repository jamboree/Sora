//===--- ASTDumper.cpp ------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/ASTVisitor.hpp"
#include "Sora/AST/Decl.hpp"
#include "Sora/AST/Expr.hpp"
#include "Sora/AST/Pattern.hpp"
#include "Sora/AST/Stmt.hpp"
#include "Sora/Common/LLVM.hpp"
#include "llvm/Support/raw_ostream.h"

using namespace sora;

namespace {

class Dumper : public SimpleASTVisitor<Dumper> {
  raw_ostream &out;
  unsigned indent;

public:
  Dumper(raw_ostream &out, unsigned indent) : out(out), indent(indent) {}

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
};
} // namespace

void Decl::dump(raw_ostream &out, unsigned indent) {}

void Expr::dump(raw_ostream &out, unsigned indent) {}

void Pattern::dump(raw_ostream &out, unsigned indent) {}

void Stmt::dump(raw_ostream &out, unsigned indent) {}