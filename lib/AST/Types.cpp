//===--- Types.cpp ----------------------------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#include "Sora/AST/Types.hpp"
#include "Sora/AST/ASTContext.hpp"
#include "Sora/AST/Type.hpp"
#include "Sora/AST/TypeRepr.hpp"
#include "Sora/AST/TypeVisitor.hpp"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>

using namespace sora;

/// Check that all types are trivially destructible. This is needed
/// because, as they are allocated in the ASTContext's arenas, their destructors
/// are never called.
#define TYPE(ID, PARENT)                                                       \
  static_assert(std::is_trivially_destructible<ID##Type>::value,               \
                #ID "Type is not trivially destructible.");
#include "Sora/AST/TypeNodes.def"

//===- TypePrinter --------------------------------------------------------===//

namespace {
class TypePrinter : public TypeVisitor<TypePrinter> {
public:
  using parent = TypeVisitor<TypePrinter>;

  raw_ostream &out;
  const TypePrintOptions &printOptions;

  TypePrinter(raw_ostream &out, const TypePrintOptions &printOptions)
      : out(out), printOptions(printOptions) {}

  void visit(Type type) {
    if (type.isNull()) {
      if (!printOptions.allowNullTypes)
        llvm_unreachable("Can't print a null type!");
      out << "<null_type>";
      return;
    }
    parent::visit(type);
  }

  void visitIntegerType(IntegerType *type) {
    // Integer types have i or u as prefix, depending on their signedness, and
    // then just have the bitwidth or 'size' for pointer-sized ints.
    if (type->isSigned())
      out << 'i';
    else
      out << 'u';
    IntegerWidth width = type->getWidth();
    if (width.isPointerSized())
      out << "size";
    else
      out << width.getWidth();
  }

  void visitFloatType(FloatType *type) { out << 'f' << type->getWidth(); }

  void visitVoidType(VoidType *type) { out << "void"; }

  void visitNullType(NullType *type) { out << "null"; }

  void visitBoolType(BoolType *type) { out << "bool"; }

  void visitReferenceType(ReferenceType *type) {
    out << '&';
    if (type->isMut())
      out << "mut ";
    visit(type->getPointeeType());
  }

  void visitMaybeType(MaybeType *type) {
    out << "maybe ";
    visit(type->getValueType());
  }

  void printTuple(ArrayRef<Type> elems) {
    out << '(';
    if (!elems.empty()) {
      visit(elems[0]);
      for (Type elem : elems.slice(1)) {
        out << ", ";
        visit(elem);
      }
    }
    out << ')';
  }

  void visitTupleType(TupleType *type) { printTuple(type->getElements()); }

  void visitFunctionType(FunctionType *type) {
    printTuple(type->getArgs());
    out << " -> ";
    visit(type->getReturnType());
  }

  void visitLValueType(LValueType *type) {
    if (printOptions.printLValues)
      out << "@lvalue ";
    visit(type->getObjectType());
  }

  void visitErrorType(ErrorType *) {
    if (!printOptions.allowErrorTypes)
      llvm_unreachable("Can't print an ErrorType");
    out << "<error_type>";
  }

  void visitTypeVariableType(TypeVariableType *type) {
    if (printOptions.printTypeVariablesAsUnderscore) {
      out << '_';
      return;
    }
    out << "$T" << type->getID();
  }
};
} // namespace

//===- Type/CanType/TypeLoc -----------------------------------------------===//

void Type::print(raw_ostream &out, const TypePrintOptions &printOptions) const {
  TypePrinter(out, printOptions).visit(*this);
}

std::string Type::getString(const TypePrintOptions &printOptions) const {
  std::string rtr;
  llvm::raw_string_ostream out(rtr);
  print(out, printOptions);
  out.str();
  return rtr;
}

bool CanType::isValid() const {
  if (const TypeBase *ptr = getPtr())
    return ptr->isCanonical();
  return true;
}

SourceRange TypeLoc::getSourceRange() const {
  return tyRepr ? tyRepr->getSourceRange() : SourceRange();
}

SourceLoc TypeLoc::getBegLoc() const {
  return tyRepr ? tyRepr->getBegLoc() : SourceLoc();
}

SourceLoc TypeLoc::getLoc() const {
  return tyRepr ? tyRepr->getLoc() : SourceLoc();
}

SourceLoc TypeLoc::getEndLoc() const {
  return tyRepr ? tyRepr->getEndLoc() : SourceLoc();
}

std::string DiagnosticArgumentFormatter<Type>::format(Type type) {
  return type.getString(TypePrintOptions::forDiagnostics());
}

//===- TypeBase/Types -----------------------------------------------------===//

void *TypeBase::operator new(size_t size, ASTContext &ctxt, ArenaKind allocator,
                             unsigned align) {
  return ctxt.allocate(size, align, allocator);
}

CanType TypeBase::getCanonicalType() const {
  /// FIXME: This isn't ideal, but nothing here will mutate the type, so it
  /// should be OK.
  TypeBase *thisType = const_cast<TypeBase *>(this);

  // This type is already canonical:
  if (isCanonical())
    /// FIXME: Ideally, there should be no const_cast here.
    return CanType(thisType);
  // This type has already computed its canonical version
  if (TypeBase *canType = ctxtOrCanType.dyn_cast<TypeBase *>())
    return CanType(canType);
  // This type hasn't calculated its canonical version.
  assert(ctxtOrCanType.is<ASTContext *>() && "Not TypeBase*/ASTContext*?");

  Type result = nullptr;
  ASTContext &ctxt = *ctxtOrCanType.get<ASTContext *>();

  switch (getKind()) {
  case TypeKind::Integer:
  case TypeKind::Float:
  case TypeKind::Void:
  case TypeKind::Bool:
  case TypeKind::Null:
  case TypeKind::Error:
  case TypeKind::TypeVariable:
    llvm_unreachable("Type is always canonical!");
  case TypeKind::Reference: {
    ReferenceType *type = cast<ReferenceType>(thisType);
    result = ReferenceType::get(type->getPointeeType()->getCanonicalType(),
                                type->isMut());
    break;
  }
  case TypeKind::Maybe: {
    MaybeType *type = cast<MaybeType>(thisType);
    result = MaybeType::get(type->getValueType()->getCanonicalType());
    break;
  }
  case TypeKind::Tuple: {
    TupleType *type = cast<TupleType>(thisType);
    // The canonical version of '()' is 'void'.
    if (type->isEmpty()) {
      result = ctxt.voidType;
      break;
    }
    assert(type->getNumElements() != 1 &&
           "Single element tuples shouldn't exist!");
    // Canonicalise elements
    SmallVector<Type, 4> elems;
    elems.reserve(type->getNumElements());
    for (Type elem : type->getElements())
      elems.push_back(elem->getCanonicalType());
    // Create the canonical type
    result = TupleType::get(ctxt, elems);
    break;
  }
  case TypeKind::Function: {
    FunctionType *type = cast<FunctionType>(thisType);
    // Canonicalize arguments
    SmallVector<Type, 4> args;
    args.reserve(type->getNumArgs());
    for (Type arg : type->getArgs())
      args.push_back(arg->getCanonicalType());
    // Canonicalize return type
    Type rtr = type->getReturnType()->getCanonicalType();
    // Create the canonical type
    result = FunctionType::get(args, rtr);
    break;
  }
  case TypeKind::LValue: {
    LValueType *type = cast<LValueType>(thisType);
    result = LValueType::get(type->getObjectType()->getCanonicalType());
    break;
  }
  }
  // Cache the canonical type & return
  assert(result && "result is nullptr?");
  assert(result->isCanonical() && "result isn't canonical?");
  ctxtOrCanType = result.getPtr();
  return CanType(result);
}

Type TypeBase::getRValue() const {
  if (LValueType *lvalue = dyn_cast<LValueType>(const_cast<TypeBase *>(this)))
    return lvalue->getObjectType()->getRValue();
  return const_cast<TypeBase *>(this);
}

bool TypeBase::isLValue() const { return isa<LValueType>(this); }

void TypeBase::print(raw_ostream &out,
                     const TypePrintOptions &printOptions) const {
  Type(const_cast<TypeBase *>(this)).print(out, printOptions);
}

std::string TypeBase::getString(const TypePrintOptions &printOptions) const {
  return Type(const_cast<TypeBase *>(this)).getString(printOptions);
}

FloatType *FloatType::get(ASTContext &ctxt, FloatKind kind) {
  switch (kind) {
  case FloatKind::IEEE32:
    return ctxt.f32Type->castTo<FloatType>();
  case FloatKind::IEEE64:
    return ctxt.f64Type->castTo<FloatType>();
  }
  llvm_unreachable("Unknown FloatKind!");
}

const llvm::fltSemantics &FloatType::getAPFloatSemantics() const {
  switch (getFloatKind()) {
  case FloatKind::IEEE32:
    return APFloat::IEEEsingle();
  case FloatKind::IEEE64:
    return APFloat::IEEEdouble();
  }
  llvm_unreachable("Unknown FloatKind!");
}

Optional<unsigned> TupleType::lookup(Identifier ident) const {
  IntegerWidth::Status status;
  // Parse the identifier string as an arbitrary-width integer
  llvm::APInt value =
      IntegerWidth::arbitrary().parse(ident.c_str(), false, 0, &status);
  // If the value couldn't be parsed successfully, it can't be an integer.
  if (status != IntegerWidth::Status::Ok)
    return None;
  // The maximum index of the tuple is like an array: its size-1.
  const unsigned maxIdx = getNumElements() - 1;
  // If the value is greater or equal to that value, the index isn't legit,
  // else, return the parsed index.
  unsigned result = value.getLimitedValue(maxIdx);
  if (result == maxIdx)
    return None;
  return result;
}

void TupleType::Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> elements) {
  id.AddInteger(elements.size());
  for (Type type : elements)
    id.AddPointer(type.getPtr());
}

void FunctionType::Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> args,
                           Type rtr) {
  for (auto arg : args)
    id.AddPointer(arg.getPtr());
  id.AddPointer(rtr.getPtr());
}
