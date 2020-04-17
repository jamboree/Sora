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
    char tyVarLetter;
    switch (type->getTypeVariableKind()) {
    case TypeVariableKind::General:
      tyVarLetter = 'T';
      break;
    case TypeVariableKind::Integer:
      tyVarLetter = 'I';
      break;
    case TypeVariableKind::Float:
      tyVarLetter = 'F';
      break;
    }
    out << '$' << tyVarLetter << type->getID();
    if (Type binding = type->getBinding()) {
      out << '(';
      visit(binding);
      out << ')';
    }
  }
};
} // namespace

//===- TypeRebuilder ------------------------------------------------------===//

namespace {
class TypeRebuilder : public TypeVisitor<TypeRebuilder, Type> {
  using Parent = TypeVisitor<TypeRebuilder, Type>;

public:
  ASTContext &ctxt;
  const std::function<Type(Type)> rebuilder;

  TypeRebuilder(ASTContext &ctxt, std::function<Type(Type)> rebuilder)
      : ctxt(ctxt), rebuilder(rebuilder) {}

  /// Calls visit on \p type to visit its children, then rebuilds \p type if
  /// needed, or returns nullptr if it did not change.
  Type doIt(Type type) {
    if (Type visited = visit(type)) {
      // The visit changed, use the rebuilt version of the result if it's
      // non-null, else just return the result of the visit.
      if (Type rebuilt = rebuilder(visited))
        return rebuilt;
      return visited;
    }
    // If the visit didn't change the type, maybe the rebuilder will.
    return rebuilder(type);
  }

  // The visit functions call doIt on the element types and rebuild the type if
  // needed. They should not call "rebuilder" or "visit" themselves.
  // "Leaf" types should just return nullptr.

  Type visitBuiltinType(BuiltinType *type) { return nullptr; }

  Type visitReferenceType(ReferenceType *type) {
    if (Type rebuilt = doIt(type->getPointeeType()))
      return ReferenceType::get(rebuilt, type->isMut());
    return nullptr;
  }

  Type visitMaybeType(MaybeType *type) {
    if (Type rebuilt = doIt(type->getValueType()))
      return MaybeType::get(rebuilt);
    return nullptr;
  }

  Type visitTupleType(TupleType *type) {
    bool needsRebuilding = false;

    SmallVector<Type, 4> elems;
    elems.reserve(type->getNumElements());
    for (Type elem : type->getElements()) {
      if (Type rebuilt = doIt(elem)) {
        needsRebuilding = true;
        elems.push_back(rebuilt);
      }
      else
        elems.push_back(elem);
    }

    if (!needsRebuilding)
      return nullptr;
    return TupleType::get(ctxt, elems);
  }

  Type visitFunctionType(FunctionType *type) {
    bool needsRebuilding = false;

    Type rtr = type->getReturnType();
    if (Type rebuilt = doIt(rtr)) {
      needsRebuilding = true;
      rtr = rebuilt;
    }

    SmallVector<Type, 4> args;
    args.reserve(type->getNumArgs());
    for (Type arg : type->getArgs()) {
      if (Type rebuilt = doIt(arg)) {
        needsRebuilding = true;
        args.push_back(rebuilt);
      }
      else
        args.push_back(arg);
    }

    if (!needsRebuilding)
      return nullptr;
    return FunctionType::get(args, rtr);
  }

  Type visitLValueType(LValueType *type) {
    if (Type rebuilt = doIt(type->getObjectType()))
      return LValueType::get(rebuilt);
    return nullptr;
  }

  Type visitErrorType(ErrorType *type) { return nullptr; }

  Type visitTypeVariableType(TypeVariableType *type) { return nullptr; }
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

std::string DiagnosticArgument<Type>::format(Type type) {
  return type.getString(TypePrintOptions::forDiagnostics());
}

//===- TypeBase -----------------------------------------------------------===//

void *TypeBase::operator new(size_t size, ASTContext &ctxt, ArenaKind allocator,
                             unsigned align) {
  return ctxt.allocate(size, align, allocator);
}

Type TypeBase::rebuildType(std::function<Type(Type)> rebuilder) const {
  // FIXME: This isn't ideal, but all types are immutable, so it should be ok.
  TypeBase *thisType = const_cast<TypeBase *>(this);
  if (Type rebuilt = TypeRebuilder(getASTContext(), rebuilder).doIt(thisType))
    return rebuilt;
  return thisType;
}

Type TypeBase::rebuildTypeWithoutLValues() const {
  if (!hasLValue())
    return const_cast<TypeBase *>(this);
  return rebuildType([&](Type type) -> Type {
    if (LValueType *lvalue = type->getAs<LValueType>())
      return lvalue->getObjectType();
    return nullptr;
  });
}

CanType TypeBase::getCanonicalType() const {
  // FIXME: This isn't ideal, but all types are immutable, so it should be ok.
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

Type TypeBase::getRValueType() const {
  if (LValueType *lvalue = dyn_cast<LValueType>(const_cast<TypeBase *>(this)))
    return lvalue->getObjectType()->getRValueType();
  return const_cast<TypeBase *>(this);
}

bool TypeBase::isLValueType() const { return isa<LValueType>(this); }

bool TypeBase::isBoolType() const { return getCanonicalType()->is<BoolType>(); }

bool TypeBase::isVoidType() const { return getCanonicalType()->is<VoidType>(); }

bool TypeBase::isNullType() const { return getCanonicalType()->is<NullType>(); }

bool TypeBase::isAnyIntegerType() const {
  return getCanonicalType()->is<IntegerType>();
}

bool TypeBase::isAnyFloatType() const {
  return getCanonicalType()->is<FloatType>();
}

bool TypeBase::isTupleType() const {
  return getDesugaredType()->is<TupleType>();
}

bool TypeBase::isMaybeType() const {
  return getCanonicalType()->is<MaybeType>();
}

Type TypeBase::getMaybeTypeValueType() const {
  if (!isMaybeType())
    return nullptr;
  return getDesugaredType()->castTo<MaybeType>()->getValueType();
}

void TypeBase::print(raw_ostream &out,
                     const TypePrintOptions &printOptions) const {
  Type(const_cast<TypeBase *>(this)).print(out, printOptions);
}

std::string TypeBase::getString(const TypePrintOptions &printOptions) const {
  return Type(const_cast<TypeBase *>(this)).getString(printOptions);
}

//===- FloatType ----------------------------------------------------------===//

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

//===- TupleType ----------------------------------------------------------===//

Optional<size_t> TupleType::lookup(Identifier ident) const {
  IntegerWidth::Status status;
  // Parse the identifier string as an arbitrary-width integer written in base
  // 10
  llvm::APInt value =
      IntegerWidth::arbitrary().parse(ident.c_str(), false, 10, &status);
  // If the value couldn't be parsed successfully, it can't be an integer.
  if (status != IntegerWidth::Status::Ok)
    return None;
  // The maximum index of the tuple is like an array: its size-1.
  const size_t numElem = getNumElements();
  // If the value is greater or equal to that value, the index isn't legit,
  // else, return the parsed index.
  size_t result = value.getLimitedValue(numElem);
  if (result == numElem)
    return None;
  return result;
}

void TupleType::Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> elements) {
  id.AddInteger(elements.size());
  for (Type type : elements)
    id.AddPointer(type.getPtr());
}

//===- FunctionType -------------------------------------------------------===//

void FunctionType::Profile(llvm::FoldingSetNodeID &id, ArrayRef<Type> args,
                           Type rtr) {
  for (auto arg : args)
    id.AddPointer(arg.getPtr());
  id.AddPointer(rtr.getPtr());
}

//===- TypeVariableType ---------------------------------------------------===//

void TypeVariableType::visitBindings(std::function<void(Type)> visitor) const {
  if (!binding)
    return;
  visitor(binding);
  auto *tvBinding = binding->getDesugaredType()->getAs<TypeVariableType>();
  if (!tvBinding)
    return;
  tvBinding->visitBindings(visitor);
}

void TypeVariableType::updateTypeVariableKind() {
  assert(isBound() && "Can only use this on bound type variables!");
  if (binding->isAnyIntegerType())
    return setTypeVariableKind(TypeVariableKind::Integer);
  if (binding->isAnyFloatType())
    return setTypeVariableKind(TypeVariableKind::Float);
  if (auto *typeVarBinding = binding->getAs<TypeVariableType>()) {
    if (isGeneralTypeVariable()) {
      if (!typeVarBinding->isGeneralTypeVariable())
        setTypeVariableKind(typeVarBinding->getTypeVariableKind());
    }
  }
}

bool TypeVariableType::canBindTo(Type type) const {
  if (isBound() || type->hasLValue())
    return false;

  CanType canType = type->getCanonicalType();
  // Cannot bind a type variable to itself.
  // FIXME: It'd be great to have a more advanced cycle detection system.
  if (auto *tv = canType->getAs<TypeVariableType>())
    if (this == tv)
      return false;

  switch (getTypeVariableKind()) {
  case TypeVariableKind::General:
    return true;
  case TypeVariableKind::Integer:
    if (canType->isAnyIntegerType())
      return true;
    if (TypeVariableType *tv = canType->getAs<TypeVariableType>())
      return tv->isIntegerTypeVariable();
    return false;
  case TypeVariableKind::Float:
    if (canType->isAnyFloatType())
      return true;
    if (TypeVariableType *tv = canType->getAs<TypeVariableType>())
      return tv->isFloatTypeVariable();
    return false;
  }
  llvm_unreachable("Unknown TypeVariableKind!");
}

void TypeVariableType::bindTo(Type type) {
  assert(canBindTo(type) && "Cannot bind to this type!");
  assert(binding.isNull() && "Overwriting binding!");
  binding = type;
  updateTypeVariableKind();
}
