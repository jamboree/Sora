//===--- Ops.td - Sora MLIR Dialect Operations Definitions ------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#ifndef SORAIR_OPS
#define SORAIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

// Provide a definition of the 'sora' dialect in the ODS framework so that we
// can define our operations.
def Sora_Dialect : Dialect {
  // The namespace of our dialect
  let name = "sora";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::mlir::sora";
}

// Base class Sora Dialect Operations 
class Sora_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Sora_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
//  Sora Operations
//===----------------------------------------------------------------------===//

//===- Constants ----------------------------------------------------------===//

class ConstantOp<string mnemonic, list<OpTrait> traits = []> :
    Sora_Op<mnemonic, traits> {
  let assemblyFormat = "$value attr-dict `->` type(results)";
}

def IntegerConstantOp : ConstantOp<"integer_constant", [NoSideEffect]> {
  let summary = "integer constant operation";
  let description = [{
    The Integer Constant operation turns an integer literal into an SSA value. 
    The data is attached to the operation as an attribute, and the type of
    the attribute must match the result type of the operation.

    Example:
      %0 = "sora.integer_constant"() { value = 0 : si64 } : () -> si64
  }];

  // FIXME: Constraint the return type and argument type to be the same.

  let arguments = (ins APIntAttr:$value);

  let results = (outs AnyInteger);

  let builders = [
    /// Shorthand builder that just takes an APInt and a Type.
    OpBuilder<"Builder *builder, OperationState &state, APInt value, "
              "Type type", [{
      build(builder, state, type, IntegerAttr::get(type, value));
    }]>,
  ];

  let verifier = [{ return ::verify(*this); }];
}

def BoolConstantOp : ConstantOp<"bool_constant", [NoSideEffect]> {
  let summary = "boolean constant operation";
  let description = [{
    The Bool Constant operation turns a boolean literal into an SSA value. 
    The data is attached to the operation as an attribute, and the return
    type of the operation is always I1.

    Example:
      %0 = "sora.bool_constant"() { value = true } : () -> i1
  }];

  let arguments = (ins BoolAttr:$value);

  let results = (outs I1);

  let builders = [
    /// Shorthand builder that just takes a bool
    OpBuilder<"Builder *builder, OperationState &state, bool value", [{
      build(builder, state, builder->getI1Type(), builder->getBoolAttr(value));
    }]>,
  ];
}

//===- Casting ------------------------------------------------------------===//

def IntegerCastOp : Sora_Op<"integer_cast", [NoSideEffect]> {
  let summary = "integer casting operation";
  let description = [{
    The Integer Casting Operation converts an SSA value of Integer Type into
    another SSA value with a different integer type. 
    This can be a widening/narrowing conversion.

    Example:
      %1 = "sora.integer_cast"() %0 : () -> si64
  }];

  let arguments = (ins AnyInteger:$value);

  let results = (outs AnyInteger);

  let assemblyFormat = 
    "`(` $value `:` type($value) `)` attr-dict `to` type(results)";
}

//===----------------------------------------------------------------------===//

#endif // SORAIR_OPS
