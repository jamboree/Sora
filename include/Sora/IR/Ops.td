//===--- Ops.td - Sora MLIR Dialect Operations Definitions -*- tablegen -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#ifndef SORAIR_OPS
#define SORAIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

//===----------------------------------------------------------------------===//
//  Dialect
//===----------------------------------------------------------------------===//

// Provide a definition of the 'sora' dialect in the ODS framework so that we
// can define our operations.
def SoraDialect : Dialect {
  // The namespace of our dialect
  let name = "sora";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::mlir::sora";
}

// Base class for Sora Dialect Operations 
class SoraOp<string mnemonic, list<OpTrait> traits = []> :
    Op<SoraDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
//  Attributes
//===----------------------------------------------------------------------===//

def APFloatAttr : Attr<CPred<"$_self.isa<FloatAttr>()">,
                       "arbitrary float attribute"> {
  let storageType = [{ FloatAttr }];
  let returnType = [{ APFloat }];
}

//===----------------------------------------------------------------------===//
//  Types
//===----------------------------------------------------------------------===//

def SoraType : DialectType<SoraDialect,
                            CPred<"$_self.isa<::sora::ir::SoraType>()">,
                            "Sora dialect type">;

def LValueType : DialectType<SoraDialect,
                              CPred<"$_self.isa<::sora::ir::LValueType>()">,
                              "Sora LValue type">;

//===----------------------------------------------------------------------===//
//  Type-Casting operation
//===----------------------------------------------------------------------===//

//===- Static Cast --------------------------------------------------------===//

def StaticCastOp : SoraOp<"static_cast", [NoSideEffect]> {
  let summary = "static casting operation";
  let description = [{
    The Static Cast operation converts an SSA value of some type into
    another type.

    Example:
      %1 = sora.static_cast(%1 : i1) to i32
  }];

  let arguments = (ins AnyType: $value);

  let results = (outs AnyType);

  let assemblyFormat = 
    "`(` $value `:` type($value) `)` attr-dict `to` type(results)";
}

//===- Load LValue --------------------------------------------------------===//

def LoadLValueOp : SoraOp<"load_lvalue", [NoSideEffect]> {
  let summary = "lvalue loading operation";
  let description = [{
    The Load LValue operation loads an LValue, converting @lvalue T into T.

    Example:
      %1 = sora.load_lvalue(%1 : !sora.lvalue<i32>) -> i32
  }];

  let arguments = (ins LValueType: $value);

  let results = (outs AnyType);

  let verifier = [{
    return ::verifyLoadLValueOp(*this);
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value lvalue", [{
      return ::buildLoadLValueOp(builder, result, lvalue);
  }]>];

  let assemblyFormat = 
    "`(` $value `:` type($value) `)` attr-dict `->` type(results)";
}

//===----------------------------------------------------------------------===//
//  Tuple Manipulation Operations
//===----------------------------------------------------------------------===//

//===- Create Tuple -------------------------------------------------------===//

def CreateTupleOp : SoraOp<"create_tuple", [NoSideEffect]> {
  let summary = "tuple creation operation";
  let description = [{
    The Create Tuple operation creates a tuple of values.

    Example:
      %2 = sora.create_tuple(%0: i32, %1: i32) -> tuple<i32, i32>
  }];

  let arguments = (ins Variadic<AnyType>: $values);

  let results = (outs AnyTuple);

  let verifier = [{
    return ::verifyCreateTupleOp(*this);
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, ArrayRef<Value> elts", [{
      return ::buildCreateTupleOp(builder, result, elts);
  }]>];

  let assemblyFormat = 
    "`(` $values `:` type($values) `)` attr-dict `->` type(results)";
}

//===- Destructure Tuple --------------------------------------------------===//

def DestructureTupleOp : SoraOp<"destructure_tuple", [NoSideEffect]> {
  let summary = "tuple destructuring operation";
  let description = [{
    The Destructure Tuple operation breaks up a tuple into all of its elements.

    Example:
      %3, %4 = sora.destructure_tuple(%2: tuple<i32, i32>) -> (i32, i32)
  }];

  let arguments = (ins AnyTuple: $value);

  let results = (outs Variadic<AnyType>);

  let verifier = [{
    return ::verifyDestructureTupleOp(*this);
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value tuple", [{
      return ::buildDestructureTupleOp(builder, result, tuple);
  }]>];

  let assemblyFormat = 
    "`(` $value `:` type($value) `)` attr-dict `->` type(results)";
}

//===----------------------------------------------------------------------===//
//  Misc. Operations
//===----------------------------------------------------------------------===//

//===- Create Default Value -----------------------------------------------===//

def CreateDefaultValueOp : SoraOp<"create_default_value", [NoSideEffect]> {
  let summary = "default value creation operation";
  let description = [{
    The Create Default Value operation creates a default value of the return
    type.

    Example:
      %1 = sora.create_default_value for i32
  }];

  let results = (outs AnyType);

  let assemblyFormat = "attr-dict `for` type(results)";
}

//===----------------------------------------------------------------------===//

#endif // SORAIR_OPS
