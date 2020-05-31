//===--- Ops.td - Sora MLIR Dialect Operations Definitions -*- tablegen -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#ifndef SORAIR_OPS
#define SORAIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def AnyPointerType : DialectType<SoraDialect,
                                 CPred<"$_self.isa<::sora::ir::PointerType>()">,
                                 "Any Sora Pointer type">;

//===----------------------------------------------------------------------===//
//  Type-Casting operation
//===----------------------------------------------------------------------===//

//===- Static Cast --------------------------------------------------------===//

def Sora_StaticCastOp : SoraOp<"static_cast", [NoSideEffect]> {
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

//===- Load ---------------------------------------------------------------===//

def Sora_LoadOp : SoraOp<"load", [NoSideEffect]> {
  let summary = "load operation";
  let description = [{
    The Load operation dereferences a pointer value, 
    converting !sora.pointer<T> into T.

    Example:
      %1 = sora.load(%1 : !sora.pointer<i32>) -> i32
  }];

  let arguments = (ins AnyPointerType: $value);

  let results = (outs AnyType);

  let verifier = [{
    return ::verifyLoadOp(*this);
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value lvalue", [{
      return ::buildLoadOp(builder, result, lvalue);
  }]>];

  let assemblyFormat = 
    "$value attr-dict `->` functional-type($value, results)";
}

//===----------------------------------------------------------------------===//
//  Tuple Manipulation Operations
//===----------------------------------------------------------------------===//

//===- Create Tuple -------------------------------------------------------===//

def Sora_CreateTupleOp : SoraOp<"create_tuple", [NoSideEffect]> {
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

def Sora_DestructureTupleOp : SoraOp<"destructure_tuple", [NoSideEffect]> {
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
//  Block Operations
//===----------------------------------------------------------------------===//

//===- Any Block ----------------------------------------------------------===//

class AnyBlock<string mnemonic, list<OpTrait> traits = []> 
  : SoraOp<mnemonic, traits> {
  let regions = (region AnyRegion:$region); 
}

//===- BlockOp ------------------------------------------------------------===//

def Sora_BlockOp : AnyBlock<"block"> {
  let summary = "block operation";

  let description = [{
    The Sora Block operation serves as a container for free blocks. A free block
    is a block that does not belong to another construct, such as a condition,
    loop or function.

    Example:
      sora.block {
        
      } loc(unknown)
  }];

  let printer = [{ return ::printAnyBlock(p, this->region(), "block"); }];
  let parser = [{ return ::parseAnyBlock(parser, result); }];
}

//===----------------------------------------------------------------------===//
//  Misc. Operations
//===----------------------------------------------------------------------===//

//===- Create Default Value -----------------------------------------------===//

def Sora_CreateDefaultValueOp : SoraOp<"create_default_value", [NoSideEffect]> {
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
