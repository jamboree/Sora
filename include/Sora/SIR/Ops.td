//===--- Ops.td - Sora IR MLIR Dialect Op Definitions ------*- tablegen -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#ifndef SIR_OPS
#define SIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
//  Dialect
//===----------------------------------------------------------------------===//

def SIRDialect : Dialect {
  let name = "sir";

  let cppNamespace = "::mlir::sir";
}

class SIROp<string mnemonic, list<OpTrait> traits = []> :
    Op<SIRDialect, mnemonic, traits> {

  // Sora operations should provide their own builder.
  let skipDefaultBuilders = 1;  

  // Sora operations should always have a verifier.
  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
//  Types
//===----------------------------------------------------------------------===//

def SIRType : DialectType<SIRDialect,
                          CPred<"$_self.isa<::sora::sir::SIRType>()">,
                          "SIR Dialect Type">;

def AnyPointerType : DialectType<SIRDialect,
                                 CPred<"$_self.isa<::sora::sir::PointerType>()">,
                                 "Any Sora IR Pointer Type">;

//===- Static Cast --------------------------------------------------------===//

def StaticCastOp : SIROp<"static_cast", [NoSideEffect]> {
  let summary = "static casting operation";
  let description = [{
    The Static Cast operation converts an SSA value of some type into
    another type.

    Example:
      %1 = sir.static_cast %1 : i1 to i32
  }];

  let arguments = (ins AnyType: $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value value, Type toType", 
    [{ result.addOperands(value); result.addTypes(toType); }]
  >];

  let assemblyFormat = 
    "$value `:` type($value) attr-dict `to` type(results)";
}

//===- Load ---------------------------------------------------------------===//

def LoadOp : SIROp<"load", [NoSideEffect]> {
  let summary = "load operation";
  let description = [{
    The Load operation dereferences a pointer value, 
    converting !sir.pointer<T> into T.

    Example:
      %1 = sir.load %1 : (!sir.pointer<i32>) -> i32
  }];

  let arguments = (ins AnyPointerType: $value);

  let results = (outs AnyType);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value lvalue", [{
      return ::buildLoadOp(builder, result, lvalue);
  }]>];

  let assemblyFormat = 
    "$value attr-dict `:` functional-type($value, results)";
}

//===- Create Tuple -------------------------------------------------------===//

def CreateTupleOp : SIROp<"create_tuple", [NoSideEffect]> {
  let summary = "tuple creation operation";
  let description = [{
    The Create Tuple operation creates a tuple of values.

    Example:
      %2 = sir.create_tuple(%0: i32, %1: i32) -> tuple<i32, i32>
  }];

  let arguments = (ins Variadic<AnyType>: $values);

  let results = (outs AnyTuple);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, ArrayRef<Value> elts", [{
      return ::buildCreateTupleOp(builder, result, elts);
  }]>];

  let assemblyFormat = 
    "`(` $values `:` type($values) `)` attr-dict `->` type(results)";
}

//===- Destructure Tuple --------------------------------------------------===//

def DestructureTupleOp : SIROp<"destructure_tuple", [NoSideEffect]> {
  let summary = "tuple destructuring operation";
  let description = [{
    The Destructure Tuple operation breaks up a tuple into all of its elements.

    Example:
      %3, %4 = sir.destructure_tuple %2: (tuple<i32, i32>) -> (i32, i32)
  }];

  let arguments = (ins AnyTuple: $value);

  let results = (outs Variadic<AnyType>);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value tuple", [{
      return ::buildDestructureTupleOp(builder, result, tuple);
  }]>];

  let assemblyFormat = 
    "$value attr-dict `:` functional-type($value, results)";
}

//===- Implicit Terminator for Blocks -------------------------------------===//

def BlockTerminatorOp : SIROp<"block_terminator", 
                                   [NoSideEffect, Terminator]> {
  let summary = "block terminator operation";

  let description = [{
    The Block Terminator operation is a special terminator for SIR blocks.
    It does not have custom syntax, and is omitted when printing 
    the blocks for brevity.
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result", [{ }]
  >];

  let printer = ?;
  let parser = ?;
}

def ImplicitSIRBlockTerminator :
  SingleBlockImplicitTerminator<"BlockTerminatorOp">;

//===- BlockOp ------------------------------------------------------------===//

def BlockOp : SIROp<"block", [ImplicitSIRBlockTerminator]> {
  let summary = "block operation";

  let description = [{
    The Block operation serves as a container for free blocks. A free block
    is a block that does not belong to another construct, such as a condition,
    loop or function.

    This operations is always created with an entry block 
    containing a BlockTerminatorOp.

    Example:
      sir.block {
        
      }
  }];

  let regions = (region SizedRegion<1>:$region); 

  let printer = [{ return ::printBlockOp(p, *this); }];
  let parser = [{ return ::parseBlockOp(parser, result); }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result", 
      [{ ::buildBlockOp(builder, result); }]
  >];

  let extraClassDeclaration = [{
    mlir::Block &getEntryBlock() {
      return *region().begin();
    }
  }];
}

//===- Create Default Value -----------------------------------------------===//

def CreateDefaultValueOp : SIROp<"create_default_value", [NoSideEffect]> {
  let summary = "default value creation operation";
  let description = [{
    The Create Default Value operation creates a default value of the return
    type.

    Example:
      %1 = sir.create_default_value for i32
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Type ofType", 
    [{ result.addTypes(ofType); }]
  >];

  let results = (outs AnyType);

  let assemblyFormat = "attr-dict `for` type(results)";
}

//===----------------------------------------------------------------------===//

#endif // SIR_OPS
