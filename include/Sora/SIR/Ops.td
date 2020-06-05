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
      %1 = sir.static_cast %1 : to i32
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

//===- SIRAllocOp ---------------------------------------------------------===//

class SIRAllocOp<string mnemonic, Resource resource, list<OpTrait> traits = []> :
    SIROp<mnemonic, !listconcat([MemoryEffects<[MemAlloc<resource>]>], traits)> {

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, ::sora::sir::PointerType type", 
    [{ result.addTypes(type); }]
  >];

  
  let extraClassDeclaration = [{
    ::sora::sir::PointerType getPointerType() { 
      return getResult().getType().cast<::sora::sir::PointerType>(); 
    }
  }];

  let results = (outs AnyPointerType);

  let assemblyFormat = "attr-dict `:` type(results)";
}
//===- Alloc Stack --------------------------------------------------------===//

def AllocStackOp : SIRAllocOp<"alloc_stack", AutomaticAllocationScopeResource> {
  let summary = "stack allocation operation";
  let description = [{
    The Alloc Stack operation allocates memory of the stack. The amount of
    memory allocated is specified by the pointee type of the pointer type.

    Example: Allocating 32 bits (i32):
      %0 = sir.alloc_stack: !sora.pointer<i32>
      
    Example: Allocating a pointer to a 32 bits value:
      %1 = sir.alloc_stack: !sora.pointer<!sora.pointer<i32>>
  }];
}

//===- Load ---------------------------------------------------------------===//

def LoadOp : SIROp<"load", [
  TypesMatchWith<"type of 'result' matches pointee type of 'pointer'", 
                  "pointer", "result",
                 "$_self.cast<::sora::sir::PointerType>().getPointeeType()">]> {

  let summary = "load operation";
  let description = [{
    The Load operation loads a value from a pointer, converting 
    !sir.pointer<T> into T.

    Example:
      %1 = sir.load %0 : !sir.pointer<i32>
  }];

  let arguments = (ins
    Arg<AnyPointerType, "the pointer to load from", [MemRead]>: $pointer
  );

  let results = (outs AnyType: $result);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value ptrValue", [{
    auto ptrType = ptrValue.getType().cast<::sora::sir::PointerType>();
    result.addTypes(ptrType.getPointeeType());
    result.addOperands(ptrValue);
  }]>];

  let extraClassDeclaration = [{
    ::sora::sir::PointerType getPointerType() { 
      return getOperand().getType().cast<::sora::sir::PointerType>();
    }
  }];

  let assemblyFormat = "$pointer attr-dict `:` type($pointer)";
}

//===- Store ---------------------------------------------------------------===//

def StoreOp : SIROp<"store", [
  TypesMatchWith<"type of 'value' matches pointee type of 'pointer'", 
                  "pointer", "value",
                 "$_self.cast<::sora::sir::PointerType>().getPointeeType()">]> {

  let summary = "store operation";
  let description = [{
    The Store operation stores a value to a location specified by a pointer.
    The type of the value must match the type of the pointer's pointee type.

    Example:
      %2 = sir.store %0, %1 : !sir.pointer<i32>
  }];

  let arguments = (ins
    AnyType: $value,
    Arg<AnyPointerType, "the pointer to load from", [MemWrite]>: $pointer
  );

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value value, Value pointer", [{
      result.addOperands(value);
      result.addOperands(pointer);
  }]>];

  let extraClassDeclaration = [{
    Value getPointer() {
      return getOperand(1);
    }

    Value getValue() {
      return getOperand(0);
    }

    ::sora::sir::PointerType getPointerType() { 
      return getPointer().getType().cast<::sora::sir::PointerType>();
    }
  }];

  let assemblyFormat = "$value `,` $pointer attr-dict `:` type($pointer)";
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

def BlockOp : SIROp<"block",
    [ImplicitSIRBlockTerminator, AutomaticAllocationScope]> {
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

  let printer = [{ return ::print(p, *this); }];
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
