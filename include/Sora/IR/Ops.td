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
//  Operations
//===----------------------------------------------------------------------===//

//===- Static Cast --------------------------------------------------------===//

def StaticCastOp : SoraOp<"static_cast", [NoSideEffect]> {
  let summary = "static casting operation";
  let description = [{
    The Static Cast Operation converts an SSA value of some type into
    another type.

    Example:
      %1 = sora.static_cast(%1 : i1) to i32
  }];

  let assemblyFormat = "$value attr-dict `->` type(results)";

  let arguments = (ins AnyType:$value);

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
      %1 = sora.load_lvalue(%1 : !sora.lvalue<i32>) to i32
  }];

  let assemblyFormat = "$value attr-dict `->` type(results)";

  let arguments = (ins LValueType:$value);

  let results = (outs AnyType);

  let verifier = [{
    return ::verify(*this);
  }];

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value lvalue", [{
      return ::build(builder, result, lvalue);
  }]>];

  let assemblyFormat = 
    "`(` $value `:` type($value) `)` attr-dict `to` type(results)";
}

//===----------------------------------------------------------------------===//

#endif // SORAIR_OPS
