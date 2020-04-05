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
def Sora_Dialect : Dialect {
  // The namespace of our dialect
  let name = "sora";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::mlir::sora";
}

// Base class for Sora Dialect Operations 
class Sora_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Sora_Dialect, mnemonic, traits>;

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

// Sora IR Type
def Sora_Type : DialectType<Sora_Dialect,
                            CPred<"$_self.isa<::mlir::sora::SoraType>()">,
                            "Sora dialect type">;

//===----------------------------------------------------------------------===//
//  Operations
//===----------------------------------------------------------------------===//

//===- Casting ------------------------------------------------------------===//

// A static cast operation.
def StaticCastOp : Sora_Op<"static_cast", [NoSideEffect]> {
  let summary = "static casting operation";
  let description = [{
    The Static Cast Operation converts an SSA value of some type into
    another type.

    Example:
      %1 = sora.static_cast(%1 : i1) to si32
  }];

  let assemblyFormat = "$value attr-dict `->` type(results)";

  let arguments = (ins AnyType:$value);

  let results = (outs AnyType);

  let assemblyFormat = 
    "`(` $value `:` type($value) `)` attr-dict `to` type(results)";
}

//===----------------------------------------------------------------------===//

#endif // SORAIR_OPS
