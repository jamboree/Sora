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
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "sora";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "::mlir::sora";
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Sora_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Sora_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
//  Sora Operations
//===----------------------------------------------------------------------===//

def IntegerConstantOp : Sora_Op<"integer_constant", [NoSideEffect]> {
  let arguments = (ins APIntAttr:$value);

  let results = (outs AnyInteger);

  let builders = [
    /// Shorthand builder that just takes an APInt and a Type.
    OpBuilder<"Builder *builder, OperationState &state, APInt value, "
              "Type type", [{
      build(builder, state, type, IntegerAttr::get(type, value));
    }]>,

  ];
}

def BoolConstantOp : Sora_Op<"bool_constant", [NoSideEffect]> {
  let arguments = (ins BoolAttr:$value);

  let results = (outs I1);

  let builders = [
    /// Shorthand builder that just takes a bool and a type
    /// FIXME: Stop taking a type - just use I1!
    OpBuilder<"Builder *builder, OperationState &state, bool value", [{
      build(builder, state, builder->getI1Type(), builder->getBoolAttr(value));
    }]>,
  ];
}

#endif // SORAIR_OPS
