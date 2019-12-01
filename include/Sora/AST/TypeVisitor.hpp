//===--- TypeVisitor.hpp - Type Visitor -------------------------*- C++ -*-===//
// Part of the Sora project, licensed under the MIT license.
// See LICENSE.txt in the project root for license information.
//
// Copyright (c) 2019 Pierre van Houtryve
//===----------------------------------------------------------------------===//

#pragma once

#include "Types.hpp"
#include "llvm/Support/ErrorHandling.h"
#include <utility>

namespace sora {
template <typename Derived, typename Rtr = void, typename... Args>
class TypeVisitor {
public:
  Rtr visit(Type type, Args... args) {
    assert(type && "Cannot be used on a null pointer");
    switch (type->getKind()) {
#define TYPE(ID, PARENT)                                                       \
  case TypeKind::ID:                                                           \
    return static_cast<Derived *>(this)->visit##ID##Type(                      \
        static_cast<ID##Type *>(type.getPtr()),                                \
        ::std::forward<Args>(args)...);
#include "Sora/AST/TypeNodes.def"
    }
    llvm_unreachable("Unknown node");
  }

#define VISIT_METHOD(NODE, PARENT)                                             \
  Rtr visit##NODE(NODE *node, Args... args) {                                  \
    return static_cast<Derived *>(this)->visit##PARENT(                        \
        node, ::std::forward<Args>(args)...);                                  \
  }
#define TYPE(KIND, PARENT) VISIT_METHOD(KIND##Type, PARENT)
#define ABSTRACT_TYPE(KIND, PARENT) VISIT_METHOD(KIND##Type, PARENT)
#include "Sora/AST/TypeNodes.def"
#undef VISIT_METHOD
};
} // namespace sora