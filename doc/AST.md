# Sora Compiler AST

### Introduction

This document is intended to highlight a few design points of the Sora AST. It is mostly intended for people that wish to understand the Sora AST, or who want to work on Sora.
(And also for me, so I don't forget things)!

NOTE: This is currently very incomplete.

### Major AST Hierachies

* Expr: represents expressions
  * Expressions can be implicit: an expression may be present in the tree but not *explicitely* in the source. (e.g. an Implicit Conversion)
  * e.g. `1+2`
* Stmt: represents statements
  * e.g. `if foo {}`
* Decl: represents declarations
  * e.g. `func foo() {}`
* Pattern: represents patterns
  * e.g. `(a, b) : (i32, i32)` in `let (a, b) : (i32, i32)`
* TypeReprs: represents type as they were written in the source.
  * This is different from Types, and they don't represents types *semantically*. They're only the *representations* of the types as they were written down by the user.
  * TypeReprs are not unique/interned, and they contain source-location informations.
* Types: represents types *semantically*
  * These are singletons. For instance, 2 uses of `i32` use the same instance of the type.
  * Types written by the user are created from TypeReprs during Semantic Analysis. The parser **does not** create types.
  * Types don't contain source-location information.
  
### Creation of AST nodes

AST nodes should be created using the overloaded `operator new`, and preferably never on the stack (unless they're created for testing purposes, then it's harmless, but *never* create nodes on the stack in the parser!). However, some nodes just can't use the `operator new` and/or must forbid stack allocation for some reasons. Then, they should make their constructor private and use a static creation method.

An AST node should use static factory methods + private constructors when one of the following criteria is met:
 * It uses `llvm::TrailingObject`
   * Because classes that use trailing objects must allocate extra memory, they need a special `create` method that allocates the extra memory needed.
 * It needs AST cleanups
   * Allocating AST nodes using the AST Context is the most common way of allocating AST nodes, but, as it's a arena-style allocator, memory is freed all at once and destructors are never called. Most nodes are fine with that, but some nodes may not be fine with that and need "cleanup" methods to be called in order to free the extra memory allocated. (e.g. if they use `SmallVector`, `SmallVector`'s destructor has to be called to free the memory it allocated). To achieve this, they'll usually call `ASTContext::addDestructorCleanup` in their constructor. However, if the node isn't allocated in the ASTContext (e.g. the node has been created for testing purposes), it can have disastrous consequences. With that in mind, it's better to prevent stack allocation of the node altogether by making its constructor private and providing a static `create` method.
 * It's a singleton or their creation needs to be controlled/restricted.
   * A prime example is the `Type` hierarchy. They're all singletons, so of course their constructors are made private and the only way of acquiring a type is to use their static `get` method.
