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
  
  
### Factory methods (static `create`) vs `operator new`?
TODO
