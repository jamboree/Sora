# Sora Language

The Sora project aims to create a compiled programming language with a "trust the programmer" kind of mindset but with a modern
syntax and some features such as non-nullable references, built-in optional types, pattern matching and type inference.

**Note:** This is a personal project that I do for learning purposes. It is not intended for widespread or production use.

### Learning Goals
  - Learn more about SSA/Basic-Block Intermediate Representations
  - Learn more about LLVM IR generation
  - Learn more about compilers (their internals and the compilation process) and compiler runtimes.
  
### Status 
Currently, Sora is still in its infancy. Nothing is really working yet.

#### Implementation Roadmap
Note: For the actual language features roadmap see [doc/roadmap](doc/Roadmap.md)

- :white_check_mark: Common (*Note: This module grows as needed. It's a constant work in progress. This only enumerates the most important parts of this module*)
  - :white_check_mark: Diagnostic Engine
  - :white_check_mark: Source Manager
- :white_check_mark: Lexer
- :white_check_mark: AST
- :white_check_mark: Parser
- :white_check_mark: Semantic Analysis (Sema)
- :hourglass: IR
- :hourglass: IR Generation (IRGen)
- :x: IR Optimization & Analysis (advanced & flow-sensitive semantic analysis) (IROpt)
- :x: IR Lowering to LLVM IR (LLVMGen)


