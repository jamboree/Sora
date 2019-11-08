# Sora Language

The Sora project aims to create a compiled programming language with a "trust the programmer" kind of mindset but with a modern
syntax and some safeguards in place to make it less likely to accidently shoot yourself in the foot. (e.g. flow-typing)

Now, I know what you're thinking.
 - Language X already does this.
 - Language Y is better.
 - Why another language? We already have too many of them!

In short, you're probably asking yourself "Why are you making another language? Go contribute to X or Y!".

**The answer is simple:** This is a personal project that I do for *fun* and *to learn*. 
I know Sora will probably never go beyond the prototyping stage, and I'm *100% fine* with that. 
As long as I learn things building Sora, I'll feel that it was a worthwhile endeavour.
I also want to build things *all by myself* in order to truly understand how they work. 


### Learning Goals
Sora is mostly a learning project. It is not intended for widespread or production use.
  - Learn more about SSA/Basic-Block Intermediate Representations
  - Learn more about LLVM IR generation
  - Learn more about compilers (their internals and the compilation process) and compiler runtimes.
  

### Status 
Currently, Sora is still in its infancy. Nothing is working yet.

#### Implementation Roadmap
Note: For the actual language features roadmap see [doc/roadmap](doc/Roadmap.md)

- :white_check_mark: Common (*Note: This module grows as needed. It's a constant work in progress. This only enumerates the most important parts of this module*)
  - :white_check_mark: Diagnostic Engine
  - :white_check_mark: Source Manager
- :white_check_mark: Lexer
- :white_check_mark: AST
- :white_check_mark: Parser
- :hourglass: Semantic Analysis (Sema)
- :x: IR
- :x: IR Generation (IRGen)
- :x: IR Optimization & Analysis (advanced & flow-sensitive semantic analysis) (IROpt)
- :x: IR Lowering to LLVM IR (LLVMGen)


