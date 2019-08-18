# Sora Roadmap

## Foreword

Every project I do is to learn something, and Sora is no exception. Learning how to implement and work with SSA/CFG IRs is my goal for this project, so I will have a SSA/CFG IR for Sora, IR that will be lowered to LLVM IR. 

As this is a complex topic, I prefer to start with a simplified version of Sora to get up and running quickly.

Also, this means that I can't guarantee that I'll go past part 1 of my plan. Maybe that after implementing part 1, I'll feel that Sora is a dead-end and that I've learned enough, and I may simply abandon the project. Who knows!

TL;DR: Stuff in Part 1 will *likely* be implemented, but after that I make no guarantee.

That said, let's begin.

## Part 1: The Simplified language

The first goal will be to implement a *heavily* simplified version of the language. This will be feature-poor and not capable of doing much, but it should be able to:
- Generate valid Sora IR
- Lower that IR to valid LLVM IR
- Run simple programs (factorials, fibonnaci, etc.) correctly

Here's a non-exhaustive list of features that will be present in the initial implementation.
- Types:
  - Primitive types: `bool`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` and `void`
  - Reference & Pointer types (`&T`, `&mut T`, `*T` and `*mut T`)
  - Tuple types
    - Tuples w/ 0 elements (`()`) will be canonicalized as `void`
    - Tuples w/ 1 element (e.g. `(0)`, of type `(i32)`) will be canonicalized as the type of the element (w/o the tuple, so `i32` in this case)
    - Tuples w/ 2+ elements will be canonicalized as a tuple of the canonicalized types of the elements
- Every statement except the `for` statement 
- Every expression except subscripts, array literals, char literals and string literals.
- Functions
  - No function types nor first-class functions either.
- Variables & Patterns (note: Sora will only support local variables at first, not because of complexity reasons, but because I want to try to do without them and see how it goes)
  - This includes basic pattern matching in variable declarations. e.g. `let (a, b) = (0, 1)`
- Basic C FFI
  - Semantics TBD (A basic one, so I can import a few C functions, e.g. a "print" function. It can even be built-in/hardcoded while waiting for a better solution)
    - Example syntax: `extern "C" { func malloc(size: usize) -> *mut void }`
- Struct

This phase is all about "making it work" & preparing for what's next.

## Part 2: Getting there

Now that we got a solid foundation, we can add the more advanced stuff. These will probably be added one-by-one, or in very small chunks (because most of the stuff listed here isn't trivial to implement and require some work to do right).

**Note:** Most of the features listed here are relatively vague. I don't have a real plan for their semantics yet. I just know that I may implement them someday.

- Array type, Slices, `array-literal` and `subscript-expression`
- `for ... in ...` statement
- Strings & Chars: `string`, `char` and slice types; `string-literal` and `char-literal`
  - Semantics of strings & string literals TBD
- Includes/Modules/Whatever: Allow the user to compile multiple files
  - Semantics TBD
- Begin writing a standard library
- Typealiases
- `maybe` type
- union/sum-type
  - Semantics and Syntax TBD. Ideally, I'd want both a tagged and untagged union type.
- Functions as first-class citizens: function type & closures.
  - Semantics TBD, but it's probably going to be straightforward (Closures with a Rust-like syntax)
- More advanced C interoperability
  - Semantics TBD (It should be a bit more advanced and capable of importing most C functions)
- (declaration) attributes
  - Syntax TBD, but it'll probably be a Rust-like syntax because it's not ambiguous, unlike C++'s `[[]]` which could cause ambiguities in a local context (with array literals)
- `public`, `private` and other access control stuff.
  - TBD. Do I even need it? Maybe yes, maybe not.

## Part 3: Endgame

**Note:** Ditto Part 2, but much worse. They're vague ideas, things I know I want but that take time to figure out and implement, and that may not even be implemented. Everything listed here is TBD, both for syntax and semantics. TL;DR: These are just ideas!

- Traits
- Generics
- Hygienic Macros
- Importing C/CPP headers through Clang (for the FFI)

