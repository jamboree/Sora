## Sora Language Specification

Note: This is a rough draft. Nothing is definitive yet.

TO-DOs:
  - Grammar/Spellchecking
  - Explain `let` / `let mut` & patterns.
  - Explain basic types
  - Explain syntax
  - Explain expressions & literals
  - Name binding/Shadowing?
  - Write about other types: int, double, char, bool, string, arrays, slices, tuples.
    - What about strings? I need to decide on their semantics. Should they be built-in or a stdlib feature?
  - Write about LValues
  
### Syntax

Sora does not require, nor allows semicolons as line terminators.

TODO: Explain basic syntax, decls, patterns etc.

### Types

#### Primitive Types

##### Integer Types
- `i8`, `i16`, `i32` and `i64` (signed 8/16/32/64 bits integers)
- `u8`, `u16`, `u32` and `u64` (unsigned 8/16/32/64 bits integers)
- `isize` and `usize` pointer-sized (platform-dependent) signed/unsigned integers.

`i64` is the preferred type for literals. *Note: is this correct?*

There is currently no simple `int` or `unsigned` type, like in C/C++. Width of integers must be explicit.

##### Floating-Point Types
Both `f32` and `f64` are available. Both are represented using IEEE-754; with `f32` being single-precision and `f64` double-precision.

`f64` is the preferred type for literals, as it's more precise. *Note: is this correct?*

##### Boolean
`bool` is the boolean type. It has 2 values, `true` and `false`.

##### Char
TBD: The exact semantics of the `char` type still have to be determined, as it depends on the semantics of the `string` type.

##### String
TBD: The exact semantics of the `string` type still have to be determined.

The current idea is to represent them as UTF-8 strings, but things such as slicing and iterating over the string need to be taken into account.

##### Void
The `void` type is used as the return type of functions that don't return anything. You cannot declare a variable with a `void` type as it doesn't have a size.

#### Arrays and Slices

Sora offers fixed-size arrays: `[T; n]` where T is the type and n is the integer constant for the size. (Currently, only integer literals are planned, but support for compile-time expressions is also planned). Please note that the size of the array is part of its type.

Having the array size as part of the type of the array can be annoying. Sometimes, you just want a reference to an array (or some part of it), and that's exactly what slices are for. A slice can be declared like this:

```
let a: &[T]       // immutable reference slice
let b: &mut [T]   // mutable reference slice
let c: *[T]       // immutable pointer slice
let d: *mut [T]   // mutable pointer slice
```

A slice is a fat pointer: it has an extra `usize` attached to store the size of the slice. The size is an immutable field of the slice,  you can access it by using `.size`.

```
let size_of_a = a.size
```

#### Tuples

The "tuple" type is a structural type. You can create a tuple type by grouping 2 or more types in parentheses, and you can create
a tuple value by grouping 2 or more expressions in parentheses.
```
let a: (i32, i64) = (0, 1)
```
Note that I say "2 or more", because:
- 0-element tuple types are equal to the `void` type.
  - `func foo() -> ()` is equal to `func foo() -> void` (which is equal to `func foo()`)
- 1-element tuple types are equal to the type of the element. 
  - `let x: (i32) = 0`, `let x: i32 = (0)`, `let x: (i32) = ((0))` are the exact same declarations.
#### Structs
TODO

#### LValues and RValues
Sora has a concept of LValues and RValues, much like C/C++. However, they're not really "first-class", you can't interact with them directly. What you need to know is that they exist and that they matter.

An LValue is a value that you can assign to. It comes from "left-value", as in something that can appear on the LHS of an assignement.

An RValue is, in Sora, everything that isn't an LValue (so most expressions). You guessed it, it comes from "right-value", as in something that can appear on the RHS of an assignement. 
Note that, in some way, LValues are also RValues. They can also appear on the RHS of an assignement!

#### Reference and Pointers
Sora has both a reference type (declared using `&` or `&mut`) and pointer type (declared using `*` or `*mut`). Reference and pointers are similar (they're just like C pointers), but pointers are nullable whereas references are not.
Pointers can implicitly become references when the can prove they won't be null. This can happen inside a condition for example.

From now on, we'll call a context in which a given pointer can become a reference a "safe context".

You can't dereference a pointer outside a safe context.

You can access members of structs through both pointers and references using the arrow syntax.
Accessing a member of a struct through a reference/pointer using `->` is equivalent to doing `(*value).member`, thus, you can't use `->` on a pointer outside a safe context.

This might seem a bit complicated, but this example should clear things up.

```
let x: *T = getT()
if x {                // same as x != null
                      // x is known to be non-null; the body of this condition is a safe context.
  let y: &T = x       // x implicitly becomes &T
  x->foo()            // can access members
  let value: T = (*)  // can dereference
}
                    // x could be null; this is an unsafe context.
let y: &T = x       // error: can't convert *T into &T
x->foo()            // error: can't access members
let value: T = (*)  // error: can't dereference
```

Pointers and reference can both be mutable and immutable, for instance `*T` is a pointer to an *immutable* instance of T, while `&mut T` is a reference to a *mutable* instance of T.

```
let a: &T       // reference to an immutable instance of T
let b: &mut T   // reference to a mutable instance of T
let c: *T       // pointer to an immutable instance of T
let d: *mut T   // pointer to a mutable instance of T
```

In order to assign to a pointed value, you must dereference it first. That way, it is clear if you're assigning to the *reference* or to the *pointed object*. Dereferencing a pointer or a reference produces an LValue only if the reference/pointer is mutable.

```
let a: &mut T = foo()
let mub b: &mut T = bar()
b = a      // a and b now point to the same object, since you're assigning to the reference itself.
b = T()    // error, b has type 'mut &mut T', and a has type '&T'
(*a) = T() // The instance of T that a and b points to has been replaced by T().
```

### Immutability and Member Access

There's 2 kinds of member access in Sora.

 - Member access through a mutable source (lvalue)
    - In this case, you 'see' the members as they are declared
      ```
        struct S {                  // let mut s = S()
          let a: int	              // s.a has type int           (rvalue)
          let mut b: int            // s.b has type int           (lvalue)
          let c: &int               // s.c has type &int          (rvalue)
          let d: &mut int           // s.d has type &mut int      (rvalue)
          let mut e: &int           // s.e has type &int          (lvalue)
          let mut f: &mut int       // s.f has type &mut int      (lvalue)
          let mut g: &mut &mut int  // s.g has type &mut &mut int (lvalue)
        }
      ```
 - Member access through an immutable source (rvalue)
    - In this case, you 'see' the members with mutability stripped on all levels. `&mut T` becomes `&T`, variables declared as mutable are seen as immutable (lvalues become rvalues), etc.
      ```
        struct S {                  // let mut s = S()
          let a: int	              // s.a has type int   (rvalue)
          let mut b: int            // s.b has type int   (rvalue)
          let c: &int               // s.c has type &int  (rvalue)
          let d: &mut int           // s.d has type &int  (rvalue)
          let mut e: &int           // s.e has type &int  (rvalue)
          let mut f: &mut int       // s.f has type &int  (rvalue)
          let mut g: &mut &mut int  // s.g has type &&int (rvalue)
        }
      ``` 

This allows Sora to provides strong, yet simple immutability guarantees *in the common case*.

### Operators

#### Precedence Rules
| Precedence | Operators                                                                         | Associativity |
|------------|-----------------------------------------------------------------------------------|---------------|
| 1          | Function call: `()`<br> Array Subscript: `[]`<br>Member access: `.`, `->`         | left-to-right |
| 2          | Prefix Unary Operators: `+`, `-`, `!`, `*`, `&`, `~`                              | right-to-left |
| 3          | Multiplication `*`, Division `/` and Remainder `%`                                | left-to-right |
| 4          | Bitwise left shift `<<` and right shift `>>`                                      | left-to-right |
| 5          | Bitwise AND `&`, XOR `^` and OR <code>&#124;</code>                               | left-to-right |
| 6          | Addition `+` and Substraction `-`                                                 | left-to-right |
| 7          | Relational operators: `==`, `!=`, `<`, `<=`, `>` and `>=`                         | left-to-right |
| 8          | Logical AND `&&` and OR <code>&#124;&#124;</code>                                 | left-to-right |
| 9          | "Ternary" operator: `?:`                                                          | right-to-left |
| 10          | Assignement: `=`, `+=`, `-=`, `/=`, `*=`, `%=`, `&=`, <code>&#124;=</code>, `^=`, `<<=` and `>>=` | right-to-left |

These precedence rules are inspired by [C3's](https://c3lang.github.io/c3docs/precedence/).

#### Binary Operators
```
binary-operator = '+' | '-' | '/' | '*' | '%'
                | ">>" | "<<" | '&' | '|' | '^' 
                | "==" | "!=" | '<' | '>' | "<=" | ">="
                | "||" | "&&"
```
TODO: Add content (describe each operator)
#### Assignement Operators
```
assignement-operator = '=' | "+=" | "-=" | "/=" | "*=" | "%="
                     | ">>=" | "<<=" | "&=" | "|=" | "^=" 
```
TODO: Add content (describe each operator)
#### Prefix Unary Operators
`prefix-operator = '+' | '-' | '!' | '*' | '&'`

The unary plus (`+`) is pretty much a no-op, kept for consistency.

The unary minus (`-`) negates the value of the operand.

The logical not (`!`) operator inverts a boolean. (`!true == false && !false == true`)

The unary `*` is the dereference operator. It is used to access the contents of a reference or pointer (it transforms `&T` into `*T` for instance)
This operator only produces an lvalue if the reference, or pointer is `mut`. You cannot dereference a pointer outside of a safe context.

```
let a: &T       // *a has type 'T' (rvalue)
let b: &mut T   // *b has type 'T' (lvalue)
let c: *T       // *c has type 'T' (rvalue)
let d: *mut T   // *d has type 'T' (lvalue)
```

The unary `&` is the address-of operator, in other terms, use it to create a reference to something.
This operator always returns a reference type, and the mutability of the reference is determined by the mutability of
the value you're taking the address of. (rvalue = `&T`, lvalue = `&mut T`)

```
let x: int = 0
let mut y: int = 0

let ref_x = &x // ref_x has type &int, since x isn't mutable (it's an rvalue)
let ref_y = &y // ref_y has type &mut y, since y is mutable (it's an lvalue)
```
