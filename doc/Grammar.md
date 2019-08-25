## Sora Language Grammar.

TODOs: 
 - Identifiers should allow unicode.
 - Escaped stuff in strings (e.g. `\u0000`)
 - Enum, Union?
 - Binary/Hex literals

### Foreword
This is currently a work in progress. 

### The grammar

This is a normal BNF grammar, maybe there's a few changes here and there but since this grammar isn't used to automatically generate a parser it should be fine.
 - ? is used for elements can optionally appear
 - \* is used for elements that can appear 0, 1 or more times.
 - \+ is used for elements that can appear 1 or more times.
 - "" or '' are used for terminal strings. ('' for single characters, "" for multiple characters)

### Statement separation
Sora does not require semicolon, and doesn't allow semicolons. Statements are separated by newlines.

### Grammar

Misc.
```
// Lexical Structure
line-break = '\r'? '\n'
line-comment-item = any character except '\n' or '\r'
line-comment = "//" line-comment-item* line-break

block-comment-item = any character except "*/"
block-comment = "/*" block-comment-item* "*/"

unit-declaration = global-declaration+
global-declaration = function-declaration | type-declaration | struct-declaration

identifier-head = (uppercase or lowercase letter | "_")
identifier-body = (identifier-head | digit)
identifier = identifier-head identifier-body*

// Declarations
struct-declaration = "struct" identifier '{' struct-member-declaration+ '}'
struct-member-declaration = pattern-initializer

type-declaration = "type" identifier "=" type

function-declaration = "func" identifier parameter-declaration-list ("->" type)? compound-statement // note: if the return type isn't present, the function returns void.
parameter-declaration-list = '(' parameter-declaration (',' parameter-declaration)* ')'
parameter-declaration = "mut"? identifier type-annotation

let-declaration = "let" pattern-initializer

// Patterns
pattern = "mut"? (tuple-pattern | identifier-pattern | wildcard-pattern) type-annotation?
wildcard-pattern = '_'
identifier-pattern = identifier
tuple-pattern = '(' pattern (',' pattern)* ')'
pattern-initializer = pattern ('=' expression)?

// Statements 
statement = brace-statement
          | expression
          | if-statement
          | while-statement
          | for-statement
          | control-transfer-statement
          | declaration-statement
          
declaration-statement = function-declaration | type-declaration | struct-declaration | let-declaration

brace-statement = '{' statement* '}'

if-statement = "if" (expression | variable-declaration) brace-statement ("else" (brace-statement | if-statement))?

while-statement = "while" expression brace-statement

for-statement = "for" pattern "in" expression brace-statement

control-transfer-statement = continue-statement 
                           | break-statement
continue-statement = "continue"
break-statement = "break"

// Types
type = identifier
     | tuple-type
     | array-type
     | reference-or-pointer-type
     | maybe-type 
 
array-type = '[' type (';' expr)? ']'
tuple-type = '(' type (',' type)* ')'
reference-or-pointer-type = ('&' | '*') "mut"? type
maybe-type = "maybe" type

type-annotation = ':' type

// Expressions
binary-operator = '+' | '-' | '/' | '*' | '%'
                | ">>" | "<<" | '&' | '|' | '^' 
                | "==" | "!=" | '<' | '>' | "<=" | ">="
                | "||" | "&&"
assignement-operator = '=' | "+=" | "-=" | "/=" | "*=" | "%="
                     | ">>=" | "<<=" | "&=" | "|=" | "^=" 
prefix-operator = '+' | '-' | '!' | '*' | '&'

expression = assignement-expression
assignement-expression = conditional-expression (assignement-operator expression)?
conditional-expression = binary-expression ('?' expression ':' expression)
binary-expression = cast-expression (binary-operator cast-expression)*
cast-expression = prefix-expression ("as" type)*
prefix-expression = prefix-operator prefix-expression
                  | postfix-expression
postfix-expression = primary-expression suffix* 
suffix = tuple-expression // suffixes = calls, member accesses and subscripts.
       | member-access-expression
       | array-subscript
primary-expression = identifier | literal | tuple-expression | wildcard-expression
wildcard-expression = '_'
tuple-expression = '(' expression-list? ')'
expression-list = expression (',' expression)*
member-access-expression = ('.' | "->") (identifier | integer-literal)
array-subscript = '[' expression ']'

literal = null-literal | integer-literal | floating-point-literal | boolean-literal | string-literal | char-literal | array-literal
null-literal = "null"
integer-literal = digit+
digit = '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
floating-point-literal = integer-literal ('.' integer-literal)?
boolean-literal = "true" | "false"
array-literal = '[' expression-list? (';' expr) ']' // note: ';' expr is only allowed if there's one element
escape-sequence = '\' escaped-item
escaped-item = 'r' | 'n' | 't' | '0' | ''' | '"' | '\'

char-literal-item = any character except ''', '\n' or '\r'
                  | escape-sequence

char-literal = ''' char-literal-item '''

string-literal-item = any character except '"', '\n' or '\r'
                    | escape-sequence
string-literal = '"' string-literal-item* '"'
```
