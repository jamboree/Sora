## Sora Language Grammar.

TODOs: 
 - Identifiers should allow unicode.
 - Escaped stuff in strings (e.g. `\u0000`)
 - Enum, Union?
 - Binary/Hex literals

### Note
This is currently a work in progress. 

### The grammar

This is a normal BNF grammar, maybe there's a few changes here and there but since this grammar isn't used to automatically generate a parser it should be fine.
 - ? is used for elements can optionally appear
 - \* is used for elements that can appear 0, 1 or more times.
 - \+ is used for elements that can appear 1 or more times.
 - "" or '' are used for terminal strings. ('' for single characters, "" for multiple characters)

### Statement separation
Sora does not require semicolon, and doesn't allow semicolons. Consecutive statements are separated by line breaks.

### Grammar

```
// Lexical Structure
line-break = '\r'? '\n'
line-comment-item = any character except '\n' or '\r'
line-comment = "//" line-comment-item* line-break

block-comment-item = any character except "*/"
block-comment = "/*" block-comment-item* "*/"

identifier-head = (uppercase or lowercase letter | "_")
identifier-body = (identifier-head | digit)
identifier = identifier-head identifier-body*

// Declarations

source-file = top-level-declaration+
top-level-declaration = function-declaration | type-declaration | struct-declaration

struct-declaration = "struct" identifier '{' struct-member-declaration+ '}'
struct-member-declaration = pattern ('=' expression)?

type-declaration = "type" identifier "=" type

function-declaration = "func" identifier parameter-declaration-list ("->" type)? block-statement // note: if the return type isn't present, the function returns void.
parameter-declaration-list = '(' parameter-declaration (',' parameter-declaration)* ')'
                           | '(' ')'
parameter-declaration = identifier ':' type

let-declaration = "let" pattern ('=' expression)?

// Patterns
pattern = "mut" (tuple-pattern | identifier | '_') (':' type)?
tuple-pattern = '(' (pattern (',' pattern)*)? ')'
          
// Statements
block-statement = '{' block-statement-item* '}'

block-statement-item =
          | expression
          | block-statement
          | if-statement
          | while-statement
          | do-while-statement
          | for-statement
          | control-transfer-statement
          | function-declaration 
          | type-declaration 
          | struct-declaration 
          | let-declaration

if-statement = "if" condition block-statement ("else" (brace-statement | if-statement))?

while-statement = "while" condition block-statement

condition = expression | let-declaration

for-statement = "for" pattern "in" expression block-statement

control-transfer-statement = continue-statement 
                           | break-statement
                           | return-statement

continue-statement = "continue"
break-statement = "break"
return-statement = "return" expression?

// Types
type = identifier
     | tuple-type
     | array-type
     | reference-type
     | maybe-type 
 
array-type = '[' type (';' expr)? ']'
tuple-type = '(' (type (',' type)*)? ')'
reference-type = '&' "mut"? type
maybe-type = "maybe" type

// Expressions
binary-operator = '+' | '-' | '/' | '*' | '%'
                | ">>" | "<<" | '&' | '|' | '^' 
                | "==" | "!=" | '<' | '>' | "<=" | ">="
                | "||" | "&&" | '??'
assignement-operator = '=' | "+=" | "-=" | "/=" | "*=" | "%="
                     | ">>=" | "<<=" | "&=" | "|=" | "^=" | '??='
prefix-operator = '+' | '-' | '!' | '*' | '&' | '~'
postfix-operator = '!'

expression = assignement-expression
assignement-expression = conditional-expression (assignement-operator assignement-expression)?
conditional-expression = binary-expression ('?' expression ':' conditional-expression)?
binary-expression = cast-expression (binary-operator cast-expression)*
cast-expression = prefix-expression ("as" type)*
prefix-expression = prefix-operator prefix-expression
                  | postfix-expression
postfix-expression = primary-expression suffix* 
suffix = tuple-expression // suffixes = calls, member accesses and subscripts.
       | member-access-expression
       | array-subscript
       | postfix-operator
primary-expression = identifier | literal | tuple-expression | discard-expression
discard-expression = '_'
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
