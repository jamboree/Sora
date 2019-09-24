include "llvm/Option/OptParser.td"

def Test : Flag<["-"], "test">, 
  HelpText<"The Test option">;

// Help
def help : Flag<["-", "--"], "help">, 
  HelpText<"Display available options">;
def h : Flag<["-"], "h">, 
  Alias<help>;

// Parsing-related options
def dump_parse : Flag<["-"], "dump-parse">,
  HelpText<"Performs parsing and dumps the resulting AST">;
def parse_only : Flag<["-"], "parse-only">,
  HelpText<"Stops compilation after parsing">;

// Diagnostic Verification
def verify : Flag<["-"], "verify">,
  HelpText<"Enables diagnostic verification">;