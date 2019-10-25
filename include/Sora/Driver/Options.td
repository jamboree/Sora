include "llvm/Option/OptParser.td"

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

// Debugging/Others
def print_mem_usage : Flag<["-"], "print-memory-usage">,
  HelpText<"Regularly print the memory usage of various compiler datastructures">;