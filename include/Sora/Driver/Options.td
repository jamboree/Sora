include "llvm/Option/OptParser.td"

// Help
def help : Flag<["-", "--"], "help">, 
  HelpText<"Display available options">;
def h : Flag<["-"], "h">, 
  Alias<help>;

// Basic options
def o : JoinedOrSeparate<["-"], "o">,
  HelpText<"Write output to <file>">, MetaVarName<"<file>">;

// Parsing-related options
def parse_only : Flag<["-"], "parse-only">,
  HelpText<"Stops compilation after parsing">;

// Semantic Analysis-related options
def sema_only : Flag<["-"], "sema-only">,
  HelpText<"Stops compilation after semantic analysis">;

// SIRGen-related options
def emit_sirgen : Flag<["-"], "emit-sirgen">,
  HelpText<"Stops after SIRGen and emits the resulting Sora IR">;
def dgb_g : Flag<["-"], "g">,
  HelpText<"Enable debug information">;
def dgb_g0 : Flag<["-"], "g0">,
  HelpText<"Disable debug information">;

// Diagnostic Verification
def verify : Flag<["-"], "verify">,
  HelpText<"Enables diagnostic verification">;

// Debugging
def print_mem_usage : Flag<["-"], "print-memory-usage">,
  HelpText<"Regularly print the memory usage of various compiler datastructures">;
def dump_parse : Flag<["-"], "dump-parse">,
  HelpText<"Dumps the raw AST after parsing">;
def dump_ast : Flag<["-"], "dump-ast">,
  HelpText<"Dumps the AST after typechecking">;
def dump_scope_maps : Separate<["-"], "dump-scope-maps">,
  HelpText<"Dumps the scope maps after typechecking, or"
    " after parsing if used in combination with -parse-only">,
  MetaVarName<"<expanded | lazy>">;

// LLVM
def Xllvm : Separate<["-"], "Xllvm">,
  Flags<[HelpHidden]>,
  MetaVarName<"<arg>">, HelpText<"Passes <arg> to the LLVM Option Parser">;