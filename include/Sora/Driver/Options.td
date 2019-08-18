include "llvm/Option/OptParser.td"

def Test : Flag<["-"], "test">, 
  HelpText<"The Test option">;

// Help
def help : Flag<["-", "--"], "help">, 
  HelpText<"Display available options">;
def h : Flag<["-"], "h">, 
  Alias<help>;
