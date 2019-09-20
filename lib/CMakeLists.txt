add_subdirectory(AST)
add_subdirectory(Common)
add_subdirectory(Driver)
add_subdirectory(Lexer)
add_subdirectory(Parser)

set(libsora_src
  ${ast_src}
  ${common_src}
  ${driver_src}
  ${lexer_src}
  ${parser_src}
  PARENT_SCOPE
)