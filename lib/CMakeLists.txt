add_subdirectory(AST)
add_subdirectory(Common)
add_subdirectory(Driver)
add_subdirectory(Lexer)

set(libsora_src
  ${ast_src}
  ${common_src}
  ${driver_src}
  ${lexer_src}
  PARENT_SCOPE
)