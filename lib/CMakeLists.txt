add_subdirectory(AST)
add_subdirectory(Common)
add_subdirectory(Diagnostics)
add_subdirectory(Driver)
add_subdirectory(Lexer)
add_subdirectory(Parser)
add_subdirectory(Sema)

set(libsora_src
  ${ast_src}
  ${common_src}
  ${diagnostics_src}
  ${driver_src}
  ${lexer_src}
  ${parser_src}
  ${sema_src}
  PARENT_SCOPE
)