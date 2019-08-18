# include GoogleTest.cmake. This will create the "googletest"
# target and define the googletest_includes variable.
include("${PROJECT_SOURCE_DIR}/thirdparty/GoogleTest.cmake")

# Add test sources to access variable ${tests_src}
add_subdirectory(src)

# Add executable, link lib & gtest
add_executable(sora_unittests ${unittest_src})
target_link_libraries(sora_unittests libsora googletest)

# add src as an include directory
include_directories(src)

# add GoogleTest include directory
include_directories(${googletest_includes})

# Add tests
add_test(NAME sora_unittests COMMAND sora_unittests)
