message("CMake version is '${CMAKE_VERSION}'")
cmake_minimum_required(VERSION 3.4.3)

# set project name
project(Sora)

# set minimal C++ standard: we need C++17 at a minimum.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include the commonly used macros/helpers.
include(cmake/AddSource.cmake)  # defines the add_source macro

# set the preferred output directories:
#   Archives go in the /lib folder
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#   Libraries go in the /lib folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#   Executables go in the /bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# add LLVM 
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
add_definitions(${LLVM_DEFINITIONS})

# set _HAS_EXCEPTIONS to 0
add_definitions(-D_HAS_EXCEPTIONS=0)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM Tools binary dir: ${LLVM_TOOLS_BINARY_DIR}")

# set tablegen exe location (should be in ${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen)
SET(LLVM_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen")

# include relevant llvm scripts
include(AddLLVM)
include(HandleLLVMOptions)
include(TableGen)

# add include paths 
include_directories(include)                              # lib includes
include_directories(${LLVM_INCLUDE_DIRS})                 # llvm includes 
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)  # tablegen'd stuff

# find the libraries that we want to use
llvm_map_components_to_libnames(llvm_libs support option)

add_subdirectory(tools/CommandLine)

# add unittests, lib and include
#   unittests: create unit test target
#   lib: creates libsora_src
#   include: creates tablegen targets.
add_subdirectory(unittests)
add_subdirectory(lib)
add_subdirectory(include/Sora)

# create lib target, make it depend on tablegen'd stuff and link it with LLVM libs.
add_library(libsora STATIC ${libsora_src})
add_dependencies(libsora DriverOptionsTableGen)
target_link_libraries(libsora ${llvm_libs})