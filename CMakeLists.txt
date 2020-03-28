message("CMake version is '${CMAKE_VERSION}'")
cmake_minimum_required(VERSION 3.4.3)

# set project name
project(Sora)

# set minimal C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

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
set(LLVM_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/llvm-tblgen")
set(MLIR_TABLEGEN_EXE "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen")

# include relevant LLVM/MLIR scripts
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(TableGen)

# add include paths 
include_directories(include)                              # lib includes
include_directories(${LLVM_INCLUDE_DIRS})                 # llvm includes 
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)  # tablegen'd stuff

# Disable RTTI, exceptions and enable /Wall
# FIXME: Only add the flags if we don't have them already
# FIXME: Is disabling exception handling needed or is D_HAS_EXCEPTIONS 0 above enough?
# FIXME: DRY the code a bit
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang" AND "x${CMAKE_CXX_SIMULATE_ID}" STREQUAL "xMSVC")
  add_definitions("/clang:-Wall")
  add_definitions("/GR-")
  # FIXME: Disable exception handling as well?
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_definitions("/GR-")
    # FIXME: Disable exception handling as well?
else() #FIXME: Is this good enough for all other compilers?
  add_definitions("-fno-rtti")
  add_definitions("-fno-exceptions") #FIXME: Is it needed?
  add_definitions("-Wall")
endif()

# add unittests, lib and include
#   unittests: create unit test target
#   lib: creates libsora_src
#   include: creates tablegen targets.
add_subdirectory(unittests)
add_subdirectory(lib)
add_subdirectory(include/Sora)

option(LIBSORA_AS_SHARED_LIB "Whether to build libsora as a shared library" OFF)

if(${LIBSORA_AS_SHARED_LIB}) 
  set(LIBSORA_LIB_KIND "SHARED")
else()
  set(LIBSORA_LIB_KIND "STATIC")
endif()

# create lib target, make it depend on tablegen'd stuff and link it with LLVM libs.
add_library(libsora ${LIBSORA_LIB_KIND} ${libsora_src})
set_target_properties(libsora PROPERTIES PREFIX "")

add_dependencies(libsora DriverOptionsTableGen SoraIROpsGen)

# get the LLVM Libraries that we need.
llvm_map_components_to_libnames(libsora_llvm_libs support option)
# add MLIR libraries.
list(APPEND libsora_llvm_libs MLIRIR MLIRLLVMIR MLIRStandardOps)

target_link_libraries(libsora ${libsora_llvm_libs})

add_subdirectory(tools/sorac)

# Add a check-sora target
set(LLVM_LIT_PATH "lit" CACHE STRING "Path to llvm-lit")

add_custom_target(check-sorac
        COMMAND ${LLVM_LIT_PATH} ${CMAKE_SOURCE_DIR}/tests -v -D sora_bin_dir=${CMAKE_BINARY_DIR}/bin
        DEPENDS sorac
)