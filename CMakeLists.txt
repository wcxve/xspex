cmake_minimum_required(VERSION 3.19)

project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)
message(STATUS "Project name: ${SKBUILD_PROJECT_NAME}")
message(STATUS "Project version: ${SKBUILD_PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -march=native)

# Define install paths
set(INSTALL_PATH ${SKBUILD_PROJECT_NAME}/_compiled)
set(INSTALL_PATH_LIB ${INSTALL_PATH}/lib)
set(INSTALL_PATH_BIN ${INSTALL_PATH}/bin)

# Find Python
find_package(
    Python
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

# Find nanobind
find_package(nanobind CONFIG REQUIRED)

# Get XLA FFI include directory
execute_process(
    COMMAND ${Python_EXECUTABLE} build_helper.py print_xla_include_dir
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    OUTPUT_VARIABLE XLA_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# Get HEADAS include directory
execute_process(
    COMMAND ${Python_EXECUTABLE} build_helper.py print_headas_include_dir
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    OUTPUT_VARIABLE HEADAS_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "HEADAS include directory: ${HEADAS_INCLUDE_DIR}")

# Get HEADAS library directory
execute_process(
    COMMAND ${Python_EXECUTABLE} build_helper.py print_headas_lib_dir
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    OUTPUT_VARIABLE HEADAS_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "HEADAS library directory: ${HEADAS_LIB_DIR}")

# Function to find library in HEADAS library directory
function(find_headas_library name)
    execute_process(
        COMMAND ${Python_EXECUTABLE}
            build_helper.py print_headas_library_path ${name}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
        OUTPUT_VARIABLE LIB_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    message(STATUS "Found ${name}: ${LIB_PATH}")
    set(${name} ${LIB_PATH} PARENT_SCOPE)
endfunction()

# Find required libraries
find_headas_library(XSFunctions)
find_headas_library(XSUtil)
find_headas_library(XS)
find_headas_library(hdsp)
find_headas_library(CCfits)
find_headas_library(cfitsio)
set(LINK_LIBS ${XSFunctions} ${XSUtil} ${XS} ${hdsp} ${CCfits} ${cfitsio})

# Get XSPEC model list
execute_process(
    COMMAND ${Python_EXECUTABLE} build_helper.py print_xspec_models
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    OUTPUT_VARIABLE XSPEC_MODELS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "XSPEC models: ${XSPEC_MODELS}")

# Get XSPEC convolution model list
execute_process(
    COMMAND ${Python_EXECUTABLE} build_helper.py print_xspec_con_models
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
    OUTPUT_VARIABLE XSPEC_CON_MODELS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "XSPEC convolution models: ${XSPEC_CON_MODELS}")

# Define compile options interface
add_library(includes INTERFACE)
target_include_directories(
    includes
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
    $<BUILD_INTERFACE:${HEADAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${XLA_INCLUDE_DIR}>
)
# Define the XSPEC models macros for auto-generated function definitions
target_compile_definitions(includes INTERFACE
    "XSPEC_MODEL_LIST=${XSPEC_MODELS}"
    "XSPEC_CON_MODEL_LIST=${XSPEC_CON_MODELS}"
)

# Install the worker binary
add_executable(worker ${CMAKE_CURRENT_SOURCE_DIR}/src/worker.cpp)
target_link_libraries(worker PRIVATE includes ${LINK_LIBS})
set_target_properties(worker PROPERTIES INSTALL_RPATH ${HEADAS_LIB_DIR})
install(
    TARGETS worker
    RUNTIME
    DESTINATION ${INSTALL_PATH_BIN}
)

# Install the library and corresponding stub
nanobind_add_module(
    libxspex
    NB_STATIC
    STABLE_ABI
    LTO
    ${CMAKE_CURRENT_SOURCE_DIR}/src/libxspex.cpp
)
target_link_libraries(libxspex PRIVATE includes ${LINK_LIBS})
set_target_properties(libxspex PROPERTIES INSTALL_RPATH ${HEADAS_LIB_DIR})
install(
    TARGETS libxspex
    LIBRARY
    DESTINATION ${INSTALL_PATH_LIB}
)
nanobind_add_stub(
    libxspex_stub
    INSTALL_TIME
    MODULE libxspex
    OUTPUT ${INSTALL_PATH_LIB}/libxspex.pyi
    PYTHON_PATH ${INSTALL_PATH_LIB}
    VERBOSE
)
