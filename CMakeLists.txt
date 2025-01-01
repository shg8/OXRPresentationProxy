set(PROJECT_NAME OXRPresentationProxy)

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

cmake_minimum_required(VERSION 3.18)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FindVulkan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create Python extension module
add_library(OXRPresentationPython MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/python_extension/PythonBindings.cpp
)

# Configure the Python extension
set_target_properties(OXRPresentationPython PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# Link dependencies
target_link_libraries(OXRPresentationPython PRIVATE
    pybind11::module
    ${PROJECT_NAME}  # Link against main project
    CUDA::cudart
)

add_subdirectory(external)
add_subdirectory(src)