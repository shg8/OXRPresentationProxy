set(PROJECT_NAME OXRPresentationProxy)

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

cmake_minimum_required(VERSION 3.18)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

include(FindVulkan)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(python_extension)