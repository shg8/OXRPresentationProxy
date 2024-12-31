set(PROJECT_NAME openxr-vulkan-example)

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

cmake_minimum_required(VERSION 3.18)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

include(FindVulkan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(external)
add_subdirectory(src)