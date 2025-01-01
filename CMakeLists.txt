set(PROJECT_NAME OXRPresentationProxy)

add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

cmake_minimum_required(VERSION 3.18)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

include(FindVulkan)

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)

if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
    message("libtorch workaround")
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    # ensure that PyTorch is told to use NVTX3 headers
    target_compile_definitions(
            CUDA::nvToolsExt INTERFACE
            TORCH_CUDA_USE_NVTX3
    )
    target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
endif()

find_package(Torch CONFIG)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create Python extension module
add_library(OXRPresentationPython MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/python_extension/PythonBindings.cpp
)

# Configure the Python extension
set_target_properties(OXRPresentationPython PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_STANDARD 17  # PyTorch requires C++17
)

target_include_directories(OXRPresentationPython PRIVATE ${Vulkan_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})

# Link dependencies
target_link_libraries(OXRPresentationPython PRIVATE
    pybind11::module
    ${PROJECT_NAME}  # Link against main project
    CUDA::cudart
    "${TORCH_LIBRARIES}"
    ${Vulkan_LIBRARIES}
    glfw glm openxr
)

# Copy PyTorch DLLs on Windows
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET OXRPresentationPython
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:OXRPresentationPython>)
endif (MSVC)

add_subdirectory(external)
add_subdirectory(src)