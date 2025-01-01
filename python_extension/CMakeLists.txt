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

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create Python extension module
add_library(OXRPresentationPython MODULE
    ${CMAKE_CURRENT_SOURCE_DIR}/PythonBindings.cpp
    ${CMAKE_SOURCE_DIR}/src/Context.cpp
    ${CMAKE_SOURCE_DIR}/src/Controllers.cpp
    ${CMAKE_SOURCE_DIR}/src/CudaInterop.cpp
    ${CMAKE_SOURCE_DIR}/src/CudaTestPattern.cu
    ${CMAKE_SOURCE_DIR}/src/DataBuffer.cpp
    ${CMAKE_SOURCE_DIR}/src/Headset.cpp
    ${CMAKE_SOURCE_DIR}/src/ImageBuffer.cpp
    ${CMAKE_SOURCE_DIR}/src/MirrorView.cpp
    ${CMAKE_SOURCE_DIR}/src/Pipeline.cpp
    ${CMAKE_SOURCE_DIR}/src/RenderProcess.cpp
    ${CMAKE_SOURCE_DIR}/src/RenderTarget.cpp
    ${CMAKE_SOURCE_DIR}/src/Renderer.cpp
    ${CMAKE_SOURCE_DIR}/src/Util.cpp
)

# Configure the Python extension
set_target_properties(OXRPresentationPython PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_STANDARD 17  # PyTorch requires C++17
)

target_include_directories(OXRPresentationPython PRIVATE 
    ${Vulkan_INCLUDE_DIRS} 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src  # Add source directory to include path
)

# Link dependencies
target_link_libraries(OXRPresentationPython PRIVATE
    pybind11::module
    CUDA::cudart
    ${TORCH_LIBRARIES}
    ${Vulkan_LIBRARIES}
    glfw glm openxr boxer
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

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_ROOT}/lib")
target_link_libraries(OXRPresentationPython PRIVATE ${TORCH_PYTHON_LIBRARY})

# Copy shared library binaries on Windows
if (WIN32)
    set(DLLS "${CMAKE_SOURCE_DIR}/external/glfw/lib/win/glfw3.dll")
    foreach (DLL ${DLLS})
        add_custom_command(TARGET OXRPresentationPython POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${DLL} "$<TARGET_FILE_DIR:OXRPresentationPython>")
    endforeach ()
endif ()