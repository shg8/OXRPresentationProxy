# Add pybind11
include(FetchContent)
set(PYBIND11_FINDPYTHON ON)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

set(PYTORCH_VERSION 2.6.0)
set(PYTORCH_CUDA_VERSION 12)

if(
    PYTORCH_VERSION VERSION_GREATER_EQUAL 2.5.0 AND
    PYTORCH_CUDA_VERSION VERSION_GREATER_EQUAL 12
)
    message(STATUS "PyTorch NVTX headers workaround: Yes")
    # only do this if nvToolsExt is not defined and CUDA::nvtx3 exists
    if(NOT TARGET CUDA::nvToolsExt AND TARGET CUDA::nvtx3)
        add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        # ensure that PyTorch is told to use NVTX3 headers
        target_compile_definitions(
            CUDA::nvToolsExt INTERFACE
            TORCH_CUDA_USE_NVTX3
        )
        target_link_libraries(CUDA::nvToolsExt INTERFACE CUDA::nvtx3)
    endif()
else()
    message(STATUS "PyTorch NVTX headers workaround: No")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Create Python extension module
pybind11_add_module(OXRPresentationPython 
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

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(OXRPresentationPython PRIVATE ${TORCH_PYTHON_LIBRARY})

target_compile_definitions(OXRPresentationPython PRIVATE $<$<CONFIG:Debug>:DEBUG>) # Add a clean DEBUG prepocessor define if applicable
set_target_properties(OXRPresentationPython PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:OXRPresentationPython>") # For MSVC debugging

install(TARGETS OXRPresentationPython DESTINATION .)