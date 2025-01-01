set(TARGET_NAME OXRPresentationProxy)

set(SRC
        Main.cpp

        Context.cpp
        Context.h

        Controllers.cpp
        Controllers.h

        DataBuffer.cpp
        DataBuffer.h

        Headset.cpp
        Headset.h

        ImageBuffer.cpp
        ImageBuffer.h

        MeshData.cpp
        MeshData.h

        MirrorView.cpp
        MirrorView.h

        Model.h

        Pipeline.cpp
        Pipeline.h

        Renderer.cpp
        Renderer.h

        RenderProcess.cpp
        RenderProcess.h

        RenderTarget.cpp
        RenderTarget.h

        Util.cpp
        Util.h

        CudaInterop.cpp
        CudaInterop.h

        CudaTestPattern.h
)

set(CUDA_SRC
        CudaTestPattern.cu
)

# Set CUDA architectures if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(${TARGET_NAME})
target_sources(${TARGET_NAME} PRIVATE ${SRC} ${CUDA_SRC})
target_include_directories(${TARGET_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE boxer glfw glm openxr tinyobjloader ${Vulkan_LIBRARIES})
target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cudart)

target_compile_definitions(${TARGET_NAME} PRIVATE $<$<CONFIG:Debug>:DEBUG>) # Add a clean DEBUG prepocessor define if applicable
set_target_properties(${TARGET_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${TARGET_NAME}>") # For MSVC debugging

# Copy shared library binaries on Windows
if (WIN32)
    set(DLLS "${CMAKE_SOURCE_DIR}/external/glfw/lib/win/glfw3.dll")
    foreach (DLL ${DLLS})
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${DLL} "$<TARGET_FILE_DIR:${TARGET_NAME}>")
    endforeach ()
endif ()