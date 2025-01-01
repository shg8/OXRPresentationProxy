#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>

#include <windows.h>

class Device; // Forward-declare a structure that wraps a Vulkan image and memory allocated for CUDA interop.
class Context; // forward declaration to avoid including "Context.h"

namespace cudainterop {
struct CudaVulkanImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkExtent2D extent = {};
#ifdef _WIN32
    HANDLE memoryHandle = nullptr;
#else
    int memoryFd = -1; // File descriptor for POSIX systems
#endif
    bool valid = false;

    // CUDA-specific members
    cudaExternalMemory_t cudaExtMem = nullptr;
    cudaMipmappedArray_t cudaMipArray = nullptr;
    cudaArray_t cudaArray = nullptr;  // Level 0 of the mipmap array
    cudaSurfaceObject_t cudaSurface = 0;  // Surface object for writing to the array
};

// Creates a Vulkan image that can be imported/exported with CUDA.
// Returns a struct containing the Vulkan resources and a file descriptor
// that can be imported into CUDA.
CudaVulkanImage createCudaVulkanImage(const Context* context,
    VkExtent2D size,
    VkFormat format);

// Releases the Vulkan resources and closes the file descriptor.
void destroyCudaVulkanImage(const Context* context, CudaVulkanImage& image);

// Import and map the Vulkan memory into CUDA
bool importVulkanMemoryToCuda(CudaVulkanImage& image,
    VkFormat format,
    VkExtent2D extent,
    VkDeviceSize memorySize);

// Copy from a CUDA device pointer to a CudaVulkanImage
cudaError_t copyFromDevicePointerToCudaImage(CudaVulkanImage& image,
    const void* devicePtr,
    size_t devicePitch,
    VkExtent2D size);

} // namespace cudainterop