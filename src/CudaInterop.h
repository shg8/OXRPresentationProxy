#pragma once

#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Forward-declare a structure that wraps a Vulkan image and memory allocated for CUDA interop.
struct CudaVulkanImage
{
    VkImage         image         = VK_NULL_HANDLE;
    VkDeviceMemory  deviceMemory  = VK_NULL_HANDLE;
    VkImageView     imageView     = VK_NULL_HANDLE;
    VkExtent2D      extent        = {};
#ifdef _WIN32
    HANDLE          memoryHandle  = nullptr;
#else
    int            memoryFd      = -1;     // File descriptor for POSIX systems
#endif
    bool           valid         = false;

    // CUDA-specific members
    cudaExternalMemory_t cudaExtMem = nullptr;
    void*                cudaDevPtr  = nullptr;
};

class Context;  // forward declaration to avoid including "Context.h"

namespace cudainterop
{
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
                                 VkDeviceSize size);

    // Get the subresource layout for the Vulkan image
    VkSubresourceLayout getCudaVulkanImageSubresourceLayout(const Context* context, CudaVulkanImage& image);

} // namespace cudainterop 