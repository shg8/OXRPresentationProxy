#include "CudaInterop.h"
#include "Context.h"
#include "Util.h" // for error handling

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cudainterop
{
    CudaVulkanImage createCudaVulkanImage(const Context* context,
                                          VkExtent2D size,
                                          VkFormat format)
    {
        CudaVulkanImage result;
        result.extent = size;
        result.valid = false;  // Will set to true only if everything succeeds
        result.memoryFd = -1;  // Initialize to invalid fd
        
        VkDevice device = context->getVkDevice();
        
        // 1. Create the image with external memory flags
        VkExternalMemoryImageCreateInfo extImageInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        extImageInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |  // For Linux
                                  VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;  // For Windows
        
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.pNext = &extImageInfo;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent = {size.width, size.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | 
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_STORAGE_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (vkCreateImage(device, &imageInfo, nullptr, &result.image) != VK_SUCCESS) {
            util::error(Error::GenericVulkan, "Failed to create image for CUDA interop");
            return result;
        }

        // 2. Get memory requirements and allocate with external memory flags
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, result.image, &memReqs);

        VkExportMemoryAllocateInfo exportAllocInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exportAllocInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
                                     VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.pNext = &exportAllocInfo;
        allocInfo.allocationSize = memReqs.size;

        // Find suitable memory type
        uint32_t memoryTypeIndex;
        VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (!util::findSuitableMemoryTypeIndex(context->getVkPhysicalDevice(), 
                                              memReqs,
                                              properties,
                                              memoryTypeIndex)) {
            util::error(Error::GenericVulkan, "Failed to find suitable memory type for CUDA interop");
            vkDestroyImage(device, result.image, nullptr);
            result.image = VK_NULL_HANDLE;
            return result;
        }
        allocInfo.memoryTypeIndex = memoryTypeIndex;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &result.deviceMemory) != VK_SUCCESS) {
            util::error(Error::GenericVulkan, "Failed to allocate memory for CUDA interop");
            vkDestroyImage(device, result.image, nullptr);
            result.image = VK_NULL_HANDLE;
            return result;
        }

        // 3. Get the platform-specific handle for the memory
#ifdef _WIN32
        VkMemoryGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        handleInfo.memory = result.deviceMemory;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        HANDLE handle = nullptr;
        auto vkGetMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
        if (!vkGetMemoryWin32HandleKHR) {
            util::error(Error::FeatureNotSupported, "vkGetMemoryWin32HandleKHR not available");
            vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            return result;
        }

        if (vkGetMemoryWin32HandleKHR(device, &handleInfo, &handle) != VK_SUCCESS) {
            util::error(Error::GenericVulkan, "Failed to get memory Win32 handle");
            vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            result.image = result.deviceMemory = VK_NULL_HANDLE;
            return result;
        }

        // Store the handle in the memoryFd field (it will be reinterpreted in importVulkanMemoryToCuda)
        result.memoryHandle = handle;
#else
        VkMemoryGetFdInfoKHR fdInfo{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        fdInfo.memory = result.deviceMemory;
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        auto vkGetMemoryFdKHR = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        if (!vkGetMemoryFdKHR) {
            util::error(Error::FeatureNotSupported, "vkGetMemoryFdKHR not available");
            vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            return result;
        }

        if (vkGetMemoryFdKHR(device, &fdInfo, &result.memoryFd) != VK_SUCCESS) {
            util::error(Error::GenericVulkan, "Failed to get memory file descriptor");
            vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            result.image = result.deviceMemory = VK_NULL_HANDLE;
            return result;
        }
#endif

        // 4. Bind the memory to the image
        if (vkBindImageMemory(device, result.image, result.deviceMemory, 0) != VK_SUCCESS) {
            util::error(Error::GenericVulkan, "Failed to bind memory for CUDA interop");
            close(result.memoryFd);  // Don't forget to close the fd on error
            vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            result.image = result.deviceMemory = VK_NULL_HANDLE;
            result.memoryFd = -1;
            return result;
        }

        // 5. Create image view
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = result.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY
        };
        viewInfo.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1,  // mip levels
            0, 1   // array layers
        };

        if (vkCreateImageView(device, &viewInfo, nullptr, &result.imageView) != VK_SUCCESS) {
            util::error(Error::GenericVulkan, "Failed to create image view for CUDA interop");
            close(result.memoryFd);
            vkFreeMemory(device, result.deviceMemory, nullptr);
            vkDestroyImage(device, result.image, nullptr);
            result.image = result.deviceMemory = VK_NULL_HANDLE;
            result.memoryFd = -1;
            return result;
        }

        // Everything succeeded
        result.valid = true;
        return result;
    }

    bool importVulkanMemoryToCuda(CudaVulkanImage& image, VkDeviceSize size)
    {
        if (!image.valid) {
            util::error(Error::GenericVulkan, "Invalid image");
            return false;
        }

        // Set up the external memory handle descriptor
        cudaExternalMemoryHandleDesc memHandleDesc = {};
#ifdef _WIN32
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memHandleDesc.handle.win32.handle = image.memoryHandle;
#else
        memHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memHandleDesc.handle.fd = image.memoryFd;
#endif
        memHandleDesc.size = size;
        memHandleDesc.flags = 0;

        // Import the external memory
        cudaError_t result = cudaImportExternalMemory(&image.cudaExtMem, &memHandleDesc);
        if (result != cudaSuccess) {
            util::error(Error::GenericVulkan, "Failed to import external memory to CUDA");
            return false;
        }

        // Map the external memory to a device pointer
        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.size = size;
        bufferDesc.flags = 0;

        result = cudaExternalMemoryGetMappedBuffer(&image.cudaDevPtr, 
                                                 image.cudaExtMem,
                                                 &bufferDesc);
        if (result != cudaSuccess) {
            util::error(Error::GenericVulkan, "Failed to map external memory to CUDA pointer");
            cudaDestroyExternalMemory(image.cudaExtMem);
            image.cudaExtMem = nullptr;
            return false;
        }

        return true;
    }

    VkSubresourceLayout getCudaVulkanImageSubresourceLayout(const Context* context, CudaVulkanImage& image)
    {
        VkSubresourceLayout layout;
        vkGetImageSubresourceLayout(context->getVkDevice(), image.image, &layout);
        return layout;
    }

    void destroyCudaVulkanImage(const Context* context, CudaVulkanImage& image)
    {
        if (!image.valid) return;
        
        // Clean up CUDA resources first
        if (image.cudaDevPtr) {
            cudaFree(image.cudaDevPtr);
            image.cudaDevPtr = nullptr;
        }
        if (image.cudaExtMem) {
            cudaDestroyExternalMemory(image.cudaExtMem);
            image.cudaExtMem = nullptr;
        }

        // Clean up Vulkan resources
        VkDevice device = context->getVkDevice();
        
        if (image.imageView) {
            vkDestroyImageView(device, image.imageView, nullptr);
            image.imageView = VK_NULL_HANDLE;
        }
        if (image.image) {
            vkDestroyImage(device, image.image, nullptr);
            image.image = VK_NULL_HANDLE;
        }
        if (image.deviceMemory) {
            vkFreeMemory(device, image.deviceMemory, nullptr);
            image.deviceMemory = VK_NULL_HANDLE;
        }

#ifdef _WIN32
        if (image.memoryFd != -1) {
            CloseHandle(reinterpret_cast<HANDLE>(image.memoryFd));
            image.memoryFd = -1;
        }
#else
        if (image.memoryFd >= 0) {
            close(image.memoryFd);
            image.memoryFd = -1;
        }
#endif
        
        image.valid = false;
    }

} 