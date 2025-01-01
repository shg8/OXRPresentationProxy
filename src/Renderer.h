#pragma once

#include <glm/fwd.hpp>

#include <vulkan/vulkan.h>

#include <queue>
#include <vector>

#include "CudaInterop.h"

class Context;
class DataBuffer;
class Headset;
class RenderProcess;

#define EYE_COUNT 2

class Renderer final {
public:
    Renderer(const Context* context, const Headset* headset);
    ~Renderer();

    void record(size_t swapchainImageIndex);
    void submit(bool useSemaphores) const;

    VkCommandBuffer getCurrentCommandBuffer() const;
    VkSemaphore getCurrentDrawableSemaphore() const;
    VkSemaphore getCurrentPresentableSemaphore() const;

    void transferToSwapchain(VkCommandBuffer cmd, int bufferPoolIndex, int swapchainImageIndex);

    // Generate test patterns in the offscreen images
    void generateTestPatterns();

    enum BufferPoolImageStatus {
        FREE,
        AVAILABLE,
        IN_FLIGHT,
    };

    std::vector<std::array<cudainterop::CudaVulkanImage, EYE_COUNT>> offscreenImages;

    const std::array<cudainterop::CudaVulkanImage, EYE_COUNT>& getOffscreenImages() const { return offscreenImages.at(currentRenderProcessIndex); }

private:
    const Context* context = nullptr;
    const Headset* headset = nullptr;

    VkCommandPool commandPool = nullptr;
    std::vector<RenderProcess*> renderProcesses;
    size_t currentRenderProcessIndex = 0u;
    uint32_t frameCounter = 0u;
};