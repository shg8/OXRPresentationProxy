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

    enum BufferPoolImageStatus {
        FREE,
        AVAILABLE,
        IN_FLIGHT,
    };

    std::vector<std::array<cudainterop::CudaVulkanImage, EYE_COUNT>> offscreenImages;

private:
    const Context* context = nullptr;
    const Headset* headset = nullptr;

    VkCommandPool commandPool = nullptr;
    std::vector<RenderProcess*> renderProcesses;
    size_t currentRenderProcessIndex = 0u;
};