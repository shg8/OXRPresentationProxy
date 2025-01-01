#include "Renderer.h"

#include "Context.h"
#include "CudaInterop.h"
#include "DataBuffer.h"
#include "Headset.h"
#include "MeshData.h"
#include "Model.h"
#include "Pipeline.h"
#include "RenderProcess.h"
#include "RenderTarget.h"
#include "Util.h"

#include <array>
#include <stdexcept>

namespace {
constexpr size_t framesInFlightCount = 2u;
} // namespace

Renderer::Renderer(const Context* context, const Headset* headset)
    : context(context)
    , headset(headset)
{
    const VkDevice device = context->getVkDevice();

    // Create a command pool
    VkCommandPoolCreateInfo commandPoolCreateInfo { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = context->getVkDrawQueueFamilyIndex();
    if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        util::error(Error::GenericVulkan);
        throw std::runtime_error("Failed to create command pool");
    }

    // Create a render process for each frame in flight
    renderProcesses.resize(framesInFlightCount);
    for (RenderProcess*& renderProcess : renderProcesses) {
        renderProcess = new RenderProcess(context, commandPool);
        if (!renderProcess->isValid()) {
            throw std::runtime_error("Failed to create render process");
        }
    }

    offscreenImages.resize(framesInFlightCount);
    VkExtent2D stereoSize { 1920, 1080 };
    for (size_t bufferPoolIndex = 0u; bufferPoolIndex < framesInFlightCount; ++bufferPoolIndex) {
        for (size_t eyeIndex = 0u; eyeIndex < EYE_COUNT; ++eyeIndex) {
            offscreenImages.at(bufferPoolIndex).at(eyeIndex) = cudainterop::createCudaVulkanImage(context, stereoSize, VK_FORMAT_R8G8B8A8_UNORM);
            if (!offscreenImages.at(bufferPoolIndex).at(eyeIndex).valid) {
                util::error(Error::FeatureNotSupported, "Failed to create CUDA-Vulkan images.");
                throw std::runtime_error("Failed to create CUDA-Vulkan images.");
            }
        }
    }

    // Perform initial layout transitions
    VkCommandBuffer cmdBuffer;
    VkCommandBufferAllocateInfo allocInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer for initial layout transition");
    }

    VkCommandBufferBeginInfo beginInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuffer, &beginInfo);

    // Transition offscreen images to general layout
    for (auto& stereoImageSet : offscreenImages) {
        for (auto& image : stereoImageSet) {
            VkImageMemoryBarrier barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.image = image.image;
            barrier.subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1,  // mip levels
                0, 1   // array layers
            };

            vkCmdPipelineBarrier(cmdBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier);
        }
    }

    // Transition swapchain images to transfer dst optimal
    for (size_t i = 0; i < headset->getSwapchainImageCount(); i++) {
        const auto& swapchainImage = headset->getRenderTarget(i)->getImage();
        VkImageMemoryBarrier barrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.image = swapchainImage;
        barrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1,  // mip levels
            0, EYE_COUNT   // array layers
        };

        vkCmdPipelineBarrier(cmdBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier);
    }

    vkEndCommandBuffer(cmdBuffer);

    VkSubmitInfo submitInfo { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;

    VkFence fence;
    VkFenceCreateInfo fenceInfo { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence for initial layout transition");
    }

    if (vkQueueSubmit(context->getVkDrawQueue(), 1, &submitInfo, fence) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit initial layout transition command buffer");
    }

    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuffer);
}

Renderer::~Renderer()
{
    const VkDevice device = context->getVkDevice();

    for (const RenderProcess* renderProcess : renderProcesses) {
        delete renderProcess;
    }

    if (device && commandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    // Clean up our images
    for (auto& stereoImageSet : offscreenImages) {
        for (auto& image : stereoImageSet) {
            cudainterop::destroyCudaVulkanImage(context, image);
        }
    }
}

void Renderer::record(size_t swapchainImageIndex)
{
    currentRenderProcessIndex = (currentRenderProcessIndex + 1u) % renderProcesses.size();

    RenderProcess* renderProcess = renderProcesses.at(currentRenderProcessIndex);
    const VkDevice device = context->getVkDevice();
    const VkFence busyFence = renderProcess->getBusyFence();

    // Wait for the previous frame using this render process to complete
    if (vkWaitForFences(device, 1, &busyFence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
        return;
    }

    if (vkResetFences(device, 1u, &busyFence) != VK_SUCCESS) {
        return;
    }

    const VkCommandBuffer commandBuffer = renderProcess->getCommandBuffer();

    if (vkResetCommandBuffer(commandBuffer, 0u) != VK_SUCCESS) {
        return;
    }

    VkCommandBufferBeginInfo commandBufferBeginInfo { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS) {
        return;
    }

    transferToSwapchain(commandBuffer, currentRenderProcessIndex, swapchainImageIndex);
}

void Renderer::submit(bool useSemaphores) const
{
    const RenderProcess* renderProcess = renderProcesses.at(currentRenderProcessIndex);
    const VkCommandBuffer commandBuffer = renderProcess->getCommandBuffer();
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        return;
    }

    constexpr VkPipelineStageFlags waitStages = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    const VkSemaphore drawableSemaphore = renderProcess->getDrawableSemaphore();
    const VkSemaphore presentableSemaphore = renderProcess->getPresentableSemaphore();
    const VkFence busyFence = renderProcess->getBusyFence();

    VkSubmitInfo submitInfo { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1u;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (useSemaphores) {
        submitInfo.waitSemaphoreCount = 1u;
        submitInfo.pWaitSemaphores = &drawableSemaphore;
        submitInfo.signalSemaphoreCount = 1u;
        submitInfo.pSignalSemaphores = &presentableSemaphore;
    }

    if (vkQueueSubmit(context->getVkDrawQueue(), 1u, &submitInfo, busyFence) != VK_SUCCESS) {
        return;
    }
}

VkCommandBuffer Renderer::getCurrentCommandBuffer() const
{
    return renderProcesses.at(currentRenderProcessIndex)->getCommandBuffer();
}

VkSemaphore Renderer::getCurrentDrawableSemaphore() const
{
    return renderProcesses.at(currentRenderProcessIndex)->getDrawableSemaphore();
}

VkSemaphore Renderer::getCurrentPresentableSemaphore() const
{
    return renderProcesses.at(currentRenderProcessIndex)->getPresentableSemaphore();
}

void Renderer::transferToSwapchain(VkCommandBuffer cmd, int bufferPoolIndex, int swapchainImageIndex)
{
    const auto& stereoImageSet = offscreenImages.at(bufferPoolIndex);
    const auto& swapchainImage = headset->getRenderTarget(swapchainImageIndex)->getImage();

    for (size_t eyeIndex = 0u; eyeIndex < EYE_COUNT; ++eyeIndex) {
        const auto& eyeImage = stereoImageSet.at(eyeIndex);

        // Transition eyeImage.image to VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        VkImageMemoryBarrier srcBarrier { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        srcBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.image = eyeImage.image;
        srcBarrier.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1, // mip levels
            0, 1 // array layers
        };

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &srcBarrier);

        // Perform the blit operation
        VkImageBlit region {};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.layerCount = 1;
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.baseArrayLayer = static_cast<uint32_t>(eyeIndex);
        region.dstSubresource.layerCount = 1;

        region.srcOffsets[0] = { 0, 0, 0 };
        region.srcOffsets[1] = {
            static_cast<int32_t>(eyeImage.extent.width),
            static_cast<int32_t>(eyeImage.extent.height),
            1
        };
        region.dstOffsets[0] = { 0, 0, 0 };
        region.dstOffsets[1] = {
            static_cast<int32_t>(eyeImage.extent.width),
            static_cast<int32_t>(eyeImage.extent.height),
            1
        };

        vkCmdBlitImage(cmd,
            eyeImage.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &region, VK_FILTER_LINEAR);

        // Transition source image back to general layout
        srcBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        srcBarrier.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        srcBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        srcBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &srcBarrier);
    }
}