#include "CudaTestPattern.h"
#include <cuda_runtime.h>

namespace {

__global__ void testPatternKernel(uint8_t* output, uint32_t width, uint32_t height, uint32_t frameNumber) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate normalized coordinates (0 to 1)
    const float u = static_cast<float>(x) / width;
    const float v = static_cast<float>(y) / height;
    
    // Create a moving gradient pattern
    const float time = frameNumber * 0.05f;
    const float pattern = sinf(u * 10.0f + time) * cosf(v * 10.0f + time) * 0.5f + 0.5f;
    
    // Generate RGB colors
    const uint8_t r = static_cast<uint8_t>((u * 255.0f));
    const uint8_t g = static_cast<uint8_t>((v * 255.0f));
    const uint8_t b = static_cast<uint8_t>((pattern * 255.0f));
    
    // Write to output (RGBA format)
    const size_t idx = (y * width + x) * 4;
    output[idx + 0] = r;
    output[idx + 1] = g;
    output[idx + 2] = b;
    output[idx + 3] = 255; // Alpha
}

} // namespace

namespace cudapattern {

cudaError_t generateTestPattern(cudainterop::CudaVulkanImage& image, uint32_t frameNumber) {
    if (!image.valid || !image.cudaDevPtr) {
        return cudaErrorInvalidValue;
    }
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (image.extent.width + blockDim.x - 1) / blockDim.x,
        (image.extent.height + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    testPatternKernel<<<gridDim, blockDim>>>(
        static_cast<uint8_t*>(image.cudaDevPtr),
        image.extent.width,
        image.extent.height,
        frameNumber
    );
    
    return cudaDeviceSynchronize(); // Make sure the kernel is done before Vulkan uses the image
}

} // namespace cudapattern 