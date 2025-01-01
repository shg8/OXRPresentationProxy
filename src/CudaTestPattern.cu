#include "CudaTestPattern.h"
#include <cuda_runtime.h>
#include <surface_functions.h>

namespace {

__global__ void testPatternKernel(cudaSurfaceObject_t surface, uint32_t width, uint32_t height, uint32_t frameNumber) {
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
    const uchar4 pixel = make_uchar4(
        static_cast<unsigned char>(u * 255.0f),         // R
        static_cast<unsigned char>(v * 255.0f),         // G
        static_cast<unsigned char>(pattern * 255.0f),   // B
        255                                             // A
    );
    
    // Write to surface using cudaBoundaryModeZero to handle edge cases
    surf2Dwrite<uchar4>(pixel, surface, x * 4, y, cudaBoundaryModeZero);
}

} // namespace

namespace cudapattern {

cudaError_t generateTestPattern(cudainterop::CudaVulkanImage& image, uint32_t frameNumber) {
    if (!image.valid || !image.cudaSurface) {
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
        image.cudaSurface,
        image.extent.width,
        image.extent.height,
        frameNumber
    );
    
    return cudaDeviceSynchronize(); // Make sure the kernel is done before Vulkan uses the image
}

} // namespace cudapattern 