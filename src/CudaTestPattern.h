#pragma once

#include <cuda_runtime.h>
#include "CudaInterop.h"

namespace cudapattern {

// Generate a test pattern in the given CUDA-Vulkan image
// The pattern will be a color gradient with some moving elements based on the frame number
cudaError_t generateTestPattern(cudainterop::CudaVulkanImage& image, uint32_t frameNumber);

} // namespace cudapattern 