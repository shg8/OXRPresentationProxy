#pragma once

#include <cuda_runtime.h>
#include "CudaInterop.h"

namespace cudapattern {

// Generate a test pattern in the given CUDA-Vulkan image
// The pattern will be a color gradient with some moving elements based on the frame number
// pitch: The number of bytes per row in the image (may be larger than width * 4 due to alignment)
cudaError_t generateTestPattern(cudainterop::CudaVulkanImage& image, size_t pitch, uint32_t frameNumber);

} // namespace cudapattern 