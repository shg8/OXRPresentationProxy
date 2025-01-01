#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "../src/Context.h"
#include "../src/Headset.h"
#include "../src/Renderer.h"
#include "../src/CudaInterop.h"
#include "../src/CudaTestPattern.h"

namespace py = pybind11;

// Global state (we'll need to manage this better in production)
static Context* g_context = nullptr;
static Headset* g_headset = nullptr;
static Renderer* g_renderer = nullptr;
static uint32_t g_currentSwapchainImageIndex = 0;

// Initialize the VR system
bool initialize() {
    if (g_context || g_headset || g_renderer) {
        return false; // Already initialized
    }

    g_context = new Context();
    if (!g_context->isValid()) {
        delete g_context;
        g_context = nullptr;
        return false;
    }

    g_headset = new Headset(g_context);
    if (!g_headset->isValid()) {
        delete g_headset;
        g_headset = nullptr;
        delete g_context;
        g_context = nullptr;
        return false;
    }

    try {
        g_renderer = new Renderer(g_context, g_headset);
    } catch (...) {
        delete g_headset;
        g_headset = nullptr;
        delete g_context;
        g_context = nullptr;
        return false;
    }

    return true;
}

// Clean up resources
void cleanup() {
    delete g_renderer;
    g_renderer = nullptr;
    delete g_headset;
    g_headset = nullptr;
    delete g_context;
    g_context = nullptr;
}

// Start a new frame and get the eye matrices
py::dict startFrame() {
    if (!g_context || !g_headset || !g_renderer) {
        throw std::runtime_error("VR system not initialized");
    }

    // Begin the frame
    Headset::BeginFrameResult result = g_headset->beginFrame(g_currentSwapchainImageIndex);
    if (result == Headset::BeginFrameResult::Error) {
        throw std::runtime_error("Failed to begin frame");
    }

    // Skip if we shouldn't render
    if (result == Headset::BeginFrameResult::SkipFully) {
        return py::dict();
    }

    // Get matrices for both eyes
    py::dict frameInfo;
    for (size_t eyeIndex = 0; eyeIndex < EYE_COUNT; ++eyeIndex) {
        // Get view and projection matrices
        glm::mat4 viewMatrix = g_headset->getEyeViewMatrix(eyeIndex);
        glm::mat4 projMatrix = g_headset->getEyeProjectionMatrix(eyeIndex);

        // Convert to numpy arrays
        py::array_t<float> viewArray({4, 4});
        py::array_t<float> projArray({4, 4});
        
        auto viewBuffer = viewArray.mutable_unchecked<2>();
        auto projBuffer = projArray.mutable_unchecked<2>();

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                viewBuffer(i, j) = viewMatrix[i][j];
                projBuffer(i, j) = projMatrix[i][j];
            }
        }

        // Add to return dictionary
        std::string eyePrefix = (eyeIndex == 0) ? "left_" : "right_";
        frameInfo[eyePrefix + "view_matrix"] = viewArray;
        frameInfo[eyePrefix + "projection_matrix"] = projArray;
    }

    return frameInfo;
}

// Submit frame data from PyTorch tensors
void submitFrame(py::object leftEyeTensor, py::object rightEyeTensor) {
    if (!g_context || !g_headset || !g_renderer) {
        throw std::runtime_error("VR system not initialized");
    }

    // Verify tensor properties
    auto verifyTensor = [](const py::object& tensor, cudainterop::CudaVulkanImage& targetImage) {
        if (!torch::is_tensor(tensor)) {
            throw std::runtime_error("Input must be a torch tensor");
        }
        
        auto t = torch::from_blob(tensor.ptr());
        if (t.device().type() != torch::kCUDA) {
            throw std::runtime_error("Tensor must be on CUDA device");
        }
        
        if (t.dim() != 3 || t.size(2) != 4) {
            throw std::runtime_error("Tensor must have shape (H, W, 4)");
        }

        // Check dimensions match the target image
        if (t.size(0) != targetImage.extent.height || t.size(1) != targetImage.extent.width) {
            throw std::runtime_error("Tensor dimensions must match target image size");
        }
    };

    // Get current frame's images
    auto& stereoImageSet = g_renderer->offscreenImages.at(g_renderer->currentRenderProcessIndex);

    // Verify tensors match our requirements
    verifyTensor(leftEyeTensor, stereoImageSet[0]);
    verifyTensor(rightEyeTensor, stereoImageSet[1]);

    // Copy data from tensors to CUDA surfaces
    auto copyTensorToImage = [](const py::object& tensor, cudainterop::CudaVulkanImage& image) {
        auto t = torch::from_blob(tensor.ptr());
        void* tensorData = t.data_ptr();
        
        // Get tensor properties
        size_t pitch = t.stride(0) * sizeof(float); // Assuming float32 tensors
        
        // Copy the data
        cudaError_t result = cudainterop::copyFromDevicePointerToCudaImage(
            image,
            tensorData,
            pitch,
            image.extent
        );
        
        if (result != cudaSuccess) {
            throw std::runtime_error("Failed to copy tensor data to CUDA surface");
        }
    };

    copyTensorToImage(leftEyeTensor, stereoImageSet[0]);
    copyTensorToImage(rightEyeTensor, stereoImageSet[1]);

    // Record and submit
    g_renderer->record(g_currentSwapchainImageIndex);
    g_renderer->submit(false);

    // End the frame
    g_headset->endFrame();
}

PYBIND11_MODULE(OXRPresentationPython, m) {
    m.doc() = "Python bindings for OXRPresentationProxy"; 
    
    m.def("initialize", &initialize, "Initialize the VR system");
    m.def("cleanup", &cleanup, "Clean up VR system resources");
    m.def("startFrame", &startFrame, "Start a new frame and get eye matrices");
    m.def("submitFrame", &submitFrame, "Submit frame data from PyTorch tensors",
          py::arg("leftEyeTensor"), py::arg("rightEyeTensor"));
} 