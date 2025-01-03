#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// PyTorch includes
#include <torch/extension.h>
#include <torch/types.h>
#include <torch/python.h>  // This provides the type caster for torch::Tensor
#include <c10/cuda/CUDAStream.h>

// GLM includes
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// Project includes
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
py::dict initialize() {
    if (g_context || g_headset || g_renderer) {
        throw std::runtime_error("Already initialized");
    }

    g_context = new Context();
    if (!g_context->isValid()) {
        delete g_context;
        g_context = nullptr;
        throw std::runtime_error("Failed to initialize Context");
    }

    // Create a temporary surface to initialize the device
    GLFWwindow* tempWindow = glfwCreateWindow(1, 1, "Context Window", nullptr, nullptr);
    if (!tempWindow) {
        delete g_context;
        g_context = nullptr;
        throw std::runtime_error("Failed to create temporary window");
    }

    VkSurfaceKHR tempSurface;
    if (glfwCreateWindowSurface(g_context->getVkInstance(), tempWindow, nullptr, &tempSurface) != VK_SUCCESS) {
        glfwDestroyWindow(tempWindow);
        delete g_context;
        g_context = nullptr;
        throw std::runtime_error("Failed to create temporary surface");
    }

    // Create the device
    if (!g_context->createDevice(tempSurface)) {
        vkDestroySurfaceKHR(g_context->getVkInstance(), tempSurface, nullptr);
        glfwDestroyWindow(tempWindow);
        delete g_context;
        g_context = nullptr;
        throw std::runtime_error("Failed to create device");
    }

    g_headset = new Headset(g_context);
    if (!g_headset->isValid()) {
        delete g_headset;
        g_headset = nullptr;
        delete g_context;
        g_context = nullptr;
        throw std::runtime_error("Failed to initialize Headset");
    }

    try {
        g_renderer = new Renderer(g_context, g_headset);
    } catch (...) {
        delete g_headset;
        g_headset = nullptr;
        delete g_context;
        g_context = nullptr;
        throw std::runtime_error("Failed to initialize Renderer");
    }

    // Get swapchain extent and return it
    VkExtent2D extent = g_headset->getEyeResolution(0);
    py::dict result;
    result["width"] = extent.width;
    result["height"] = extent.height;
    return result;
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
    for (size_t eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
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
        frameInfo[py::str(eyePrefix + "view_matrix")] = viewArray;
        frameInfo[py::str(eyePrefix + "projection_matrix")] = projArray;
    }

    return frameInfo;
}

// Submit frame data from PyTorch tensors
void submitFrame(torch::Tensor leftEyeTensor, torch::Tensor rightEyeTensor) {
    if (!g_context || !g_headset || !g_renderer) {
        throw std::runtime_error("VR system not initialized");
    }

    // Verify tensor properties
    auto verifyTensor = [](const torch::Tensor& tensor, cudainterop::CudaVulkanImage& targetImage) {
        if (tensor.device().type() != torch::kCUDA) {
            throw std::runtime_error("Tensor must be on CUDA device");
        }
        
        if (tensor.dim() != 3 || tensor.size(2) != 4) {
            throw std::runtime_error("Tensor must have shape (H, W, 4)");
        }

        // Check dimensions match the target image
        if (tensor.size(0) != targetImage.extent.height || tensor.size(1) != targetImage.extent.width) {
            throw std::runtime_error("Tensor dimensions must match target image size");
        }
    };

    // Get current frame's images
    auto& stereoImageSet = g_renderer->offscreenImages.at(g_renderer->currentRenderProcessIndex);

    // Verify tensors match our requirements
    verifyTensor(leftEyeTensor, stereoImageSet[0]);
    verifyTensor(rightEyeTensor, stereoImageSet[1]);

    // Copy data from tensors to CUDA surfaces
    auto copyTensorToImage = [](const torch::Tensor& tensor, cudainterop::CudaVulkanImage& image) {
        void* tensorData = tensor.data_ptr();
        
        // Get tensor properties
        size_t pitch = tensor.stride(0) * sizeof(float); // Assuming float32 tensors
        
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
    
    m.def("initialize", &initialize, "Initialize the VR system and return swapchain dimensions");
    m.def("cleanup", &cleanup, "Clean up VR system resources");
    m.def("startFrame", &startFrame, "Start a new frame and get eye matrices");
    m.def("submitFrame", &submitFrame, "Submit frame data from PyTorch tensors",
          py::arg("leftEyeTensor"), py::arg("rightEyeTensor"));
} 