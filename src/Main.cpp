#include "Context.h"
#include "Controllers.h"
#include "Headset.h"
#include "MeshData.h"
#include "MirrorView.h"
#include "Model.h"
#include "Renderer.h"

#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

namespace {
constexpr float flySpeedMultiplier = 2.5f;
}

int main()
{
    glm::mat4 cameraMatrix = glm::mat4(1.0f); // Transform from world to stage space

    Context* context = new Context();
    if (!context->isValid()) {
        return EXIT_FAILURE;
    }

    MirrorView mirrorView(context);
    if (!mirrorView.isValid()) {
        return EXIT_FAILURE;
    }

    if (!context->createDevice(mirrorView.getSurface())) {
        return EXIT_FAILURE;
    }

    Headset headset(context);
    if (!headset.isValid()) {
        return EXIT_FAILURE;
    }

    Renderer* renderer;
    try {
        renderer = new Renderer(context, &headset);
    } catch (...) {
        return EXIT_FAILURE;
    }

    if (!mirrorView.connect(&headset, renderer)) {
        return EXIT_FAILURE;
    }

    bool enableMirrorView = false;

    // Main loop
    std::chrono::high_resolution_clock::time_point previousTime = std::chrono::high_resolution_clock::now();
    while (!headset.isExitRequested() && !mirrorView.isExitRequested()) {
        // Calculate the delta time in seconds
        const std::chrono::high_resolution_clock::time_point nowTime = std::chrono::high_resolution_clock::now();
        const long long elapsedNanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(nowTime - previousTime).count();
        const float deltaTime = static_cast<float>(elapsedNanoseconds) / 1e9f;
        previousTime = nowTime;

        mirrorView.processWindowEvents();

        uint32_t swapchainImageIndex;
        const Headset::BeginFrameResult frameResult = headset.beginFrame(swapchainImageIndex);
        if (frameResult == Headset::BeginFrameResult::Error) {
            return EXIT_FAILURE;
        } else if (frameResult == Headset::BeginFrameResult::SkipFully) {
            continue;
        }

        renderer->generateTestPatterns();
        renderer->record(swapchainImageIndex);

        // Add mirror view rendering
        MirrorView::RenderResult mirrorResult;
        if (enableMirrorView) {
            mirrorResult = mirrorView.render(swapchainImageIndex);
            if (mirrorResult == MirrorView::RenderResult::Error) {
                return EXIT_FAILURE;
            }
        }

        const bool mirrorViewVisible = (mirrorResult == MirrorView::RenderResult::Visible);

        renderer->submit(mirrorViewVisible);
        if (mirrorViewVisible) {
            mirrorView.present();
        }

        if (frameResult == Headset::BeginFrameResult::RenderFully || frameResult == Headset::BeginFrameResult::SkipRender) {
            headset.endFrame();
        }
    }

    context->sync(); // Sync before destroying so that resources are free
    delete renderer;
    delete context;
    return EXIT_SUCCESS;
}