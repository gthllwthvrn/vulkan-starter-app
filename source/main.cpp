#include <cstdint>
#include <climits>
#include <iostream>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>

#include <windows.h>

#include "graphics_internal.hpp"
#include "application.hpp"

namespace {

    constexpr int32_t default_window_width = 1280;
    constexpr int32_t default_window_height = 720;
    constexpr char    default_window_title[] = "Vulkan Starter App";

    GLFWwindow* glfw_window = nullptr;

} // namespace

int main() {
    if (!glfwInit()) {
        MessageBoxA(nullptr, "glfwInit failed", "Error", MB_OK);
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfw_window = glfwCreateWindow(default_window_width, default_window_height,
        default_window_title, nullptr, nullptr);
    if (!glfw_window) {
        MessageBoxA(nullptr, "glfwCreateWindow failed", "Error", MB_OK);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwSetFramebufferSizeCallback(glfw_window, [](GLFWwindow*, int w, int h) {
        if (w == 0 || h == 0) return;
        graphics::internal::resize((uint32_t)w, (uint32_t)h);
        });

    if (!ImGui::CreateContext()) {
        MessageBoxA(nullptr, "ImGui::CreateContext failed", "Error", MB_OK);
        glfwDestroyWindow(glfw_window); glfwTerminate();
        return EXIT_FAILURE;
    }

    if (!ImGui_ImplGlfw_InitForVulkan(glfw_window, true)) {
        MessageBoxA(nullptr, "ImGui_ImplGlfw_InitForVulkan failed", "Error", MB_OK);
        ImGui::DestroyContext();
        glfwDestroyWindow(glfw_window); glfwTerminate();
        return EXIT_FAILURE;
    }

    if (!graphics::internal::initialize(glfw_window)) {
        MessageBoxA(nullptr, "graphics::internal::initialize failed", "Error", MB_OK);
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(glfw_window); glfwTerminate();
        return EXIT_FAILURE;
    }

    if (!application::initialize()) {
        MessageBoxA(nullptr, "application::initialize failed", "Error", MB_OK);
        // всё равно вызовем shutdown, чтобы корректно освободить всё, что успело создаться
        application::shutdown();
        graphics::internal::shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(glfw_window); glfwTerminate();
        return EXIT_FAILURE;
    }

    // MessageBoxA(nullptr, "Init OK! Now entering main loop...", "Debug", MB_OK);

    while (!glfwWindowShouldClose(glfw_window)) {
        const double time = glfwGetTime();

        glfwPollEvents();
        ImGui_ImplGlfw_NewFrame();

        ImGui::NewFrame();
        application::update(time);
        ImGui::Render();

        graphics::internal::FrameData fd = graphics::internal::prepare();
        application::render(fd);
        graphics::internal::submitAndPresent();
    }

    application::shutdown();
    graphics::internal::shutdown();

    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(glfw_window);
    glfwTerminate();

    return EXIT_SUCCESS;
}