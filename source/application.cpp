#include "application.hpp"

#include <imgui.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

// ---------- Минимальная математика ----------

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Mat4 {
    float m[16] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1
    };

    static Mat4 translation(const Vec3& t) {
        Mat4 r;
        r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
        return r;
    }
    static Mat4 scale(const Vec3& s) {
        Mat4 r;
        r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z;
        return r;
    }
    static Mat4 rotationX(float rad) {
        Mat4 r;
        float c = std::cos(rad), s = std::sin(rad);
        r.m[5] = c;  r.m[6] = s;
        r.m[9] = -s; r.m[10] = c;
        return r;
    }
    static Mat4 rotationY(float rad) {
        Mat4 r;
        float c = std::cos(rad), s = std::sin(rad);
        r.m[0] = c;  r.m[2] = -s;
        r.m[8] = s;  r.m[10] = c;
        return r;
    }
    static Mat4 rotationZ(float rad) {
        Mat4 r;
        float c = std::cos(rad), s = std::sin(rad);
        r.m[0] = c;  r.m[1] = s;
        r.m[4] = -s; r.m[5] = c;
        return r;
    }
    static Mat4 perspective(float fovY, float aspect, float zn, float zf) {
        Mat4 r;
        float t = 1.0f / std::tan(fovY * 0.5f);
        r.m[0] = t / aspect;
        r.m[5] = t;
        r.m[10] = zf / (zn - zf);
        r.m[11] = -1.0f;
        r.m[14] = (zf * zn) / (zn - zf);
        r.m[15] = 0.0f;
        return r;
    }
    static Mat4 ortho(float l, float r_, float b, float t, float zn, float zf) {
        Mat4 r;
        r.m[0] = 2.0f / (r_ - l);
        r.m[5] = 2.0f / (t - b);
        r.m[10] = 1.0f / (zn - zf);
        r.m[12] = -(r_ + l) / (r_ - l);
        r.m[13] = -(t + b) / (t - b);
        r.m[14] = zn / (zn - zf);
        return r;
    }
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = { center.x - eye.x, center.y - eye.y, center.z - eye.z };
        float fl = std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
        f.x /= fl; f.y /= fl; f.z /= fl;

        Vec3 s = { f.y * up.z - f.z * up.y, f.z * up.x - f.x * up.z, f.x * up.y - f.y * up.x };
        float sl = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
        s.x /= sl; s.y /= sl; s.z /= sl;

        Vec3 u = { s.y * f.z - s.z * f.y, s.z * f.x - s.x * f.z, s.x * f.y - s.y * f.x };

        Mat4 r;
        r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z;
        r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
        r.m[12] = -(s.x * eye.x + s.y * eye.y + s.z * eye.z);
        r.m[13] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
        r.m[14] = (f.x * eye.x + f.y * eye.y + f.z * eye.z);
        return r;
    }

    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int row = 0; row < 4; ++row) {
                float sum = 0;
                for (int k = 0; k < 4; ++k)
                    sum += m[k * 4 + row] * o.m[c * 4 + k];
                r.m[c * 4 + row] = sum;
            }
        return r;
    }
};

struct UniformBufferObject {
    Mat4 model;
    Mat4 view;
    Mat4 proj;
    float colorMultiplier[4];
};

// ---------- Состояние приложения ----------

namespace {

    struct Vertex {
        float pos[3];
        float color[3];
    };

    VkPipelineLayout       pipelineLayout = VK_NULL_HANDLE;
    VkPipeline             graphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout  descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool       descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet        descriptorSet = VK_NULL_HANDLE;

    VkBuffer       vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation  vertexBufferAlloc = VK_NULL_HANDLE;

    VkBuffer       indexBuffer = VK_NULL_HANDLE;
    VmaAllocation  indexBufferAlloc = VK_NULL_HANDLE;

    VkBuffer       uniformBuffer = VK_NULL_HANDLE;
    VmaAllocation  uniformBufferAlloc = VK_NULL_HANDLE;

    // НЕ используем vmaUnmapMemory — эта память замаплена на всё время жизни.
    // Просто храним указатели, полученные при создании.
    Vertex* vertexBufferMapped = nullptr;
    uint32_t* indexBufferMapped = nullptr;
    UniformBufferObject* uniformBufferMapped = nullptr;

    uint32_t indexCount = 0;

    // UI
    bool  usePerspective = true;
    float position[3] = { 0.f, 0.f, 0.f };
    float rotation[3] = { 0.f, 0.f, 0.f };
    float scaleVal[3] = { 1.f, 1.f, 1.f };
    float color[3] = { 1.f, 1.f, 1.f };

    bool  animate = false;
    float animTime = 0.f;
    float animSpeed = 1.0f;
    float animRadius = 2.0f;

    // ---------- Утилиты ----------

    std::vector<char> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << path << "\n";
            return {};
        }
        size_t size = (size_t)file.tellg();
        std::vector<char> buf(size);
        file.seekg(0);
        file.read(buf.data(), size);
        return buf;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = code.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule mod = VK_NULL_HANDLE;
        vkCreateShaderModule(graphics::internal::context.device, &ci, nullptr, &mod);
        return mod;
    }

    // ---------- Сфера ----------

    void generateSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
        int stacks, int slices) {
        vertices.clear();
        indices.clear();

        for (int i = 0; i <= stacks; ++i) {
            float phi = float(M_PI) * float(i) / float(stacks);
            for (int j = 0; j <= slices; ++j) {
                float theta = 2.0f * float(M_PI) * float(j) / float(slices);
                float x = std::sin(phi) * std::cos(theta);
                float y = std::cos(phi);
                float z = std::sin(phi) * std::sin(theta);

                Vertex v;
                v.pos[0] = x; v.pos[1] = y; v.pos[2] = z;
                v.color[0] = (x + 1.0f) * 0.5f;
                v.color[1] = (y + 1.0f) * 0.5f;
                v.color[2] = (z + 1.0f) * 0.5f;
                vertices.push_back(v);
            }
        }

        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < slices; ++j) {
                uint32_t a = i * (slices + 1) + j;
                uint32_t b = a + slices + 1;

                indices.push_back(a);
                indices.push_back(b);
                indices.push_back(a + 1);

                indices.push_back(a + 1);
                indices.push_back(b);
                indices.push_back(b + 1);
            }
        }
    }

    // ---------- Буферы ----------
    // Создаём сразу с маппингом. VMA сама держит маппинг на всё время жизни буфера,
    // а при vmaDestroyBuffer — освобождает. Никаких vmaMapMemory/vmaUnmapMemory не нужно.

    void createMappedBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkBuffer& outBuffer, VmaAllocation& outAlloc,
        void** outMapped) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
            | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        VkResult r = vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai,
            &outBuffer, &outAlloc, &info);
        if (r != VK_SUCCESS) {
            std::cerr << "Failed to create buffer, VkResult = " << r << "\n";
            outBuffer = VK_NULL_HANDLE;
            outAlloc = VK_NULL_HANDLE;
            if (outMapped) *outMapped = nullptr;
            return;
        }
        if (outMapped) *outMapped = info.pMappedData;
    }

    // ---------- Pipeline ----------

    bool createPipeline() {
        auto& ctx = graphics::internal::context;

        VkDescriptorSetLayoutBinding uboBinding{};
        uboBinding.binding = 0;
        uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboBinding.descriptorCount = 1;
        uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo dslCI{};
        dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslCI.bindingCount = 1;
        dslCI.pBindings = &uboBinding;

        if (vkCreateDescriptorSetLayout(ctx.device, &dslCI, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor set layout\n";
            return false;
        }

        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount = 1;
        plCI.pSetLayouts = &descriptorSetLayout;

        if (vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &pipelineLayout) != VK_SUCCESS) {
            std::cerr << "Failed to create pipeline layout\n";
            return false;
        }

        auto vertCode = readFile("shaders/sphere.vert.spv");
        auto fragCode = readFile("shaders/sphere.frag.spv");
        if (vertCode.empty() || fragCode.empty()) {
            std::cerr << "Shader files are empty or missing\n";
            return false;
        }

        VkShaderModule vertMod = createShaderModule(vertCode);
        VkShaderModule fragMod = createShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertMod;
        stages[0].pName = "main";

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragMod;
        stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0; attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1; attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, color);

        VkPipelineVertexInputStateCreateInfo viCI{};
        viCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        viCI.vertexBindingDescriptionCount = 1;
        viCI.pVertexBindingDescriptions = &binding;
        viCI.vertexAttributeDescriptionCount = 2;
        viCI.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo iaCI{};
        iaCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vpCI{};
        vpCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpCI.viewportCount = 1;
        vpCI.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rastCI{};
        rastCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rastCI.polygonMode = VK_POLYGON_MODE_FILL;
        rastCI.cullMode = VK_CULL_MODE_NONE;
        rastCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rastCI.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo msCI{};
        msCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo dsCI{};
        dsCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dsCI.depthTestEnable = VK_TRUE;
        dsCI.depthWriteEnable = VK_TRUE;
        dsCI.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState blendAtt{};
        blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        blendAtt.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cbCI{};
        cbCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cbCI.attachmentCount = 1;
        cbCI.pAttachments = &blendAtt;

        VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynCI{};
        dynCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynCI.dynamicStateCount = 2;
        dynCI.pDynamicStates = dynStates;

        VkGraphicsPipelineCreateInfo gpCI{};
        gpCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpCI.stageCount = 2;
        gpCI.pStages = stages;
        gpCI.pVertexInputState = &viCI;
        gpCI.pInputAssemblyState = &iaCI;
        gpCI.pViewportState = &vpCI;
        gpCI.pRasterizationState = &rastCI;
        gpCI.pMultisampleState = &msCI;
        gpCI.pDepthStencilState = &dsCI;
        gpCI.pColorBlendState = &cbCI;
        gpCI.pDynamicState = &dynCI;
        gpCI.layout = pipelineLayout;
        gpCI.renderPass = ctx.render_pass;
        gpCI.subpass = 0;

        VkResult res = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpCI, nullptr, &graphicsPipeline);

        vkDestroyShaderModule(ctx.device, vertMod, nullptr);
        vkDestroyShaderModule(ctx.device, fragMod, nullptr);

        if (res != VK_SUCCESS) {
            std::cerr << "Failed to create graphics pipeline, VkResult = " << res << "\n";
            return false;
        }
        return true;
    }

    bool createDescriptors() {
        auto& ctx = graphics::internal::context;

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.poolSizeCount = 1;
        poolCI.pPoolSizes = &poolSize;
        poolCI.maxSets = 1;

        if (vkCreateDescriptorPool(ctx.device, &poolCI, nullptr, &descriptorPool) != VK_SUCCESS) {
            std::cerr << "Failed to create descriptor pool\n";
            return false;
        }

        VkDescriptorSetAllocateInfo allocCI{};
        allocCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocCI.descriptorPool = descriptorPool;
        allocCI.descriptorSetCount = 1;
        allocCI.pSetLayouts = &descriptorSetLayout;

        if (vkAllocateDescriptorSets(ctx.device, &allocCI, &descriptorSet) != VK_SUCCESS) {
            std::cerr << "Failed to allocate descriptor set\n";
            return false;
        }

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = uniformBuffer;
        bufInfo.offset = 0;
        bufInfo.range = sizeof(UniformBufferObject);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet;
        write.dstBinding = 0;
        write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo = &bufInfo;

        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
        return true;
    }

} // namespace

// ---------- Публичные функции ----------

namespace application {

    bool initialize() {
        auto& ctx = graphics::internal::context;

        // Сброс на случай повторного запуска в одной сессии отладки
        vertexBuffer = VK_NULL_HANDLE;  vertexBufferAlloc = VK_NULL_HANDLE;
        indexBuffer = VK_NULL_HANDLE;  indexBufferAlloc = VK_NULL_HANDLE;
        uniformBuffer = VK_NULL_HANDLE;  uniformBufferAlloc = VK_NULL_HANDLE;
        vertexBufferMapped = nullptr;
        indexBufferMapped = nullptr;
        uniformBufferMapped = nullptr;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        generateSphere(vertices, indices, 9, 9);
        indexCount = (uint32_t)indices.size();

        VkDeviceSize vbSize = sizeof(Vertex) * vertices.size();
        VkDeviceSize ibSize = sizeof(uint32_t) * indices.size();

        // --- Vertex buffer ---
        {
            void* mapped = nullptr;
            createMappedBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                vertexBuffer, vertexBufferAlloc, &mapped);
            if (vertexBuffer == VK_NULL_HANDLE || mapped == nullptr) {
                std::cerr << "Vertex buffer creation failed\n";
                return false;
            }
            vertexBufferMapped = static_cast<Vertex*>(mapped);
            std::memcpy(vertexBufferMapped, vertices.data(), (size_t)vbSize);
        }

        // --- Index buffer ---
        {
            void* mapped = nullptr;
            createMappedBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                indexBuffer, indexBufferAlloc, &mapped);
            if (indexBuffer == VK_NULL_HANDLE || mapped == nullptr) {
                std::cerr << "Index buffer creation failed\n";
                return false;
            }
            indexBufferMapped = static_cast<uint32_t*>(mapped);
            std::memcpy(indexBufferMapped, indices.data(), (size_t)ibSize);
        }

        // --- Uniform buffer ---
        {
            void* mapped = nullptr;
            createMappedBuffer(sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                uniformBuffer, uniformBufferAlloc, &mapped);
            if (uniformBuffer == VK_NULL_HANDLE || mapped == nullptr) {
                std::cerr << "Uniform buffer creation failed\n";
                return false;
            }
            uniformBufferMapped = static_cast<UniformBufferObject*>(mapped);
        }

        if (!createPipeline()) return false;
        if (!createDescriptors()) return false;

        std::cerr << "[application::initialize] success\n";
        return true;
    }

    void shutdown() {
        std::cerr << "[application::shutdown] entered\n";
        auto& ctx = graphics::internal::context;

        if (ctx.device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx.device);
        }

        if (graphicsPipeline) { vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr); graphicsPipeline = VK_NULL_HANDLE; }
        if (pipelineLayout) { vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr); pipelineLayout = VK_NULL_HANDLE; }
        if (descriptorPool) { vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr); descriptorPool = VK_NULL_HANDLE; }
        if (descriptorSetLayout) { vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, nullptr); descriptorSetLayout = VK_NULL_HANDLE; }

        // НИКАКОГО vmaUnmapMemory. VMA сама снимает маппинг при vmaDestroyBuffer.

        if (vertexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAlloc);
        }
        vertexBuffer = VK_NULL_HANDLE;
        vertexBufferAlloc = VK_NULL_HANDLE;
        vertexBufferMapped = nullptr;

        if (indexBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAlloc);
        }
        indexBuffer = VK_NULL_HANDLE;
        indexBufferAlloc = VK_NULL_HANDLE;
        indexBufferMapped = nullptr;

        if (uniformBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(ctx.allocator, uniformBuffer, uniformBufferAlloc);
        }
        uniformBuffer = VK_NULL_HANDLE;
        uniformBufferAlloc = VK_NULL_HANDLE;
        uniformBufferMapped = nullptr;

        std::cerr << "[application::shutdown] done\n";
    }

    void update(double time) {
        (void)time;

        ImGui::Begin("Sphere Controls");
        ImGui::Checkbox("Perspective projection", &usePerspective);

        ImGui::Separator();
        ImGui::Text("Transform");
        ImGui::SliderFloat3("Position", position, -5.0f, 5.0f);
        ImGui::SliderFloat3("Rotation (deg)", rotation, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", scaleVal, 0.1f, 3.0f);

        ImGui::Separator();
        ImGui::Text("Animation");
        ImGui::Checkbox("Animate", &animate);
        ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 5.0f);
        ImGui::SliderFloat("Radius", &animRadius, 0.5f, 5.0f);

        ImGui::Separator();
        ImGui::ColorEdit3("Color", color);

        ImGui::End();

        double dt = ImGui::GetIO().DeltaTime;
        if (animate) animTime += (float)dt * animSpeed;

        float px = position[0];
        float py = position[1];
        float pz = position[2];
        if (animate) {
            px = std::cos(animTime) * animRadius;
            pz = std::sin(animTime) * animRadius;
        }

        Mat4 model = Mat4::translation({ px, py, pz })
            * Mat4::rotationY(rotation[1] * 3.14159265f / 180.0f)
            * Mat4::rotationX(rotation[0] * 3.14159265f / 180.0f)
            * Mat4::rotationZ(rotation[2] * 3.14159265f / 180.0f)
            * Mat4::scale({ scaleVal[0], scaleVal[1], scaleVal[2] });

        auto& ctx = graphics::internal::context;
        float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
        Mat4 view = Mat4::lookAt({ 0.0f, 0.0f, 5.0f }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });

        Mat4 proj;
        if (usePerspective) {
            proj = Mat4::perspective(45.0f * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);
        }
        else {
            float orthoSize = 3.0f;
            proj = Mat4::ortho(-orthoSize * aspect, orthoSize * aspect,
                -orthoSize, orthoSize, 0.1f, 100.0f);
        }

        if (uniformBufferMapped) {
            UniformBufferObject ubo{};
            ubo.model = model;
            ubo.view = view;
            ubo.proj = proj;
            ubo.colorMultiplier[0] = color[0];
            ubo.colorMultiplier[1] = color[1];
            ubo.colorMultiplier[2] = color[2];
            ubo.colorMultiplier[3] = 1.0f;
            *uniformBufferMapped = ubo;
        }
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;
        VkCommandBuffer cmd = fd.command_buffer;

        if (graphicsPipeline == VK_NULL_HANDLE) return;

        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginCI{};
        beginCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginCI);

        VkClearValue clearValues[2]{};
        clearValues[0].color = { {0.05f, 0.05f, 0.08f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rpCI{};
        rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpCI.renderPass = ctx.render_pass;
        rpCI.framebuffer = fd.framebuffer;
        rpCI.renderArea.offset = { 0, 0 };
        rpCI.renderArea.extent = ctx.swapchain_extent;
        rpCI.clearValueCount = 2;
        rpCI.pClearValues = clearValues;

        vkCmdBeginRenderPass(cmd, &rpCI, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)ctx.swapchain_extent.width;
        viewport.height = (float)ctx.swapchain_extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = ctx.swapchain_extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application