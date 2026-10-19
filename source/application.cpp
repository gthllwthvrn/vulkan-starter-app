#include "application.hpp"

#include <imgui.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct Mat4 {
    float m[16] = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 };
    static Mat4 translation(const Vec3& t) { Mat4 r; r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z; return r; }
    static Mat4 scale(const Vec3& s) { Mat4 r; r.m[0] = s.x; r.m[5] = s.y; r.m[10] = s.z; return r; }
    static Mat4 rotationX(float rad) { Mat4 r; float c = std::cos(rad), s = std::sin(rad); r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c; return r; }
    static Mat4 rotationY(float rad) { Mat4 r; float c = std::cos(rad), s = std::sin(rad); r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c; return r; }
    static Mat4 rotationZ(float rad) { Mat4 r; float c = std::cos(rad), s = std::sin(rad); r.m[0] = c; r.m[1] = s; r.m[4] = -s; r.m[5] = c; return r; }
    static Mat4 perspective(float fovY, float aspect, float zn, float zf) {
        Mat4 r; float t = 1.0f / std::tan(fovY * 0.5f);
        r.m[0] = t / aspect; r.m[5] = t; r.m[10] = zf / (zn - zf); r.m[11] = -1.0f; r.m[14] = (zf * zn) / (zn - zf); r.m[15] = 0.0f; return r;
    }
    static Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
        Vec3 f = { center.x - eye.x, center.y - eye.y, center.z - eye.z };
        float fl = std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z); f.x /= fl; f.y /= fl; f.z /= fl;
        Vec3 s = { f.y * up.z - f.z * up.y, f.z * up.x - f.x * up.z, f.x * up.y - f.y * up.x };
        float sl = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z); s.x /= sl; s.y /= sl; s.z /= sl;
        Vec3 u = { s.y * f.z - s.z * f.y, s.z * f.x - s.x * f.z, s.x * f.y - s.y * f.x };
        Mat4 r;
        r.m[0] = s.x; r.m[4] = s.y; r.m[8] = s.z; r.m[1] = u.x; r.m[5] = u.y; r.m[9] = u.z;
        r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z;
        r.m[12] = -(s.x * eye.x + s.y * eye.y + s.z * eye.z); r.m[13] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z); r.m[14] = (f.x * eye.x + f.y * eye.y + f.z * eye.z);
        return r;
    }
    Mat4 operator*(const Mat4& o) const {
        Mat4 r;
        for (int c = 0; c < 4; ++c) for (int row = 0; row < 4; ++row) {
            float sum = 0;
            for (int k = 0; k < 4; ++k) sum += m[k * 4 + row] * o.m[c * 4 + k];
            r.m[c * 4 + row] = sum;
        }
        return r;
    }
};

struct SceneUBO {
    Mat4 viewProj;              // 64
    float cameraPos[4];         // 16
    float dirDir[4];            // 16
    float dirColor[4];          // 16
    float ambient[4];           // 16
    int32_t pointCount;         // 4
    int32_t spotCount;          // 4
    int32_t padding[2];         // 8
};                              // = 144

struct ModelUBO {
    Mat4 model;                 // 64
    float albedo[4];            // 16
    float specular[4];          // 16
};                              // = 96

struct PointLight {
    float pos[4];               // 16
    float color[4];             // 16
};                              // = 32

struct SpotLight {
    float pos[4];               // 16
    float color[4];             // 16
    float dir[4];               // 16
    float params[4];            // 16
};                              // = 64

namespace {

    struct Vertex {
        float pos[3];
        float normal[3];
    };

    VkPipelineLayout       pipelineLayout = VK_NULL_HANDLE;
    VkPipeline             graphicsPipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout  sceneSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout  modelSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool       descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet        sceneDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet        modelDescriptorSet = VK_NULL_HANDLE;

    VkBuffer       vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation  vertexBufferAlloc = VK_NULL_HANDLE;
    VkBuffer       indexBuffer = VK_NULL_HANDLE;
    VmaAllocation  indexBufferAlloc = VK_NULL_HANDLE;

    VkBuffer       sceneUBO = VK_NULL_HANDLE;
    VmaAllocation  sceneUBOAlloc = VK_NULL_HANDLE;
    SceneUBO* sceneUBOMapped = nullptr;

    VkBuffer       modelUBO = VK_NULL_HANDLE;
    VmaAllocation  modelUBOAlloc = VK_NULL_HANDLE;
    ModelUBO* modelUBOMapped = nullptr;

    VkBuffer       pointLightBuffer = VK_NULL_HANDLE;
    VmaAllocation  pointLightAlloc = VK_NULL_HANDLE;
    PointLight* pointLightMapped = nullptr;

    VkBuffer       spotLightBuffer = VK_NULL_HANDLE;
    VmaAllocation  spotLightAlloc = VK_NULL_HANDLE;
    SpotLight* spotLightMapped = nullptr;

    constexpr uint32_t MAX_LIGHTS = 16;

    uint32_t indexCount = 0;
    int32_t pointLightCount = 1;
    int32_t spotLightCount = 1;

    float cameraPos[3] = { 0.0f, 0.0f, 5.0f };
    float cameraFov = 45.0f;
    float ambientColor[3] = { 0.1f, 0.1f, 0.1f };
    float dirDir[3] = { -0.5f, -1.0f, -0.5f };
    float dirColor[3] = { 1.0f, 1.0f, 1.0f };
    float albedo[3] = { 0.8f, 0.4f, 0.2f };
    float specularColor[3] = { 1.0f, 1.0f, 1.0f };
    float modelPosition[3] = { 0.0f, 0.0f, 0.0f };
    float modelRotation[3] = { 0.0f, 0.0f, 0.0f };
    float modelScale[3] = { 1.0f, 1.0f, 1.0f };
    bool  animate = false;
    float animTime = 0.f;
    float animSpeed = 1.0f;
    float animRadius = 2.0f;

    std::vector<char> readFile(const std::string& path) {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open()) { std::cerr << "Failed to open " << path << "\n"; return {}; }
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

    void generateSphere(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, int stacks, int slices) {
        vertices.clear(); indices.clear();
        for (int i = 0; i <= stacks; ++i) {
            float phi = float(M_PI) * float(i) / float(stacks);
            for (int j = 0; j <= slices; ++j) {
                float theta = 2.0f * float(M_PI) * float(j) / float(slices);
                float x = std::sin(phi) * std::cos(theta);
                float y = std::cos(phi);
                float z = std::sin(phi) * std::sin(theta);
                Vertex v;
                v.pos[0] = x; v.pos[1] = y; v.pos[2] = z;
                v.normal[0] = x; v.normal[1] = y; v.normal[2] = z;
                vertices.push_back(v);
            }
        }
        for (int i = 0; i < stacks; ++i) {
            for (int j = 0; j < slices; ++j) {
                uint32_t a = i * (slices + 1) + j;
                uint32_t b = a + slices + 1;
                indices.push_back(a);     indices.push_back(b);     indices.push_back(a + 1);
                indices.push_back(a + 1); indices.push_back(b);     indices.push_back(b + 1);
            }
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& outBuffer, VmaAllocation& outAlloc, void** outMapped = nullptr) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO;
        ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        if (outMapped) ai.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;

        VmaAllocationInfo info{};
        VkResult r = vmaCreateBuffer(graphics::internal::context.allocator, &bi, &ai, &outBuffer, &outAlloc, &info);
        if (r != VK_SUCCESS) {
            std::cerr << "Buffer create failed: " << r << "\n";
            outBuffer = VK_NULL_HANDLE; outAlloc = VK_NULL_HANDLE;
            if (outMapped) *outMapped = nullptr;
            return;
        }
        if (outMapped) *outMapped = info.pMappedData;
    }

    bool createPipeline() {
        auto& ctx = graphics::internal::context;

        VkDescriptorSetLayoutBinding sceneBindings[3]{};
        sceneBindings[0].binding = 0; sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; sceneBindings[0].descriptorCount = 1; sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        sceneBindings[1].binding = 1; sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sceneBindings[1].descriptorCount = 1; sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        sceneBindings[2].binding = 2; sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sceneBindings[2].descriptorCount = 1; sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo sceneCI{};
        sceneCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        sceneCI.bindingCount = 3;
        sceneCI.pBindings = sceneBindings;
        vkCreateDescriptorSetLayout(ctx.device, &sceneCI, nullptr, &sceneSetLayout);

        VkDescriptorSetLayoutBinding modelBinding{};
        modelBinding.binding = 0; modelBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; modelBinding.descriptorCount = 1; modelBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo modelCI{};
        modelCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        modelCI.bindingCount = 1;
        modelCI.pBindings = &modelBinding;
        vkCreateDescriptorSetLayout(ctx.device, &modelCI, nullptr, &modelSetLayout);

        VkDescriptorSetLayout layouts[2] = { sceneSetLayout, modelSetLayout };
        VkPipelineLayoutCreateInfo plCI{};
        plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plCI.setLayoutCount = 2;
        plCI.pSetLayouts = layouts;
        vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &pipelineLayout);

        auto vertCode = readFile("shaders/sphere.vert.spv");
        auto fragCode = readFile("shaders/sphere.frag.spv");
        if (vertCode.empty() || fragCode.empty()) return false;

        VkShaderModule vertMod = createShaderModule(vertCode);
        VkShaderModule fragMod = createShaderModule(fragCode);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertMod; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragMod; stages[1].pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0; binding.stride = sizeof(Vertex); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(Vertex, normal);

        VkPipelineVertexInputStateCreateInfo viCI{};
        viCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        viCI.vertexBindingDescriptionCount = 1; viCI.pVertexBindingDescriptions = &binding;
        viCI.vertexAttributeDescriptionCount = 2; viCI.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo iaCI{};
        iaCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        iaCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vpCI{};
        vpCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpCI.viewportCount = 1; vpCI.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rastCI{};
        rastCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rastCI.polygonMode = VK_POLYGON_MODE_FILL; rastCI.cullMode = VK_CULL_MODE_NONE; rastCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rastCI.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo msCI{};
        msCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        msCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo dsCI{};
        dsCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dsCI.depthTestEnable = VK_TRUE; dsCI.depthWriteEnable = VK_TRUE; dsCI.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState blendAtt{};
        blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; blendAtt.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cbCI{};
        cbCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cbCI.attachmentCount = 1; cbCI.pAttachments = &blendAtt;

        VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynCI{};
        dynCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynCI.dynamicStateCount = 2; dynCI.pDynamicStates = dynStates;

        VkGraphicsPipelineCreateInfo gpCI{};
        gpCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpCI.stageCount = 2; gpCI.pStages = stages;
        gpCI.pVertexInputState = &viCI; gpCI.pInputAssemblyState = &iaCI; gpCI.pViewportState = &vpCI;
        gpCI.pRasterizationState = &rastCI; gpCI.pMultisampleState = &msCI; gpCI.pDepthStencilState = &dsCI; gpCI.pColorBlendState = &cbCI; gpCI.pDynamicState = &dynCI;
        gpCI.layout = pipelineLayout; gpCI.renderPass = ctx.render_pass; gpCI.subpass = 0;

        vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gpCI, nullptr, &graphicsPipeline);

        vkDestroyShaderModule(ctx.device, vertMod, nullptr);
        vkDestroyShaderModule(ctx.device, fragMod, nullptr);
        return graphicsPipeline != VK_NULL_HANDLE;
    }

    bool createDescriptors() {
        auto& ctx = graphics::internal::context;
        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; poolSizes[0].descriptorCount = 2;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; poolSizes[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo poolCI{};
        poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCI.poolSizeCount = 2; poolCI.pPoolSizes = poolSizes; poolCI.maxSets = 2;
        vkCreateDescriptorPool(ctx.device, &poolCI, nullptr, &descriptorPool);

        VkDescriptorSetAllocateInfo sceneAlloc{};
        sceneAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        sceneAlloc.descriptorPool = descriptorPool; sceneAlloc.descriptorSetCount = 1; sceneAlloc.pSetLayouts = &sceneSetLayout;
        vkAllocateDescriptorSets(ctx.device, &sceneAlloc, &sceneDescriptorSet);

        VkDescriptorSetAllocateInfo modelAlloc{};
        modelAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        modelAlloc.descriptorPool = descriptorPool; modelAlloc.descriptorSetCount = 1; modelAlloc.pSetLayouts = &modelSetLayout;
        vkAllocateDescriptorSets(ctx.device, &modelAlloc, &modelDescriptorSet);

        VkDescriptorBufferInfo sceneInfo{ sceneUBO, 0, sizeof(SceneUBO) };
        VkDescriptorBufferInfo pointInfo{ pointLightBuffer, 0, sizeof(PointLight) * MAX_LIGHTS };
        VkDescriptorBufferInfo spotInfo{ spotLightBuffer, 0, sizeof(SpotLight) * MAX_LIGHTS };
        VkDescriptorBufferInfo modelInfo{ modelUBO, 0, sizeof(ModelUBO) };

        VkWriteDescriptorSet writes[4]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[0].dstSet = sceneDescriptorSet; writes[0].dstBinding = 0; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[0].descriptorCount = 1; writes[0].pBufferInfo = &sceneInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[1].dstSet = sceneDescriptorSet; writes[1].dstBinding = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].descriptorCount = 1; writes[1].pBufferInfo = &pointInfo;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[2].dstSet = sceneDescriptorSet; writes[2].dstBinding = 2; writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[2].descriptorCount = 1; writes[2].pBufferInfo = &spotInfo;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[3].dstSet = modelDescriptorSet; writes[3].dstBinding = 0; writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; writes[3].descriptorCount = 1; writes[3].pBufferInfo = &modelInfo;

        vkUpdateDescriptorSets(ctx.device, 4, writes, 0, nullptr);
        return true;
    }

}

namespace application {

    bool initialize() {
        auto& ctx = graphics::internal::context;
        vertexBuffer = VK_NULL_HANDLE; vertexBufferAlloc = VK_NULL_HANDLE;
        indexBuffer = VK_NULL_HANDLE; indexBufferAlloc = VK_NULL_HANDLE;
        sceneUBO = VK_NULL_HANDLE; sceneUBOAlloc = VK_NULL_HANDLE; sceneUBOMapped = nullptr;
        modelUBO = VK_NULL_HANDLE; modelUBOAlloc = VK_NULL_HANDLE; modelUBOMapped = nullptr;
        pointLightBuffer = VK_NULL_HANDLE; pointLightAlloc = VK_NULL_HANDLE; pointLightMapped = nullptr;
        spotLightBuffer = VK_NULL_HANDLE; spotLightAlloc = VK_NULL_HANDLE; spotLightMapped = nullptr;

        std::vector<Vertex> vertices; std::vector<uint32_t> indices;
        generateSphere(vertices, indices, 9, 9); indexCount = (uint32_t)indices.size();
        VkDeviceSize vbSize = sizeof(Vertex) * vertices.size();
        VkDeviceSize ibSize = sizeof(uint32_t) * indices.size();

        createBuffer(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexBufferAlloc);
        {
            void* mapped = nullptr;
            vmaMapMemory(ctx.allocator, vertexBufferAlloc, &mapped);
            std::memcpy(mapped, vertices.data(), (size_t)vbSize);
            vmaUnmapMemory(ctx.allocator, vertexBufferAlloc);
        }

        createBuffer(ibSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexBufferAlloc);
        {
            void* mapped = nullptr;
            vmaMapMemory(ctx.allocator, indexBufferAlloc, &mapped);
            std::memcpy(mapped, indices.data(), (size_t)ibSize);
            vmaUnmapMemory(ctx.allocator, indexBufferAlloc);
        }

        createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, sceneUBO, sceneUBOAlloc, (void**)&sceneUBOMapped);
        createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, modelUBO, modelUBOAlloc, (void**)&modelUBOMapped);

        createBuffer(sizeof(PointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, pointLightBuffer, pointLightAlloc, (void**)&pointLightMapped);
        createBuffer(sizeof(SpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spotLightBuffer, spotLightAlloc, (void**)&spotLightMapped);

        std::memset(pointLightMapped, 0, sizeof(PointLight) * MAX_LIGHTS);
        std::memset(spotLightMapped, 0, sizeof(SpotLight) * MAX_LIGHTS);
        std::memset(sceneUBOMapped, 0, sizeof(SceneUBO));
        std::memset(modelUBOMapped, 0, sizeof(ModelUBO));

        pointLightMapped[0].pos[0] = 2.0f; pointLightMapped[0].pos[1] = 2.0f; pointLightMapped[0].pos[2] = 2.0f;
        pointLightMapped[0].color[0] = 1.0f; pointLightMapped[0].color[1] = 0.3f; pointLightMapped[0].color[2] = 0.3f;

        spotLightMapped[0].pos[0] = -2.0f; spotLightMapped[0].pos[1] = 3.0f; spotLightMapped[0].pos[2] = 2.0f;
        spotLightMapped[0].color[0] = 0.3f; spotLightMapped[0].color[1] = 0.3f; spotLightMapped[0].color[2] = 1.0f;
        spotLightMapped[0].dir[0] = 0.0f; spotLightMapped[0].dir[1] = -1.0f; spotLightMapped[0].dir[2] = 0.0f;
        spotLightMapped[0].params[0] = std::cos(15.0f * 3.14159f / 180.0f);
        spotLightMapped[0].params[1] = std::cos(30.0f * 3.14159f / 180.0f);

        if (!createPipeline()) return false;
        if (!createDescriptors()) return false;
        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        if (ctx.device != VK_NULL_HANDLE) vkDeviceWaitIdle(ctx.device);
        if (graphicsPipeline) vkDestroyPipeline(ctx.device, graphicsPipeline, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
        if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
        if (sceneSetLayout) vkDestroyDescriptorSetLayout(ctx.device, sceneSetLayout, nullptr);
        if (modelSetLayout) vkDestroyDescriptorSetLayout(ctx.device, modelSetLayout, nullptr);
        if (vertexBuffer) vmaDestroyBuffer(ctx.allocator, vertexBuffer, vertexBufferAlloc);
        if (indexBuffer) vmaDestroyBuffer(ctx.allocator, indexBuffer, indexBufferAlloc);
        if (sceneUBO) vmaDestroyBuffer(ctx.allocator, sceneUBO, sceneUBOAlloc);
        if (modelUBO) vmaDestroyBuffer(ctx.allocator, modelUBO, modelUBOAlloc);
        if (pointLightBuffer) vmaDestroyBuffer(ctx.allocator, pointLightBuffer, pointLightAlloc);
        if (spotLightBuffer) vmaDestroyBuffer(ctx.allocator, spotLightBuffer, spotLightAlloc);
    }

    void update(double time) {
        (void)time;
        ImGui::Begin("Scene");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Camera pos", cameraPos, -20.0f, 20.0f);
            ImGui::SliderFloat("FOV", &cameraFov, 10.0f, 120.0f);
        }
        if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient color", ambientColor);
        }
        if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Dir direction", dirDir, -1.0f, 1.0f);
            ImGui::ColorEdit3("Dir color", dirColor);
        }
        if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add point light") && pointLightCount < (int)MAX_LIGHTS) pointLightCount++;
            ImGui::SameLine();
            if (ImGui::Button("Remove point light") && pointLightCount > 0) pointLightCount--;
            for (int i = 0; i < pointLightCount; ++i) {
                ImGui::PushID(i);
                ImGui::Text("Point light %d", i);
                ImGui::SliderFloat3("Pos", pointLightMapped[i].pos, -10.0f, 10.0f);
                ImGui::ColorEdit3("Color", pointLightMapped[i].color);
                ImGui::PopID();
            }
        }
        if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add spot light") && spotLightCount < (int)MAX_LIGHTS) spotLightCount++;
            ImGui::SameLine();
            if (ImGui::Button("Remove spot light") && spotLightCount > 0) spotLightCount--;
            for (int i = 0; i < spotLightCount; ++i) {
                ImGui::PushID(i + 1000);
                ImGui::Text("Spot light %d", i);
                ImGui::SliderFloat3("Pos", spotLightMapped[i].pos, -10.0f, 10.0f);
                ImGui::ColorEdit3("Color", spotLightMapped[i].color);
                ImGui::SliderFloat3("Dir", spotLightMapped[i].dir, -1.0f, 1.0f);
                float innerDeg = std::acos(spotLightMapped[i].params[0]) * 180.0f / 3.14159f;
                float outerDeg = std::acos(spotLightMapped[i].params[1]) * 180.0f / 3.14159f;
                ImGui::SliderFloat("Inner angle", &innerDeg, 1.0f, 45.0f);
                ImGui::SliderFloat("Outer angle", &outerDeg, 1.0f, 60.0f);
                spotLightMapped[i].params[0] = std::cos(innerDeg * 3.14159f / 180.0f);
                spotLightMapped[i].params[1] = std::cos(outerDeg * 3.14159f / 180.0f);
                ImGui::PopID();
            }
        }
        if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo", albedo);
            ImGui::ColorEdit3("Specular color", specularColor);
        }
        if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Model position", modelPosition, -10.0f, 10.0f);
            ImGui::SliderFloat3("Model rotation", modelRotation, -180.0f, 180.0f);
            ImGui::SliderFloat3("Model scale", modelScale, 0.1f, 5.0f);
        }
        if (ImGui::CollapsingHeader("Animation")) {
            ImGui::Checkbox("Animate", &animate);
            ImGui::SliderFloat("Speed", &animSpeed, 0.0f, 5.0f);
            ImGui::SliderFloat("Radius", &animRadius, 0.5f, 5.0f);
        }
        ImGui::End();

        double dt = ImGui::GetIO().DeltaTime;
        if (animate) animTime += (float)dt * animSpeed;
        float px = modelPosition[0], py = modelPosition[1], pz = modelPosition[2];
        if (animate) { px = std::cos(animTime) * animRadius; pz = std::sin(animTime) * animRadius; }

        Mat4 model = Mat4::translation({ px, py, pz })
            * Mat4::rotationY(modelRotation[1] * 3.14159265f / 180.0f)
            * Mat4::rotationX(modelRotation[0] * 3.14159265f / 180.0f)
            * Mat4::rotationZ(modelRotation[2] * 3.14159265f / 180.0f)
            * Mat4::scale({ modelScale[0], modelScale[1], modelScale[2] });

        auto& ctx = graphics::internal::context;
        float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
        Mat4 view = Mat4::lookAt({ cameraPos[0], cameraPos[1], cameraPos[2] }, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
        Mat4 proj = Mat4::perspective(cameraFov * 3.14159265f / 180.0f, aspect, 0.1f, 100.0f);

        if (sceneUBOMapped) {
            sceneUBOMapped->viewProj = proj * view;
            sceneUBOMapped->cameraPos[0] = cameraPos[0]; sceneUBOMapped->cameraPos[1] = cameraPos[1]; sceneUBOMapped->cameraPos[2] = cameraPos[2]; sceneUBOMapped->cameraPos[3] = 1.0f;
            sceneUBOMapped->dirDir[0] = dirDir[0]; sceneUBOMapped->dirDir[1] = dirDir[1]; sceneUBOMapped->dirDir[2] = dirDir[2]; sceneUBOMapped->dirDir[3] = 0.0f;
            sceneUBOMapped->dirColor[0] = dirColor[0]; sceneUBOMapped->dirColor[1] = dirColor[1]; sceneUBOMapped->dirColor[2] = dirColor[2]; sceneUBOMapped->dirColor[3] = 1.0f;
            sceneUBOMapped->ambient[0] = ambientColor[0]; sceneUBOMapped->ambient[1] = ambientColor[1]; sceneUBOMapped->ambient[2] = ambientColor[2]; sceneUBOMapped->ambient[3] = 1.0f;
            sceneUBOMapped->pointCount = pointLightCount;
            sceneUBOMapped->spotCount = spotLightCount;
        }

        if (modelUBOMapped) {
            modelUBOMapped->model = model;
            modelUBOMapped->albedo[0] = albedo[0]; modelUBOMapped->albedo[1] = albedo[1]; modelUBOMapped->albedo[2] = albedo[2]; modelUBOMapped->albedo[3] = 1.0f;
            modelUBOMapped->specular[0] = specularColor[0]; modelUBOMapped->specular[1] = specularColor[1]; modelUBOMapped->specular[2] = specularColor[2]; modelUBOMapped->specular[3] = 1.0f;
        }
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;
        VkCommandBuffer cmd = fd.command_buffer;
        if (graphicsPipeline == VK_NULL_HANDLE) return;

        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginCI{}; beginCI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginCI);

        VkClearValue clearValues[2]{}; clearValues[0].color = { {0.05f, 0.05f, 0.08f, 1.0f} }; clearValues[1].depthStencil = { 1.0f, 0 };
        VkRenderPassBeginInfo rpCI{}; rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpCI.renderPass = ctx.render_pass; rpCI.framebuffer = fd.framebuffer; rpCI.renderArea.extent = ctx.swapchain_extent;
        rpCI.clearValueCount = 2; rpCI.pClearValues = clearValues;
        vkCmdBeginRenderPass(cmd, &rpCI, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{}; viewport.width = (float)ctx.swapchain_extent.width; viewport.height = (float)ctx.swapchain_extent.height; viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{}; scissor.extent = ctx.swapchain_extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkDescriptorSet sets[2] = { sceneDescriptorSet, modelDescriptorSet };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 2, sets, 0, nullptr);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

}