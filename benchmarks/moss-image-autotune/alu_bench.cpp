// Minimal Vulkan compute runner for alu_bench.comp (fp16 dot2 vs int8 dp4a).
//
// build:
//   glslc -O -o /tmp/alu_f16.spv benchmarks/.../alu_bench.comp
//   glslc -O -DINT8 -o /tmp/alu_i8.spv benchmarks/.../alu_bench.comp
//   g++ -O2 -std=c++17 alu_bench.cpp -lvulkan -o /tmp/alu_bench
// usage: alu_bench /tmp/alu_f16.spv [iters] [wg] [threads]
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

#define VK_CHECK(x)                                                                  \
    do {                                                                             \
        VkResult res = (x);                                                          \
        if (res != VK_SUCCESS) {                                                     \
            fprintf(stderr, "vk error %d at %s:%d\n", (int) res, __FILE__, __LINE__); \
            exit(1);                                                                 \
        }                                                                            \
    } while (0)

static std::vector<char> read_file(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t n = (size_t) f.tellg();
    std::vector<char> buf(n);
    f.seekg(0);
    f.read(buf.data(), (std::streamsize) n);
    return buf;
}

int main(int argc, char ** argv) {
    const char * spv_path = argc > 1 ? argv[1] : "/tmp/alu_f16.spv";
    const uint32_t iters   = argc > 2 ? (uint32_t) atoi(argv[2]) : 20000;
    const uint32_t wg      = argc > 3 ? (uint32_t) atoi(argv[3]) : 1024;   // workgroups
    const uint32_t threads = argc > 4 ? (uint32_t) atoi(argv[4]) : 0;      // dispatch threads
    const uint32_t n_threads = threads ? threads : wg * 64;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t ndev = 1;
    VkPhysicalDevice phys;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, &phys));
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos    = &qci;
    VkDevice dev;
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    // host-visible buffers
    const size_t n_a = 1024, n_d = n_threads;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = n_a * sizeof(uint32_t);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buf_a;
    VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf_a));
    VkMemoryRequirements mra;
    vkGetBufferMemoryRequirements(dev, buf_a, &mra);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mra.size;
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t host_idx = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((mra.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            host_idx = i;
            break;
        }
    }
    mai.memoryTypeIndex = host_idx;
    VkDeviceMemory mem_a, mem_d;
    VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem_a));
    VK_CHECK(vkBindBufferMemory(dev, buf_a, mem_a, 0));

    bci.size = n_d * sizeof(uint32_t);
    VkBuffer buf_d;
    VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf_d));
    VkMemoryRequirements mrd;
    vkGetBufferMemoryRequirements(dev, buf_d, &mrd);
    mai.allocationSize = mrd.size;
    VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem_d));
    VK_CHECK(vkBindBufferMemory(dev, buf_d, mem_d, 0));

    void * pa;
    VK_CHECK(vkMapMemory(dev, mem_a, 0, VK_WHOLE_SIZE, 0, &pa));
    for (size_t i = 0; i < n_a; ++i) ((uint32_t *) pa)[i] = (uint32_t) (i * 2654435761u + 12345u);

    // descriptors
    VkDescriptorSetLayoutBinding binds[2]{};
    for (int i = 0; i < 2; ++i) {
        binds[i].binding         = (uint32_t) i;
        binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslc.bindingCount = 2;
    dslc.pBindings    = binds;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslc, nullptr, &dsl));

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpc.maxSets       = 1;
    dpc.poolSizeCount = 1;
    dpc.pPoolSizes    = &pool_size;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &dpc, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet dset;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
    VkDescriptorBufferInfo dbi[2] = {{buf_a, 0, VK_WHOLE_SIZE}, {buf_d, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[2]{};
    for (int i = 0; i < 2; ++i) {
        wds[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet          = dset;
        wds[i].dstBinding      = (uint32_t) i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo     = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 2, wds, 0, nullptr);

    // pipeline
    auto code = read_file(spv_path);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode    = (const uint32_t *) code.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plc.setLayoutCount         = 1;
    plc.pSetLayouts            = &dsl;
    plc.pushConstantRangeCount = 1;
    plc.pPushConstantRanges    = &pcr;
    VkPipelineLayout playout;
    VK_CHECK(vkCreatePipelineLayout(dev, &plc, nullptr, &playout));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName  = "main";
    cpci.layout       = playout;
    VkPipeline pipe;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpc.queueFamilyIndex = 0;
    VkCommandPool cmdpool;
    VK_CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &cmdpool));
    VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool        = cmdpool;
    cba.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));

    auto dispatch = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(iters), &iters);
        vkCmdDispatch(cmd, wg, 1, 1);
        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    };

    dispatch();  // warmup
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        const double t0 = now_ms();
        dispatch();
        const double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }

    // 8 chains per thread per iteration; fp16 does 4 MACs in 2 instructions,
    // int8 does 4 MACs in 1 instruction.
    const double macs   = (double) n_threads * iters * 8.0 * 4.0;
    const double instrs = (double) n_threads * iters * 8.0;
    printf("%-28s %8.3f ms  %7.1f GMAC/s  %7.1f G instr/s\n", spv_path, best, macs / best / 1e6,
           instrs / best / 1e6);

    vkDeviceWaitIdle(dev);
    vkDestroyFence(dev, fence, nullptr);
    vkDestroyCommandPool(dev, cmdpool, nullptr);
    vkDestroyPipeline(dev, pipe, nullptr);
    vkDestroyPipelineLayout(dev, playout, nullptr);
    vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyBuffer(dev, buf_a, nullptr);
    vkDestroyBuffer(dev, buf_d, nullptr);
    vkFreeMemory(dev, mem_a, nullptr);
    vkFreeMemory(dev, mem_d, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    (void) props;
    return 0;
}
