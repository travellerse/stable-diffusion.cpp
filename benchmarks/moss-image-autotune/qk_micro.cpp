// Host for qk_micro.comp: runs the FA QK pass (f16 dot2 vs int8 dp4a) on real
// Q/K dumps and reports time, plus the first tile's scores for validation.
//
// build:
//   glslc -O        -o /tmp/qk_f16.spv benchmarks/.../qk_micro.comp
//   glslc -O -DINT8 -o /tmp/qk_i8.spv  benchmarks/.../qk_micro.comp
//   g++ -O2 -std=c++17 qk_micro.cpp -lvulkan -o /tmp/qk_micro
// usage: qk_micro <q.bin> <k.bin> <N> <KV> <f16|int8> <rounds> [s_out.bin]
//   q.bin/k.bin: f32, row-major [rows][128]
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

#define VK_CHECK(x)                                                                   \
    do {                                                                              \
        VkResult res = (x);                                                           \
        if (res != VK_SUCCESS) {                                                      \
            fprintf(stderr, "vk error %d at %s:%d\n", (int) res, __FILE__, __LINE__);  \
            exit(1);                                                                  \
        }                                                                             \
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

static std::vector<float> load_f32(const char * path, size_t n) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return v;
}

struct Buf {
    VkBuffer buf;
    VkDeviceMemory mem;
    void * mapped = nullptr;
    size_t bytes  = 0;
};

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s q.bin k.bin N KV f16|int8 rounds [s_out.bin]\n", argv[0]);
        return 1;
    }
    const char * qpath = argv[1];
    const char * kpath = argv[2];
    const uint32_t N   = (uint32_t) atoi(argv[3]);
    const uint32_t KV  = (uint32_t) atoi(argv[4]);
    const bool int8    = std::string(argv[5]) == "int8";
    const int rounds   = atoi(argv[6]);
    const char * sout  = argc > 7 ? argv[7] : nullptr;
    const uint32_t HSK = 128;

    auto qh = load_f32(qpath, (size_t) N * HSK);
    auto kh = load_f32(kpath, (size_t) KV * HSK);

    const uint32_t words_per_row = int8 ? HSK / 4 : HSK / 2;
    std::vector<uint32_t> qw((size_t) N * words_per_row), kw((size_t) KV * words_per_row);
    std::vector<float> qs(N, 1.f), ks(KV, 1.f);

    for (uint32_t r = 0; r < N; ++r) {
        const float * src = qh.data() + (size_t) r * HSK;
        if (int8) {
            float amax = 0.f;
            for (uint32_t d = 0; d < HSK; ++d) amax = std::max(amax, std::fabs(src[d]));
            qs[r] = amax > 0 ? amax / 127.f : 1.f;
            for (uint32_t w = 0; w < words_per_row; ++w) {
                uint32_t packed = 0;
                for (int b = 0; b < 4; ++b) {
                    const float v = src[w * 4 + b] / qs[r];
                    int q = (int) std::lrint(v);
                    q = std::max(-127, std::min(127, q));
                    packed |= (uint32_t) (uint8_t) (int8_t) q << (8 * b);
                }
                qw[(size_t) r * words_per_row + w] = packed;
            }
        } else {
            for (uint32_t w = 0; w < words_per_row; ++w) {
                uint32_t lo = 0, hi = 0;
                memcpy(&lo, &src[w * 2 + 0], 4);  // f32 -> not f16; converted below
                memcpy(&hi, &src[w * 2 + 1], 4);
                (void) lo; (void) hi;
                uint32_t packed = 0;
                for (int b = 0; b < 2; ++b) {
                    // convert f32 to f16 bits
                    const float v = src[w * 2 + b];
                    uint32_t bits;
                    // simple f32->f16 conversion (round to nearest even, no denormals)
                    const uint32_t f = *(const uint32_t *) &v;
                    const uint32_t sign = (f >> 31) & 1u;
                    int32_t exp = (int32_t) ((f >> 23) & 0xff) - 127;
                    uint32_t mant = f & 0x7fffffu;
                    uint32_t h;
                    if (exp > 15) { h = (sign << 15) | (0x1fu << 10); }
                    else if (exp < -14) { h = (sign << 15); }
                    else { h = (sign << 15) | ((uint32_t) (exp + 15) << 10) | (mant >> 13); }
                    bits = h;
                    packed |= bits << (16 * b);
                }
                qw[(size_t) r * words_per_row + w] = packed;
            }
        }
    }
    for (uint32_t r = 0; r < KV; ++r) {
        const float * src = kh.data() + (size_t) r * HSK;
        if (int8) {
            float amax = 0.f;
            for (uint32_t d = 0; d < HSK; ++d) amax = std::max(amax, std::fabs(src[d]));
            ks[r] = amax > 0 ? amax / 127.f : 1.f;
            for (uint32_t w = 0; w < words_per_row; ++w) {
                uint32_t packed = 0;
                for (int b = 0; b < 4; ++b) {
                    const float v = src[w * 4 + b] / ks[r];
                    int q = (int) std::lrint(v);
                    q = std::max(-127, std::min(127, q));
                    packed |= (uint32_t) (uint8_t) (int8_t) q << (8 * b);
                }
                kw[(size_t) r * words_per_row + w] = packed;
            }
        } else {
            for (uint32_t w = 0; w < words_per_row; ++w) {
                uint32_t packed = 0;
                for (int b = 0; b < 2; ++b) {
                    const float v = src[w * 2 + b];
                    const uint32_t f = *(const uint32_t *) &v;
                    const uint32_t sign = (f >> 31) & 1u;
                    int32_t exp = (int32_t) ((f >> 23) & 0xff) - 127;
                    uint32_t mant = f & 0x7fffffu;
                    uint32_t h;
                    if (exp > 15) { h = (sign << 15) | (0x1fu << 10); }
                    else if (exp < -14) { h = (sign << 15); }
                    else { h = (sign << 15) | ((uint32_t) (exp + 15) << 10) | (mant >> 13); }
                    packed |= h << (16 * b);
                }
                kw[(size_t) r * words_per_row + w] = packed;
            }
        }
    }

    const char * spv = int8 ? "/tmp/qk_i8.spv" : "/tmp/qk_f16.spv";

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t ndev = 1;
    VkPhysicalDevice phys;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ndev, &phys));

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    auto make_buf = [&](size_t bytes) {
        Buf b{};
        b.bytes = bytes;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &b.buf));
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(dev, b.buf, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((mr.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                mai.memoryTypeIndex = i;
                break;
            }
        }
        VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &b.mem));
        VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
        VK_CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.mapped));
        return b;
    };
    Buf bq = make_buf(qw.size() * 4), bk = make_buf(kw.size() * 4), bs = make_buf((size_t) 8 * KV * 4);
    Buf bqs = make_buf((size_t) N * 4), bks = make_buf((size_t) KV * 4);
    memcpy(bq.mapped, qw.data(), qw.size() * 4);
    memcpy(bk.mapped, kw.data(), kw.size() * 4);
    memcpy(bqs.mapped, qs.data(), qs.size() * 4);
    memcpy(bks.mapped, ks.data(), ks.size() * 4);

    VkDescriptorSetLayoutBinding binds[5]{};
    for (int i = 0; i < 5; ++i) {
        binds[i].binding = (uint32_t) i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslc.bindingCount = 5;
    dslc.pBindings = binds;
    VkDescriptorSetLayout dsl;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslc, nullptr, &dsl));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
    VkDescriptorPoolCreateInfo dpc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpc.maxSets = 1;
    dpc.poolSizeCount = 1;
    dpc.pPoolSizes = &ps;
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &dpc, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet dset;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &dset));
    VkDescriptorBufferInfo dbi[5] = {{bq.buf, 0, VK_WHOLE_SIZE}, {bk.buf, 0, VK_WHOLE_SIZE},
                                     {bs.buf, 0, VK_WHOLE_SIZE}, {bqs.buf, 0, VK_WHOLE_SIZE},
                                     {bks.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wds[5]{};
    for (int i = 0; i < 5; ++i) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = dset;
        wds[i].dstBinding = (uint32_t) i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 5, wds, 0, nullptr);

    auto code = read_file(spv);
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode = (const uint32_t *) code.data();
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    struct Push { uint32_t N, KV; float scale; } push{N, KV, 1.0f / std::sqrt((float) HSK)};
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push)};
    VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plc.setLayoutCount = 1;
    plc.pSetLayouts = &dsl;
    plc.pushConstantRangeCount = 1;
    plc.pPushConstantRanges = &pcr;
    VkPipelineLayout playout;
    VK_CHECK(vkCreatePipelineLayout(dev, &plc, nullptr, &playout));
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = playout;
    VkPipeline pipe;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpc.queueFamilyIndex = 0;
    VkCommandPool cmdpool;
    VK_CHECK(vkCreateCommandPool(dev, &cpc, nullptr, &cmdpool));
    VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cba.commandPool = cmdpool;
    cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cba.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));

    const uint32_t wg = (N + 7) / 8;
    auto dispatch = [&]() {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, wg, 1, 1);
        VK_CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    };

    dispatch();
    std::vector<double> times;
    for (int i = 0; i < rounds; ++i) {
        const double t0 = now_ms();
        dispatch();
        times.push_back(now_ms() - t0);
    }
    std::sort(times.begin(), times.end());
    const double med = times[times.size() / 2];
    const double flops = 2.0 * (double) N * KV * HSK;
    printf("%-5s N=%u KV=%u  median %.2f ms  %.0f GFLOP/s  (min %.2f)\n", int8 ? "int8" : "f16", N, KV,
           med, flops / med / 1e6, times.front());

    if (sout != nullptr) {
        FILE * f = fopen(sout, "wb");
        if (f) { fwrite(bs.mapped, 1, (size_t) 8 * KV * 4, f); fclose(f); }
    }
    vkDeviceWaitIdle(dev);
    return 0;
}
