// Compare ggml_flash_attn_ext on Vulkan against the CPU backend for the
// MOSS-Image attention shape family (D=128, f32 q, f16 k/v, f32 accumulation).
// usage: attn_check L H [iters]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

struct Runner {
    ggml_backend_t backend    = nullptr;
    ggml_context* ctx         = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph* gf           = nullptr;
    ggml_tensor* out          = nullptr;
    std::vector<float> host;
};

static bool setup(Runner& r, ggml_backend_t backend, int D, int L, int H,
                  const std::vector<float>& q, const std::vector<ggml_fp16_t>& k,
                  const std::vector<ggml_fp16_t>& v) {
    r.backend           = backend;
    ggml_init_params ip = { 4 * 1024 * 1024, nullptr, true };
    r.ctx               = ggml_init(ip);
    auto tq             = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, D, L, H, 1);
    auto tk             = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F16, D, L, H, 1);
    auto tv             = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F16, D, L, H, 1);
    r.out               = ggml_flash_attn_ext(r.ctx, tq, tk, tv, nullptr, 1.0f / sqrtf((float) D), 0, 0);
    ggml_prec_set_acc(r.out, GGML_PREC_F32);
    r.buf = ggml_backend_alloc_ctx_tensors(r.ctx, backend);
    if (!r.buf) {
        fprintf(stderr, "alloc failed\n");
        return false;
    }
    ggml_backend_tensor_set(tq, q.data(), 0, q.size() * sizeof(float));
    ggml_backend_tensor_set(tk, k.data(), 0, k.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(tv, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
    r.gf = ggml_new_graph(r.ctx);
    ggml_build_forward_expand(r.gf, r.out);
    return true;
}

int main(int argc, char** argv) {
    const int L     = argc > 1 ? atoi(argv[1]) : 512;
    const int H     = argc > 2 ? atoi(argv[2]) : 4;
    const int iters = argc > 3 ? atoi(argv[3]) : 3;
    const bool noref = argc > 4 && !strcmp(argv[4], "noref");
    const int D     = 128;

    std::vector<float> q((size_t) D * L * H);
    std::vector<ggml_fp16_t> k((size_t) D * L * H), v((size_t) D * L * H);
    uint32_t state = 12345;
    auto next      = [&]() {
        state = state * 1664525u + 1013904223u;
        return (float) (state >> 8) / 16777216.f - 0.5f;
    };
    for (auto& e : q) e = next();
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = ggml_fp32_to_fp16(next() * 0.5f);
        v[i] = ggml_fp32_to_fp16(next() * 0.75f);
    }

    ggml_backend_load_all();
    ggml_backend_t cpu         = nullptr;
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const auto type      = ggml_backend_dev_type(d);
        if (type == GGML_BACKEND_DEVICE_TYPE_CPU && !cpu) {
            cpu = ggml_backend_dev_init(d, nullptr);
        }
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_dev_name(d), "Vulkan")) {
            gpu_dev = d;
        }
    }
    if (!cpu || !gpu_dev) {
        fprintf(stderr, "backends not found\n");
        return 1;
    }
    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);

    Runner rc, rg;
    if (!setup(rc, cpu, D, L, H, q, k, v) || !setup(rg, gpu, D, L, H, q, k, v)) {
        return 1;
    }
    if (ggml_backend_graph_compute(rc.backend, rc.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "cpu compute failed\n");
        return 1;
    }
    if (ggml_backend_graph_compute(rg.backend, rg.gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "gpu compute failed\n");
        return 1;
    }
    ggml_backend_synchronize(rg.backend);

    if (noref) {
        double t0 = now_ms();
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(rg.backend, rg.gf);
        }
        ggml_backend_synchronize(rg.backend);
        const double ms = (now_ms() - t0) / iters;
        printf("L=%d H=%d D=%d  vulkan time %.2f ms  (%.0f GFLOP/s)\n", L, H, D, ms,
               4.0 * (double) L * L * D * H / ms / 1e6);
        return 0;
    }

    rc.host.resize((size_t) D * L * H);
    rg.host.resize(rc.host.size());
    ggml_backend_tensor_get(rc.out, rc.host.data(), 0, rc.host.size() * sizeof(float));
    ggml_backend_tensor_get(rg.out, rg.host.data(), 0, rg.host.size() * sizeof(float));

    double max_abs = 0, max_ref = 0, sum_sq = 0;
    for (size_t i = 0; i < rc.host.size(); ++i) {
        const double d = (double) rg.host[i] - rc.host[i];
        max_abs        = std::max(max_abs, fabs(d));
        max_ref        = std::max(max_ref, fabs((double) rc.host[i]));
        sum_sq += d * d;
    }
    printf("L=%d H=%d D=%d  gpu-vs-cpu: max_abs=%.3e  max_rel=%.3e  rms=%.3e (ref max %.3f)\n",
           L, H, D, max_abs, max_abs / (max_ref > 0 ? max_ref : 1), sqrt(sum_sq / rc.host.size()), max_ref);

    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(rg.backend, rg.gf);
    }
    ggml_backend_synchronize(rg.backend);
    const double ms = (now_ms() - t0) / iters;
    printf("vulkan time %.2f ms  (%.0f GFLOP/s)\n", ms, 4.0 * (double) L * L * D * H / ms / 1e6);
    return 0;
}
