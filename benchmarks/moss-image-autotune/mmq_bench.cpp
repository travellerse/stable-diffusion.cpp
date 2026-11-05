// Baseline throughput for the K-quant matmuls Qwen-Image-2.1 issues per step.
// usage: mmq_bench [--iters N] [--shape all|q5|q6] [--type q5_K|q6_K]
#include <algorithm>
#include <chrono>
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

struct Shape {
    const char* name;
    ggml_type type;
    int64_t k;     // in features
    int64_t n;     // out features (weight rows)
    int64_t rows;  // tokens
};

int main(int argc, char** argv) {
    int iters       = 5;
    const char* which = "all";
    ggml_type forced_type = GGML_TYPE_COUNT;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--shape") && i + 1 < argc) {
            which = argv[++i];
        } else if (!strcmp(argv[i], "--type") && i + 1 < argc) {
            const char* t = argv[++i];
            if (!strcmp(t, "q5_K")) forced_type = GGML_TYPE_Q5_K;
            else if (!strcmp(t, "q6_K")) forced_type = GGML_TYPE_Q6_K;
            else if (!strcmp(t, "q8_0")) forced_type = GGML_TYPE_Q8_0;
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && strstr(ggml_backend_dev_name(d), "Vulkan")) {
            dev = d;
            break;
        }
    }
    if (!dev) {
        fprintf(stderr, "vulkan device not found\n");
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    std::vector<Shape> shapes = {
        { "q5_K", GGML_TYPE_Q5_K, 4096, 24576, 4096 },
        { "q6_K", GGML_TYPE_Q6_K, 4096, 4096, 4096 },
    };
    printf("%-6s %5s %7s %7s %5s %10s %8s\n", "shape", "k", "n", "rows", "type", "gemm_ms", "GFLOP/s");
    for (Shape s : shapes) {
        if (strcmp(which, "all") && strcmp(which, s.name)) {
            continue;
        }
        if (forced_type != GGML_TYPE_COUNT) {
            // Compare a shape under a different weight type.
            s.type = forced_type;
        }
        if (s.k % ggml_blck_size(s.type) != 0) {
            fprintf(stderr, "%s: k=%lld not a multiple of %lld\n", s.name, (long long) s.k,
                    (long long) ggml_blck_size(s.type));
            continue;
        }
        ggml_init_params p = { 16 * 1024 * 1024, nullptr, true };
        ggml_context* ctx  = ggml_init(p);
        ggml_tensor* w     = ggml_new_tensor_2d(ctx, s.type, s.k, s.n);
        ggml_tensor* x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.rows);
        ggml_tensor* y     = ggml_mul_mat(ctx, w, x);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "%s: alloc failed (%.2f GB)\n", s.name,
                    (double) (ggml_nbytes(w) + ggml_nbytes(x) + ggml_nbytes(y)) / 1e9);
            ggml_free(ctx);
            continue;
        }
        std::vector<char> zero_w(ggml_nbytes(w), 0), zero_x(ggml_nbytes(x), 0);
        ggml_backend_tensor_set(w, zero_w.data(), 0, zero_w.size());
        ggml_backend_tensor_set(x, zero_x.data(), 0, zero_x.size());
        if (!ggml_backend_supports_op(backend, y)) {
            fprintf(stderr, "%s: op unsupported\n", s.name);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", s.name);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(backend);
        const double t0 = now_ms();
        for (int it = 0; it < iters; ++it) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);
        const double ms = (now_ms() - t0) / iters;
        const double flops = 2.0 * (double) s.k * (double) s.n * (double) s.rows;
        printf("%-6s %5lld %7lld %7lld %5s %10.2f %8.1f\n", s.name, (long long) s.k, (long long) s.n,
               (long long) s.rows, ggml_type_name(s.type), ms, flops / ms / 1e6);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    return 0;
}
