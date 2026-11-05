// Microbenchmark for the Vulkan elementwise ops that dominate MOSS-Image's
// non-GEMM/non-attention time: MUL (same shape and broadcast), ADD, SCALE, TANH.
// Shapes are the real 1440x1920 activation shapes (tokens x hidden).
//
// build:
//   g++ -O2 -std=c++17 -fopenmp -Iggml/include -Iggml/src elem_bench.cpp \
//     -Wl,--start-group build-vulkan/ggml/src/libggml-base.a \
//     build-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a \
//     build-vulkan/ggml/src/libggml-cpu.a build-vulkan/ggml/src/libggml.a \
//     -Wl,--end-group -lvulkan -lpthread -ldl -o /tmp/elem_bench
// usage: elem_bench [iters]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static double bench(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * out, int iters) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(backend, gf);
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, gf);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20;
    const int64_t ne = 3072, rows = 10842;
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) { fprintf(stderr, "no vulkan backend\n"); return 1; }

    struct Case { const char * name; int kind; ggml_type ta, tb; int64_t nbb; };
    const Case cases[] = {
        { "mul_same_f32",   0, GGML_TYPE_F32, GGML_TYPE_F32, ne },
        { "mul_bcast_f32",  0, GGML_TYPE_F32, GGML_TYPE_F32, 1 },   // b is [ne] broadcast over rows
        { "add_same_f32",   1, GGML_TYPE_F32, GGML_TYPE_F32, ne },
        { "mul_same_f16",   0, GGML_TYPE_F16, GGML_TYPE_F16, ne },
        { "add_same_f16",   1, GGML_TYPE_F16, GGML_TYPE_F16, ne },
        { "scale_f32",      2, GGML_TYPE_F32, GGML_TYPE_F32, 0 },
        { "tanh_f32",       3, GGML_TYPE_F32, GGML_TYPE_F32, 0 },
    };
    printf("%-16s %9s %9s %9s %10s\n", "case", "ms", "GB/s", "elem", "note");
    for (const Case & c : cases) {
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * a = ggml_new_tensor_2d(ctx, c.ta, ne, rows);
        ggml_tensor * out = nullptr;
        if (c.kind == 0 || c.kind == 1) {
            ggml_tensor * b = c.nbb == ne ? ggml_new_tensor_2d(ctx, c.tb, ne, rows)
                                           : ggml_new_tensor_1d(ctx, c.tb, ne);
            out = c.kind == 0 ? ggml_mul(ctx, a, b) : ggml_add(ctx, a, b);
        } else if (c.kind == 2) {
            out = ggml_scale(ctx, a, 1.5f);
        } else {
            out = ggml_tanh(ctx, a);
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
        const size_t esz = ggml_type_size(c.ta);
        std::vector<char> zeros((size_t) ne * rows * esz, 0);
        ggml_backend_tensor_set(a, zeros.data(), 0, zeros.size());
        double ms = bench(backend, ctx, out, iters);
        double bytes = 2.0 * (double) ne * rows * esz;  // read + write
        if (c.kind <= 1) bytes += (double) c.nbb * rows * esz;
        printf("%-16s %9.3f %9.1f %9.0f %10s\n", c.name, ms, bytes / ms / 1e6,
               (double) ne * rows, c.kind <= 1 ? "-" : "unary");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return 0;
}
