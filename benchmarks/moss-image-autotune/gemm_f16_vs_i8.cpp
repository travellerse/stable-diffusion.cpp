// Throughput comparison for the two matmul paths an INT8-QK attention would
// choose between on RDNA2: f16 x f16 mul_mat (dot2/f16 FMA path) and the
// tensorwise int8 kernel (dp4a path) used by MOSS-Image.
//
// build:
//   g++ -O2 -std=c++17 -fopenmp -Iggml/include -Iggml/src gemm_f16_vs_i8.cpp \
//     -Wl,--start-group build-vulkan/ggml/src/libggml-base.a \
//     build-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a \
//     build-vulkan/ggml/src/libggml-cpu.a build-vulkan/ggml/src/libggml.a \
//     -Wl,--end-group -lvulkan -lpthread -ldl -o /tmp/gemm_f16_vs_i8
// usage: gemm_f16_vs_i8 [K] [M] [N] [iters]
#include <chrono>
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
        double t = now_ms() - t0;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char ** argv) {
    const int K     = argc > 1 ? atoi(argv[1]) : 3072;   // reduction dim (d_head for QK)
    const int M     = argc > 2 ? atoi(argv[2]) : 3072;   // output rows
    const int N     = argc > 3 ? atoi(argv[3]) : 10812;  // tokens
    const int iters = argc > 4 ? atoi(argv[4]) : 10;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) { fprintf(stderr, "no vulkan backend\n"); return 1; }

    const double flops = 2.0 * K * M * N;

    // --- f16 mul_mat: a[M,K] . b[K,N]
    {
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * a     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, M);
        ggml_tensor * b     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
        ggml_tensor * out   = ggml_mul_mat(ctx, a, b);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<char> zeros((size_t) ggml_nbytes(a) + ggml_nbytes(b), 0);
        ggml_backend_tensor_set(a, zeros.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, zeros.data(), 0, ggml_nbytes(b));
        double ms = bench(backend, ctx, out, iters);
        printf("f16  mul_mat K=%d M=%d N=%d: %8.2f ms  %7.1f TFLOP/s\n", K, M, N, ms, flops / ms / 1e9);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // --- int8 tensorwise matmul (same layout the DiT uses)
    {
        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, true };
        ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * a     = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, M);
        ggml_tensor * b     = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, K, N);
        ggml_tensor * out   = ggml_mul_mat(ctx, a, b);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "i8 alloc failed\n"); return 1; }
        std::vector<char> zeros((size_t) ggml_nbytes(a) + ggml_nbytes(b), 0);
        ggml_backend_tensor_set(a, zeros.data(), 0, ggml_nbytes(a));
        ggml_backend_tensor_set(b, zeros.data(), 0, ggml_nbytes(b));
        double ms = bench(backend, ctx, out, iters);
        printf("i8   mul_mat K=%d M=%d N=%d: %8.2f ms  %7.1f TFLOP/s\n", K, M, N, ms, flops / ms / 1e9);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    // --- per-row int8 quantization pass cost (Q/K preprocessing for INT8 QK)
    {
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * ctx  = ggml_init(ip);
        ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_tensor * out   = ggml_scale(ctx, x, 1.0f);  // same traffic as a quantize pass
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        std::vector<char> zeros((size_t) ggml_nbytes(x), 0);
        ggml_backend_tensor_set(x, zeros.data(), 0, ggml_nbytes(x));
        double ms = bench(backend, ctx, out, iters);
        printf("f32 pass over %d x %d: %8.3f ms (quantize-pass cost proxy)\n", K, N, ms);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    ggml_backend_free(backend);
    return 0;
}
