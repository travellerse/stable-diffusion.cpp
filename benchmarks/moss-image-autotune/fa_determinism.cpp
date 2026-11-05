// Repeat the Flash Attention op on fixed, real Q/K/V and report whether the
// output is bit-stable across calls.
//
// build:
//   g++ -O2 -std=c++17 -fopenmp -Iggml/include -Iggml/src fa_determinism.cpp \
//     -Wl,--start-group build-vulkan/ggml/src/libggml-base.a \
//     build-vulkan/ggml/src/libggml-vulkan.a build-vulkan/ggml/src/libggml-cpu.a \
//     build-vulkan/ggml/src/libggml.a -Wl,--end-group -lvulkan -lpthread -ldl \
//     -o /tmp/fa_determinism
// usage: fa_determinism q.bin k.bin v.bin L iter
//   q.bin  f32 [128, L, 24]   (post-RoPE, as consumed by FA)
//   k.bin  f32 [128, L, 24]
//   v.bin  f32 [128, 24, L]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>

#include "ggml-backend.h"
#include "ggml.h"

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

static std::vector<float> load(const char * path, size_t n) {
    std::vector<float> v(n);
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    if (fread(v.data(), sizeof(float), n, f) != n) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s q.bin k.bin v.bin L [iters]\n", argv[0]);
        return 1;
    }
    const int64_t L     = atoll(argv[4]);
    const int iters     = argc > 5 ? atoi(argv[5]) : 8;
    const int64_t D     = 128, H = 24;

    auto qh = load(argv[1], (size_t) D * L * H);
    auto kh = load(argv[2], (size_t) D * L * H);
    auto vh = load(argv[3], (size_t) D * H * L);

    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) { fprintf(stderr, "no vulkan backend\n"); return 1; }

    ggml_init_params ip = { (size_t) 256 * 1024 * 1024, nullptr, true };
    ggml_context * ctx  = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, H);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, L, H);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, L);

    // mirror ggml_ext_attention_ext / build_kqv: k cast to f16, v permuted then cast
    ggml_tensor * k_f16 = ggml_cast(ctx, k, GGML_TYPE_F16);
    ggml_tensor * v_p   = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * v_c   = ggml_cont(ctx, v_p);
    ggml_tensor * v_f16 = ggml_cast(ctx, v_c, GGML_TYPE_F16);
    ggml_tensor * v_in  = ggml_reshape_3d(ctx, v_f16, D, L, H);

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k_f16, v_in, nullptr, 1.0f / sqrtf((float) D), 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
    ggml_backend_tensor_set(q, qh.data(), 0, qh.size() * sizeof(float));
    ggml_backend_tensor_set(k, kh.data(), 0, kh.size() * sizeof(float));
    ggml_backend_tensor_set(v, vh.data(), 0, vh.size() * sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    std::vector<float> ref((size_t) D * L * H), cur(ref.size());
    std::map<size_t, std::map<float, int>> seen;  // idx -> value -> count
    for (int it = 0; it < iters; ++it) {
        const double t0 = now_ms();
        if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n");
            return 1;
        }
        ggml_backend_synchronize(gpu);
        const double ms = now_ms() - t0;
        ggml_backend_tensor_get(out, cur.data(), 0, cur.size() * sizeof(float));
        // second readback after another sync: distinguishes kernel nondeterminism
        // from a racy host-side readback path
        ggml_backend_synchronize(gpu);
        std::vector<float> cur2(cur.size());
        ggml_backend_tensor_get(out, cur2.data(), 0, cur2.size() * sizeof(float));
        {
            size_t rb = 0;
            for (size_t i = 0; i < cur.size(); ++i) {
                if (cur[i] != cur2[i]) ++rb;
            }
            if (rb != 0) {
                printf("    readback mismatch: %zu elements differ between two reads\n", rb);
            }
        }
        if (it == 0) {
            ref = cur;
            printf("iter 0: %.1f ms (baseline)\n", ms);
            continue;
        }
        for (size_t i = 0; i < cur.size(); ++i) {
            if (cur[i] != ref[i]) {
                seen[i][cur[i]] += 1;
            }
        }
        size_t ndiff = 0;
        double maxd = 0, sq = 0;
        for (size_t i = 0; i < cur.size(); ++i) {
            const double d = std::fabs((double) cur[i] - (double) ref[i]);
            if (d > 0) ++ndiff;
            maxd = std::max(maxd, d);
            sq += d * d;
        }
        printf("iter %d: %.1f ms  ndiff=%zu/%zu  max=%.3e  rmse=%.3e\n", it, ms, ndiff,
               cur.size(), maxd, std::sqrt(sq / (double) cur.size()));
        if (ndiff > 0) {
            size_t shown = 0;
            for (size_t i = 0; i < cur.size() && shown < 12; ++i) {
                if (cur[i] != ref[i]) {
                    const size_t d = i % (size_t) D;
                    const size_t r = (i / (size_t) D) % (size_t) L;
                    const size_t h = i / ((size_t) D * (size_t) L);
                    printf("    idx=%zu (d=%zu,row=%zu,head=%zu) ref=%.6g cur=%.6g\n",
                           i, d, r, h, ref[i], cur[i]);
                    ++shown;
                }
            }
        }
    }
    if (!seen.empty()) {
        printf("\nvalue distribution at differing elements (ref = iter0, not counted):\n");
        for (auto & [idx, counts] : seen) {
            const size_t d = idx % (size_t) D;
            const size_t r = (idx / (size_t) D) % (size_t) L;
            const size_t h = idx / ((size_t) D * (size_t) L);
            printf("  idx=%zu (d=%zu,row=%zu,head=%zu) ref=%.9g", idx, d, r, h, ref[idx]);
            for (auto & [val, cnt] : counts) {
                printf(" | %.9g x%d", val, cnt);
            }
            printf("\n");
        }
    } else {
        printf("\nall iterations bit-identical\n");
    }
    if (argc > 6) {
        FILE * f = fopen(argv[6], "wb");
        if (f) { fwrite(ref.data(), sizeof(float), ref.size(), f); fclose(f); }
        printf("wrote %s\n", argv[6]);
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(gpu);
    return 0;
}
