// Microbenchmark for GGML_OP_MUL_MAT_I8_TENSORWISE on the Vulkan backend.
//
// Runs the same shapes MOSS-Image issues at 1024^2 (16384 image tokens) and
// reports latency and effective TOPS for: the activation quantizer alone, the
// INT8 GEMM alone, and both together (what the model actually executes).
//
// Build (from the repo root, after building build-vulkan):
//   g++ -O2 -std=c++17 -Iggml/include -Iggml/src ggml_i8bench.cpp \
//       build-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a \
//       build-vulkan/ggml/src/libggml.a build-vulkan/ggml/src/libggml-cpu.a \
//       build-vulkan/ggml/src/libggml-base.a -lvulkan -lpthread -ldl -o /tmp/i8bench
//
// usage: i8bench [--iters N] [--shape qkv|upgate|down|all] [--check] [--dump FILE]
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>

struct Shape {
    const char * name;
    int64_t k;     // in features (ne0 of the weight)
    int64_t n;     // out features (ne1 of the weight)
    int64_t rows;  // activation rows (image tokens)
};

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static ggml_context * g_ctx = nullptr;
static ggml_backend_t g_backend = nullptr;

static void fill(ggml_tensor * t, int seed) {
    std::vector<uint8_t> data(ggml_nbytes(t));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (uint8_t) (((i * 17u + seed * 29u) % 61u) - 30);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size());
}

static void fill_f32(ggml_tensor * t, int seed) {
    std::vector<float> data((size_t) ggml_nelements(t));
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = (float) (((int) ((i * 13 + seed * 7) % 43)) - 21) / 9.0f;
    }
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

// Reference: exact int32 accumulation over K, then tensorwise scaling, matching
// the shader's math (so a correct kernel must agree to the last bit).
static void cpu_reference(const Shape & s, const std::vector<float> & x, const std::vector<int8_t> & w,
                          const std::vector<float> & ws, std::vector<float> & out) {
    out.assign((size_t) (s.n * s.rows), 0.0f);
    for (int64_t r = 0; r < s.rows; ++r) {
        // quantize like ggml_quantize_i8_convrot: absmax per row
        float amax = 0.0f;
        for (int64_t i = 0; i < s.k; ++i) {
            amax = std::max(amax, std::fabs(x[r * s.k + i]));
        }
        const float ascale = amax / 127.0f;
        std::vector<int32_t> q((size_t) s.k);
        for (int64_t i = 0; i < s.k; ++i) {
            q[i] = (int32_t) std::lrint(x[r * s.k + i] / ascale);
            q[i] = std::max(-127, std::min(127, q[i]));
        }
        for (int64_t c = 0; c < s.n; ++c) {
            int32_t sum = 0;
            for (int64_t i = 0; i < s.k; ++i) {
                sum += (int32_t) w[c * s.k + i] * q[i];
            }
            out[(size_t) (r * s.n + c)] = (float) sum * ascale * ws[c];
        }
    }
}

int main(int argc, char ** argv) {
    int iters = 5;
    std::string which = "all";
    std::string dump;
    int64_t rows_override = 0;
    bool check = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--shape") && i + 1 < argc) {
            which = argv[++i];
        } else if (!strcmp(argv[i], "--check")) {
            check = true;
        } else if (!strcmp(argv[i], "--dump") && i + 1 < argc) {
            dump = argv[++i];
        } else if (!strcmp(argv[i], "--rows") && i + 1 < argc) {
            rows_override = atoll(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--iters N] [--shape qkv|upgate|down|all] [--check]\n", argv[0]);
            return 1;
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
    g_backend = ggml_backend_dev_init(dev, nullptr);

    std::vector<Shape> shapes = {
        { "qkv",    3072,  9216, 16384 },
        { "upgate", 3072, 16384, 16384 },
        { "down",   8192,  3072, 16384 },
    };

    printf("%-8s %6s %6s %7s %9s %9s %9s %9s %8s\n",
           "shape", "k", "n", "rows", "quant_ms", "gemm_ms", "both_ms", "gemm_Tops", "GFLOPs");
    for (Shape s : shapes) {
        if (which != "all" && which != s.name) {
            continue;
        }
        if (rows_override > 0) {
            s.rows = rows_override;
        }
        ggml_init_params p = { 16 * 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(p);
        ggml_tensor * w  = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, s.k, s.n);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.rows);
        ggml_tensor * ws = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.n);
        ggml_tensor * q  = ggml_quantize_i8_convrot(ctx, x, 256);
        ggml_tensor * y  = ggml_mul_mat_i8_tensorwise(ctx, w, q, ws, nullptr, 256);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (!buf) {
            fprintf(stderr, "%s: alloc failed (%.2f GB needed)\n", s.name,
                    (double) (ggml_nbytes(w) + ggml_nbytes(x) + ggml_nbytes(q) + ggml_nbytes(y)) / 1e9);
            ggml_free(ctx);
            continue;
        }
        fill(w, 3);
        fill_f32(x, 5);
        fill_f32(ws, 7);
        if (!ggml_backend_supports_op(g_backend, q) || !ggml_backend_supports_op(g_backend, y)) {
            fprintf(stderr, "%s: op unsupported by backend\n", s.name);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }

        ggml_cgraph * gq = ggml_new_graph(ctx);
        ggml_build_forward_expand(gq, q);
        ggml_cgraph * gy = ggml_new_graph(ctx);
        ggml_build_forward_expand(gy, y);
        ggml_cgraph * gb = ggml_new_graph(ctx);
        ggml_build_forward_expand(gb, q);
        ggml_build_forward_expand(gb, y);

        if (ggml_backend_graph_compute(g_backend, gb) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: compute failed\n", s.name);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            continue;
        }
        ggml_backend_synchronize(g_backend);

        auto time_graph = [&](ggml_cgraph * g) {
            ggml_backend_graph_compute(g_backend, g);  // warm up
            ggml_backend_synchronize(g_backend);
            double t0 = now_ms();
            for (int it = 0; it < iters; ++it) {
                ggml_backend_graph_compute(g_backend, g);
            }
            ggml_backend_synchronize(g_backend);
            return (now_ms() - t0) / iters;
        };

        double t_quant = time_graph(gq);
        double t_gemm  = time_graph(gy);
        double t_both  = time_graph(gb);
        double flops   = 2.0 * (double) s.k * (double) s.n * (double) s.rows;
        printf("%-8s %6lld %6lld %7lld %9.2f %9.2f %9.2f %9.2f %8.1f\n", s.name,
               (long long) s.k, (long long) s.n, (long long) s.rows,
               t_quant, t_gemm, t_both, flops / (t_gemm * 1e6), flops / 1e9);

        if (!dump.empty()) {
            const size_t nb = (size_t) ggml_nbytes(y);
            std::vector<uint8_t> blob(nb);
            ggml_backend_tensor_get(y, blob.data(), 0, nb);
            std::string path = dump + "." + s.name + ".bin";
            FILE * f = fopen(path.c_str(), "wb");
            fwrite(blob.data(), 1, nb, f);
            fclose(f);
            uint64_t h = 1469598103934665603ull;
            for (uint8_t b : blob) { h ^= b; h *= 1099511628211ull; }
            printf("         dump %s (%zu bytes) fnv1a=%016llx\n", path.c_str(), nb,
                   (unsigned long long) h);
        }
        if (check) {
            // small exactness check on a reduced K to keep the CPU reference fast
            std::vector<float> xd((size_t) (s.k * 8)), outd((size_t) (s.n * 8));
            std::vector<int8_t> wd((size_t) (s.k * s.n));
            std::vector<float> wsd((size_t) s.n);
            ggml_backend_tensor_get(x, xd.data(), 0, xd.size() * sizeof(float));
            ggml_backend_tensor_get(w, wd.data(), 0, wd.size());
            ggml_backend_tensor_get(ws, wsd.data(), 0, wsd.size());
            Shape small = { s.name, s.k, s.n, 8 };
            std::vector<float> ref;
            cpu_reference(small, xd, wd, wsd, ref);
            std::vector<float> got(ref.size());
            ggml_backend_tensor_get(y, got.data(), 0, got.size() * sizeof(float));
            double amax = 0.0, rel = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                amax = std::max(amax, (double) std::fabs(got[i] - ref[i]));
                rel  = std::max(rel, (double) (std::fabs(got[i] - ref[i]) / std::max(1.0f, std::fabs(ref[i]))));
            }
            printf("         check: max_abs=%.6g max_rel=%.3g\n", amax, rel);
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    ggml_backend_free(g_backend);
    return 0;
}
