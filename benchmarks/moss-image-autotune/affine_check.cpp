// Checks that the fused modulation chain
//   y = add(mul(rms_norm(x, eps), w * (1 + s)), shift)
// matches the unfused formulation
//   y = add(add(mul(mul(rms_norm(x, eps), w), s), mul(rms_norm(x, eps), w)), shift)
// and reports the per-call time of each.
//
// build:
//   g++ -O2 -std=c++17 -fopenmp -Iggml/include -Iggml/src affine_check.cpp \
//     -Wl,--start-group build-vulkan/ggml/src/libggml-base.a \
//     build-vulkan/ggml/src/ggml-vulkan/libggml-vulkan.a \
//     build-vulkan/ggml/src/libggml-cpu.a build-vulkan/ggml/src/libggml.a \
//     -Wl,--end-group -lvulkan -lpthread -ldl -o /tmp/affine_check
// usage: affine_check [dim] [rows] [iters]
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

static double now_ms() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// mod is shaped (6*dim, 1, 1) like MOSS's shared_block_modulation output;
// s and shift are dim-0 slices of it, i.e. per-channel vectors.
static ggml_tensor * build_fused(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * x, ggml_tensor * w,
                                 ggml_tensor * s, ggml_tensor * shift, float eps) {
    ggml_tensor * w_eff = ggml_add(ctx, w, ggml_mul(ctx, w, s));
    ggml_build_forward_expand(gf, w_eff);
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w_eff);
    return ggml_add(ctx, y, shift);
}

static ggml_tensor * build_ref(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                               ggml_tensor * s, ggml_tensor * shift, float eps) {
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, w);
    ggml_tensor * m = ggml_mul(ctx, y, s);
    y = ggml_add(ctx, y, m);
    return ggml_add(ctx, y, shift);
}

struct Side {
    ggml_context * ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor *x, *w, *s, *shift, *out;
    ggml_cgraph * graph;
};

static Side make_side(ggml_backend_t backend, int64_t dim, int64_t rows,
                      const std::vector<float> & xd, const std::vector<float> & wd,
                      const std::vector<float> & md, bool fused) {
    Side sd{};
    ggml_init_params ip = { (size_t) 64 * 1024 * 1024 + (size_t) dim * rows * 16, nullptr, true };
    sd.ctx       = ggml_init(ip);
    sd.x         = ggml_new_tensor_2d(sd.ctx, GGML_TYPE_F32, dim, rows);
    sd.w         = ggml_new_tensor_1d(sd.ctx, GGML_TYPE_F32, dim);
    ggml_tensor * mod = ggml_new_tensor_3d(sd.ctx, GGML_TYPE_F32, 6 * dim, 1, 1);
    sd.s         = ggml_view_4d(sd.ctx, mod, dim, 1, 1, 1, mod->nb[1], mod->nb[2], mod->nb[3], 1 * dim * mod->nb[0]);
    sd.shift     = ggml_view_4d(sd.ctx, mod, dim, 1, 1, 1, mod->nb[1], mod->nb[2], mod->nb[3], 0 * dim * mod->nb[0]);
    sd.graph     = ggml_new_graph(sd.ctx);
    if (fused) {
        // The model pre-expands the modulation slices, so they never land
        // between the rms_norm/mul/add triple.
        ggml_build_forward_expand(sd.graph, sd.s);
        ggml_build_forward_expand(sd.graph, sd.shift);
    }
    sd.out       = fused ? build_fused(sd.ctx, sd.graph, sd.x, sd.w, sd.s, sd.shift, 1e-6f)
                         : build_ref(sd.ctx, sd.x, sd.w, sd.s, sd.shift, 1e-6f);
    sd.buf       = ggml_backend_alloc_ctx_tensors(sd.ctx, backend);
    if (!sd.buf) { fprintf(stderr, "alloc failed\n"); exit(1); }
    ggml_backend_tensor_set(sd.x, xd.data(), 0, xd.size() * sizeof(float));
    ggml_backend_tensor_set(sd.w, wd.data(), 0, wd.size() * sizeof(float));
    ggml_backend_tensor_set(mod, md.data(), 0, md.size() * sizeof(float));
    ggml_build_forward_expand(sd.graph, sd.out);
    return sd;
}

static double bench(ggml_backend_t backend, ggml_cgraph * graph, int iters) {
    ggml_backend_graph_compute(backend, graph);
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
        double t0 = now_ms();
        ggml_backend_graph_compute(backend, graph);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char ** argv) {
    const int64_t dim  = argc > 1 ? atoll(argv[1]) : 3072;
    const int64_t rows = argc > 2 ? atoll(argv[2]) : 10842;
    const int iters    = argc > 3 ? atoi(argv[3]) : 30;

    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) { fprintf(stderr, "backend init failed\n"); return 1; }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    std::vector<float> xd(dim * rows), wd(dim), md(6 * dim);
    for (auto & v : xd) v = nd(rng);
    for (auto & v : wd) v = 1.f + 0.1f * ud(rng);
    for (auto & v : md) v = 0.1f * ud(rng);

    Side g = make_side(gpu, dim, rows, xd, wd, md, true);
    Side r = make_side(cpu, dim, rows, xd, wd, md, false);

    ggml_backend_graph_compute(gpu, g.graph);
    ggml_backend_graph_compute(cpu, r.graph);

    std::vector<float> gv(dim * rows), rv(dim * rows);
    ggml_backend_tensor_get(g.out, gv.data(), 0, gv.size() * sizeof(float));
    ggml_backend_tensor_get(r.out, rv.data(), 0, rv.size() * sizeof(float));

    double max_abs = 0, sum_sq = 0, max_rel = 0;
    for (size_t i = 0; i < gv.size(); ++i) {
        double d = std::fabs((double) gv[i] - (double) rv[i]);
        max_abs = std::max(max_abs, d);
        sum_sq += d * d;
        if (std::fabs(rv[i]) > 1e-3) max_rel = std::max(max_rel, d / std::fabs((double) rv[i]));
    }
    const double rms = std::sqrt(sum_sq / (double) gv.size());
    printf("n=(%" PRId64 "x%" PRId64 ") max_abs=%.3e max_rel=%.3e rms=%.3e\n", dim, rows, max_abs, max_rel, rms);

    // repeat the fused chain and check run-to-run determinism
    {
        std::vector<float> first(dim * rows), cur(dim * rows);
        ggml_backend_graph_compute(gpu, g.graph);
        ggml_backend_synchronize(gpu);
        ggml_backend_tensor_get(g.out, first.data(), 0, first.size() * sizeof(float));
        int bad = 0;
        float worst = 0.f;
        for (int i = 0; i < iters; ++i) {
            ggml_backend_graph_compute(gpu, g.graph);
            ggml_backend_synchronize(gpu);
            ggml_backend_tensor_get(g.out, cur.data(), 0, cur.size() * sizeof(float));
            size_t nd = 0;
            for (size_t j = 0; j < cur.size(); ++j) {
                if (cur[j] != first[j]) { ++nd; worst = std::max(worst, std::fabs(cur[j] - first[j])); }
            }
            if (nd) { ++bad; printf("  repeat %d: %zu differing elements (max %.3e)\n", i, nd, worst); }
        }
        printf("determinism: %d/%d repeats differed\n", bad, iters);
    }

    double t_fused = bench(gpu, g.graph, iters);
    double t_ref   = bench(cpu, r.graph, iters);
    printf("fused(vulkan) %.3f ms   ref(cpu) %.3f ms\n", t_fused, t_ref);
    printf("gpu graph nodes=%d  cpu graph nodes=%d\n", ggml_graph_n_nodes(g.graph), ggml_graph_n_nodes(r.graph));
    for (int i = 0; i < ggml_graph_n_nodes(g.graph); ++i) {
        ggml_tensor * t = ggml_graph_node(g.graph, i);
        printf("  node %2d %-10s ne=(%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ") srcs=%s,%s\n", i,
               ggml_op_name(t->op), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
               t->src[0] ? ggml_op_name(t->src[0]->op) : "-", t->src[1] ? ggml_op_name(t->src[1]->op) : "-");
    }

    ggml_backend_buffer_free(g.buf);
    ggml_backend_buffer_free(r.buf);
    ggml_free(g.ctx);
    ggml_free(r.ctx);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return max_abs < 1e-3 ? 0 : 2;
}
