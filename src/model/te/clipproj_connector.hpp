#ifndef __SD_MODEL_TE_CLIPPROJ_CONNECTOR_HPP__
#define __SD_MODEL_TE_CLIPPROJ_CONNECTOR_HPP__

#include <string>

#include "core/ggml_runner.h"
#include "model/common/ggml_block.hpp"

// Maps a small Qwen3-VL hidden state into the MiniMax-H3 conditioning space:
//   cond = ((h - mean_in) / std_in) * W * std_out + mean_out
// The published "-mlp" matrices carry no W; the residual network replaces it and
// is applied inside the same standardised space it was trained in.
struct ClipProjConnector : public UnaryBlock {
    int64_t input_dim;
    int64_t output_dim;
    int64_t hidden_dim;

    ClipProjConnector(int64_t input_dim, int64_t output_dim, int64_t hidden_dim)
        : input_dim(input_dim), output_dim(output_dim), hidden_dim(hidden_dim) {
        blocks["mlp.0"] = std::make_shared<Linear>(input_dim, hidden_dim, true, false, true);
        blocks["mlp.2"] = std::make_shared<Linear>(hidden_dim, output_dim, true, false, true);
    }

    void init_params(ggml_context* ctx,
                     const String2TensorStorage& tensor_storage_map = {},
                     const std::string prefix                       = "") override {
        SD_UNUSED(tensor_storage_map);
        SD_UNUSED(prefix);
        params["mean_in"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
        params["std_in"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input_dim);
        params["mean_out"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim);
        params["std_out"]  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_dim);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) override {
        auto c     = ctx->ggml_ctx;
        auto first = std::dynamic_pointer_cast<Linear>(blocks["mlp.0"]);
        auto last  = std::dynamic_pointer_cast<Linear>(blocks["mlp.2"]);
        x          = ggml_div(c, ggml_sub(c, x, params["mean_in"]), params["std_in"]);
        x          = ggml_gelu(c, first->forward(ctx, x));
        x          = last->forward(ctx, x);
        return ggml_add(c, ggml_mul(c, x, params["std_out"]), params["mean_out"]);
    }
};

struct ClipProjConnectorRunner : public GGMLRunner {
    ClipProjConnector model;
    static constexpr const char* prefix = "embeddings_connectors";

    static int64_t dim_of(const String2TensorStorage& tensor_storage_map,
                          const std::string& name,
                          int axis = 0,
                          int64_t fallback = 0) {
        auto entry = tensor_storage_map.find(std::string(prefix) + "." + name);
        if (entry == tensor_storage_map.end()) {
            return fallback;
        }
        return entry->second.ne[axis];
    }

    ClipProjConnectorRunner(ggml_backend_t backend,
                            const String2TensorStorage& tensor_storage_map,
                            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          model(dim_of(tensor_storage_map, "mean_in"),
                dim_of(tensor_storage_map, "mean_out"),
                dim_of(tensor_storage_map, "mlp.0.weight", 1)) {
        model.init(params_ctx, tensor_storage_map, prefix);
    }

    std::string get_desc() override {
        return "clipproj";
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) {
        model.get_param_tensors(tensors, prefix);
    }

    sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& hidden_states) {
        auto get_graph = [&]() {
            auto gf         = ggml_new_graph(compute_ctx);
            auto x          = make_input(hidden_states);
            auto runner_ctx = get_context();
            ggml_build_forward_expand(gf, model.forward(&runner_ctx, x));
            return gf;
        };
        return take_or_empty(GGMLRunner::compute(get_graph, n_threads, false));
    }
};

#endif  // __SD_MODEL_TE_CLIPPROJ_CONNECTOR_HPP__
