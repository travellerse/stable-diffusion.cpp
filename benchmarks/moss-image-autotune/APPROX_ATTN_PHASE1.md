# MOSS-Image 近似 Attention 研究 · 第一阶段报告

日期：2026-10-30 · 设备：RX 6700 XT（gfx1031 / RDNA2 / RADV）· 配置：1440×1920、8 steps、CFG=1、
`--flash-attn diffusion=on --conv-direct vae=on` · 基线二进制 `/tmp/sdcli-final`（md5 349b89f7…）。

本阶段只做诊断与离线测量，**没有修改默认 Attention 算法、没有把任何近似路径合入默认执行**。
新增代码全部由环境变量门控，默认路径行为不变。

---

## 0. 结论摘要

| 方向 | 判定 | 依据（实测） |
|---|---|---|
| 2. Training-free Block Sparse | **关闭** | 注意力接近均匀（熵 8.92 / 最大 9.29）；90 % 质量需 75 % 的 KV 块；代理路由 recall@10 % 只有 0.19；保留 50 % 块时输出相对误差仍 71 % |
| 3. SpargeAttn / Sol-Attn / FlashPrefill 的适用性 | **不适用** | 三者均为 CUDA tensor-core 实现；SpargeAttn 依赖块稀疏（见上）；FlashPrefill V2 的 mean correction 在我们的数据上把误差从 1.19e-2 放大到 1.30e-1 |
| 1. INT8 QK + FP16 PV | **有条件** | 离线输出相对误差 1.19 %（per-row INT8）；但本卡 f16 GEMM 已达 16.6 TFLOP/s，与 int8 GEMM（16.6）相同，说明 QK 量化的收益只能来自 FA 内指令数（理论 FA ≈0.75×，e2e ≈ −9 %），需端到端质量验证 |
| 附带：A/A 不确定性根因 | **已定位** | 当前默认 `block_cols=64` 的 FA 内核在 RDNA2 上偶发单元素错误（≈1/10 次调用）；`block_cols=32`（上游默认）20 次调用 0 错误 |

**建议**：先修复 `block_cols=64` 的偶发错误（否则任何有损改动的质量评估都被它污染），再决定是否进入
INT8 QK 的 kernel 阶段。块稀疏方向不建议继续。

---

## 1. A/A 不确定性调查

### 1.1 方法

同一二进制、模型、Recipe、Seed(42)、Prompt；每次运行在采样循环里导出：
每步 latent（`lin_sNNN`）、每步模型输出（`pred_sNNN`）、DiT 输入（`dx` / `dctx`），以及选定层的
graph 中间张量（`ggml_set_output` 保活后导出）。对比指标：FNV-1a 哈希、MAE、RMSE、最大绝对误差、相对误差。

所有导出由 `SD_DUMP_STEPS` 环境变量门控；诊断代码在
`src/pipeline/diffusion_engine.cpp`（latent/pred/ctx）、`src/core/ggml_runner.cpp`（graph 节点导出）、
`src/model/diffusion/moss_image.hpp` / `src/model/common/rope.hpp`（层输入与 Q/K/V 导出）。

> **测量陷阱（已修正）**：只调用 `ggml_build_forward_expand` 而不同时 `ggml_set_output` 时，中间张量的
> 显存会被后续节点复用，导出到的其实是“别人”的数据。第一版导出因此得到过 l1…l41 完全相同、且两次运行
> 差异 10 % 的假结果。修正保活后，layer 0 的输入/归一化/Q/K/V/attention 输出全部逐位一致。

### 1.2 结果（1440×1920，8 步，3 次重复）

| 张量 | 结果 |
|---|---|
| `lin_s001`（初始噪声） | 3 次运行**逐位一致** → RNG（Philox，默认 `cuda`）与 seed 处理是确定性的 |
| `dctx` / `dx`（DiT 输入） | 逐位一致 → 文本编码器（Qwen3.5-9B，Vulkan 计算 + CPU 常驻权重）确定性 |
| `pred_s001`（第一次前向输出） | **不一致**：rel 2.4–4.0 %，max abs 1.1–1.5 |
| `lin_s002…s008` | 不一致，误差逐级放大（rel 1.1e-3 → 2.2e-1），是典型混沌放大 |

逐层二分（`SD_DUMP_X_LAYERS`，每次只导出一个层的 block 输入）：

| 层输入 | RMSE | 相对误差 |
|---|---|---|
| l7 | 0 | 0 |
| l14 | 9.93e-06 | 5.66e-08 |
| l21 | 2.71e-02 | 9.21e-03 |
| l28 | 1.03e-01 | 2.52e-02 |
| l35 | 5.24e-01 | 5.93e-02 |
| l41 | 2.94e+00 | 8.62e-02 |

**第一次分歧是 1-ulp 级（5.7e-08），出现在 block 7–13 之间**；之后 34 层把它放大到输出端 3.6 %。

### 1.3 排除项

| 假设 | 实验 | 结果 |
|---|---|---|
| CPU/GPU 归约顺序、采样器 | `--rng std_default` / 不同线程数 | 输入逐位一致，非根因 |
| 权重预取 | `--disable-prefetch` | 仍不一致（rel 2.4 %） |
| 惰性加载 | `--eager-load` | 仍不一致（rel 2.3 %） |
| 显存计划漂移 | 固定 `--max-vram 10` → 工作区大小两次完全相同（1668.85 MB） | **仍不一致**（rel 3.5 %）→ 不是计划问题 |
| **Flash Attention 路径** | `--flash-attn diffusion=off`（manual attention） | **逐位一致** ← 根因锁定在 FA |

### 1.4 根因：`block_cols=64` 的 FA 内核偶发单元素错误

隔离测试（`fa_determinism.cpp`）：用真实 Q/K/V（layer 0、L=10812、24 heads）重复调用同一个
`ggml_flash_attn_ext`，逐次比对输出。

| 配置 | 调用次数 | 出现差异的调用 | 元素 | 位置 |
|---|---|---|---|---|
| `block_cols=32`（上游 D=128 默认） | 20 | **0** | — | — |
| `block_cols=64`（我们上一轮采纳的 RDNA2 override） | 40 | 3–4 | 1–7 / 33.2 M | 全部在 **d=69** |

- 差异元素数量极少（1–7 个），但数值可达 1e-4–6.5e+04（后者是垃圾值级）。
- 位置固定在 d=69，与 tile 宽度 Bc 的取值直接相关。
- 每次模型前向有 42 次 FA 调用、每步 8 次、共 8 步 → 单次运行内有数百次机会触发，足以解释 A/A 图像差异。

**这不是浮点误差，而是当前默认路径里的偶发正确性缺陷**。它同时也解释了历史记录中
"A/A 图像 PSNR 只有 25–33 dB" 的现象。

### 1.5 结论与建议

1. 在修复前，**任何有损/近似改动的端到端质量评估都不具备可比性**（噪声底就是 3 % 级）。
2. 修复路径（需另行授权）：a) 回退 `block_cols=64`（代价：FA 慢 12.7 %、e2e 约 +7.6 %，但恢复确定性）；
   b) 在 shader 内定位 d=69 越界/竞态（更可能是某个 lane 的尾部处理），修好后保留 Bc=64 的性能收益。
3. 修复后应把 `fa_determinism` 纳入回归门槛（≥40 次调用 0 差异）。

---

## 2. 真实 Q/K/V 导出

### 2.1 导出点

`Rope::attention`（`src/model/common/rope.hpp`）中 `ggml_ext_attention_ext` 之前的 q/k/v —— 即
**Flash Attention 真正消费的张量**（q 已完成 q_norm + RoPE；k 已完成 k_norm + RoPE；v 未过 RoPE）。

### 2.2 形状与类型（layer 0 / 21，1440×1920，L=10842）

| 张量 | dump 形状 | FA 内部形状 | dtype | 说明 |
|---|---|---|---|---|
| q | `[128, 10842, 24, 1]` | `[128, L, 24]` | f32 | 不做 cast，FA 直接吃 f32 |
| k | `[128, 10842, 24, 1]` | `[128, L, 24]` | f32→**f16** | `ggml_cast(k, F16)` |
| v | `[128, 24, 10842, 1]` | `[128, L, 24]` | f32→**f16** | qkv 的 strided view，导出前 `ggml_cont` |

v 是 `proj.qkv` 的视图，直接 `ggml_backend_tensor_get` 只能拿到 3/4 的数据（129.6 MB / 133.2 MB），
必须 `ggml_cont` 后再导出——否则 v 的分析结果是错的（本报告已修正）。

### 2.3 数据分布（layer 0，head 0，前 1024 个 query）

| 指标 | 值 |
|---|---|
| logit std | 0.699（p1 −1.99，p99 1.89，max 7.66） |
| softmax 最大概率均值 | 0.0020（均匀分布为 9.25e-5） |
| softmax 熵（均值） | **8.917**（ln L = 9.288）→ 最大熵的 **96 %** |
| q / k 每行 absmax 均值 | 3.18 / 3.12 |

→ **注意力权重接近均匀**，这是后面块稀疏失败的直接原因。

---

## 3. INT8 QK 离线误差

计算链严格按部署路径：`QK^T (f32 累加) → f32 softmax → PV（V 为 f16，f32 累加）`；
量化仅作用在 Q/K 上，PV 保持 f16/f32（与 SageAttention 的 INT8 QK + FP16 PV 一致）。

数据：Lk=10812（1440×1920 真实模型数据），layers 0/21 × step 1/8，heads 0/12，每次 2048 个 query。

**输出相对误差（RMSE / ‖参考输出‖）**

| 层 / 步 | per-row INT8 | +全局 K 均值 | 每 64 keys | 每 128 keys | per-row + 块内均值 |
|---|---|---|---|---|---|
| L0 / step1 | 1.21 % | 1.15 % | 1.40 % | 1.46 % | 12.6 % |
| L0 / step8 | 0.80 % | 0.80 % | 0.90 % | 0.94 % | 12.1 % |
| L21 / step1 | 2.02 % | 1.94 % | 2.41 % | 2.41 % | 24.5 % |
| L21 / step8 | 1.96 % | 1.93 % | 2.41 % | 2.42 % | 20.7 % |

QK logit 误差：L0 7.9e-3–9.3e-3，L21 2.6e-2–2.7e-2（logit std 0.70，layer 0）。

观察：

1. **per-row（≈ per-thread）粒度最好**：输出相对误差 1.19 %。
2. 全局 K 均值平滑在 MOSS 上**没有收益**（略差）：模型在 RoPE 前已经做了 `k_norm`（每 head-dim
   RMSNorm），K 的离群值已被压平——这与 SageAttention 面向 LLM（无 k_norm）的动机不同。
3. 块内均值平滑显著变差（13 %）：减去块均值后残差的动态范围并未改善，反而放大了量化噪声。
4. 结论：**INT8 QK 在这个模型上是“可用量级”的误差（1 % 输出），但没有任何平滑技巧能把它降到 <0.5 %**。

---

## 4. Block Sparse 统计

同一批数据；流式分块计算，从不物化完整注意力矩阵。

### 4.1 质量覆盖（Bc=128，heads 0/12，每次 512 个 query）

| 层 / 步 | 50 % 质量 | 75 % | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|
| L0 / step1 | 35.7 % 块 | 57.0 % | 73.2 % | 80.9 % | 91.7 % |
| L0 / step8 | 43.5 % | 70.1 % | 87.8 % | 93.9 % | 98.8 % |
| L21 / step1 | 21.0 % | 42.6 % | 64.2 % | 75.2 % | 89.6 % |
| L21 / step8 | **11.2 %** | 24.0 % | **39.1 %** | 48.6 % | 65.1 % |

Bc=64 与 Bc=128 的覆盖率差异 <2 个百分点（见上表与首轮测量），块大小不是决定因素。

### 4.2 代理路由与稀疏输出误差（Bc=128，keep=50 %）

| 层 / 步 | 代理 top-k 命中率@10 % | 丢弃相对误差@50 % | 均值修正@50 % |
|---|---|---|---|
| L0 / step1 | 0.245 | 0.396 | 0.295 |
| L0 / step8 | 0.231 | 0.667 | 0.510 |
| L21 / step1 | 0.175 | 0.631 | 0.459 |
| L21 / step8 | 0.322 | 0.265 | 0.198 |

完整 keep∈{10,20,30,50 %} 的曲线见分析脚本输出；keep=10 % 时丢弃误差 1.2–2.3、均值修正 0.7–1.4。

### 4.3 结论

- 密度随层/步变化很大：最稠密的是 L0 末步（90 % 质量需 88 % 的块），最稀疏的是 L21 末步
  （90 % 质量需 39 % 的块）。
- 换 seed（7）与换 prompt（"neon-lit alley"）后结论不变：L0 末步 90 %→87 % 块，L21 末步 90 %→42 % 块，
  keep=50 % 的均值修正误差 47 % / 26 %。稀疏度是层/步的函数，不是 prompt 的函数。
- 但即使在最稀疏的 L21/step8，保留 50 % 的块仍带来 **20 % 的输出相对误差**（均值修正后），
  保留 10 % 时更差；代理路由命中率最高只有 0.32。
- 因此训练无关块稀疏在本模型上没有可用工作点：**方向关闭**。

---

## 5. 文献与 RDNA2 适用性

依据论文摘要（arXiv 官方 API）与官方代码 README 核实，不是搜索摘要。

| 方法 | arXiv | Training-free | 需要微调 | 依赖的硬件特性 | 图像模型验证 | RDNA2 |
|---|---|---|---|---|---|---|
| SageAttention (ICLR'25) | 2410.02367 | 是 | 否 | INT8 tensor core | 是（含图像/视频） | 无 |
| SageAttention2 (ICML'25) | 2411.10958 | 是 | 否 | INT4/FP8 tensor core | 是 | 无 |
| SageAttention3 (NeurIPS'25) | 2505.11594 | 是 | 否 | **FP4 tensor core（Blackwell）** | 是 | 无 |
| SpargeAttn (ICML'25) | 2502.18137 | 是 | 否 | INT8 tensor core + 块稀疏 | 是 | 无 |
| SpargeAttention2 | 2602.13515 | 否 | **是（蒸馏微调）** | — | 视频 | 无 |
| Sol-Attn | 2607.24027 | 是 | 否 | CUDA kernel | 视频 | 无 |
| FlashPrefill V2 | 2608.19758 | 是 | 否 | FP8 + FA3/4 级 kernel | **否（LLM prefill）** | 无 |

共同点：全部是 NVIDIA CUDA/tensor-core 实现；没有任何一篇报告 AMD/RDNA2 数据。
SageAttention 的加速来源是 tensor core 的低精度吞吐；RDNA2 没有 tensor core，只能用 `dp4a`
（见 §6），因此**不能把论文里的 2–5× 直接搬到本机**。

---

## 6. 性能分量

| 测量 | 结果 |
|---|---|
| f16 `mul_mat`（K=3072, M=3072, N=10812） | 12.30 ms → **16.6 TFLOP/s** |
| int8 tensorwise GEMM（MOSS 形状，此前实测） | 16.1–16.7 TFLOP/s |
| FA 内核（L=10842, H=24, D=128） | 192–197 ms → **7.4–7.5 TFLOP/s** |
| 单次逐元素 pass（3072×10812 f32） | ≈0.8 ms（≈250 MB 流量 @325 GB/s） → 每层每步 Q/K 量化前处理约 1.6 ms，42 层 8 步合计 <0.1 % |

**关键事实**：在这张卡上 f16 与 int8 的 GEMM 吞吐已经相同（16.6 TFLOP/s），说明两者都不是
“吞吐受限”，而是各自被 LDS/发射/带宽限制。因此 INT8 QK 的收益只能来自 **FA 内核内 QK 段指令数减半**
（每 16 个元素：4×dot2×2 = 8 条 → 4 条 dp4a）。

若 QK 段指令减半、PV 不变：FA ≈ 0.75×（194 ms → ~146 ms），端到端 ≈ **−9 %**（FA 占 45 %）。
这是上限，未计算量化准备、scale 恢复与寄存器压力变化。

---

## 7. 是否进入 kernel 阶段

| 方向 | 建议 |
|---|---|
| Block sparse（training-free） | **不进入**。数据不支持，任何实现都无法达到可用误差。 |
| Sol-Attn / SpargeAttn / FlashPrefill | **不进入**。同上，且实现均为 CUDA。 |
| INT8 QK + FP16 PV | **有条件进入**。离线误差 1.19 % 可接受量级；理论 e2e 上限 −9 %；但必须先解决 §1 的确定性缺陷，否则无法做质量 A/B。 |

建议顺序：
1. 修复/回退 `block_cols=64` 的偶发错误，恢复确定性（并把 `fa_determinism` 作为门槛）。
2. 若仍要做 INT8 QK：先做 **1 层 1 步** 的 Vulkan 原型（Q/K 预量化 + dp4a），在真实图像上做 A/B；
   若 op-level 误差保持在 1 % 且 e2e 收益 ≥5 %，再考虑推广。

---

## 附：本阶段新增的诊断工具（全部默认关闭）

| 文件 | 作用 |
|---|---|
| `benchmarks/moss-image-autotune/det_compare.py` | 逐步对比两次运行的 latent/pred（哈希、RMSE、max、相对误差） |
| `benchmarks/moss-image-autotune/fa_determinism.cpp` | 固定输入重复调用 FA，检测偶发错误 |
| `benchmarks/moss-image-autotune/qkv_int8_study.py` | 真实 Q/K/V 上的 INT8 量化误差（多种粒度） |
| `benchmarks/moss-image-autotune/block_sparsity.py` | 流式块稀疏统计 + 代理路由 + 均值修正评估 |
| `benchmarks/moss-image-autotune/gemm_f16_vs_i8.cpp` | f16 / int8 GEMM 吞吐对比 |
| `SD_DUMP_STEPS` / `SD_DUMP_X_LAYERS` / `SD_DUMP_ATTN_LAYERS` / `SD_DUMP_NRM` | 诊断导出门控 |
