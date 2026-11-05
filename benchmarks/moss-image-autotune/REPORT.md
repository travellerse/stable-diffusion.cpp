# MOSS-Image 端到端性能与显存调优报告

日期：2026-10-29（实验窗口 03:53–05:26 local）
仓库：`/home/trave/code/stable-diffusion.cpp`，分支 `dev`，HEAD `c9ee80a`（实验期间工作区无源码改动）
构建：`build-vulkan`（`SD_VULKAN=ON`，Release `-O3 -DNDEBUG`，ccache）
固定配方：`/home/trave/models/recipes/moss-refined-realism.json`
sha256 `50eae7d26ce13939a434de396d2f91d43cdc624f146664ab496cf9249eae8bdf`（**未被修改**，实验前后一致）

---

## 1. 结论摘要

0. **第二轮（用户授权修改 `ggml/` 后）主要收益来自 INT8 GEMM 内核重写**：把 `ggml-vulkan`
   的 `mul_mat_i8_tensorwise` 从 16×16 tile / 单累加器 / 每 32 int8 两次 barrier，改为
   64×32 tile / 每线程 8 累加器 / 每 64 int8 两次 barrier。微基准确认 **3.0–3.2×**
   （3.97 → 12.5 TFLOP/s），端到端全矩阵 **−49 %～−58 %**（1024²: 119.9 → 55.1 s；
   1440×1920: 385.4 → 192.8 s），且输出与旧内核**逐位相同**（三者 FNV-1a 完全一致），
   仓库自带 `test-int8-convrot --vulkan` 通过。详见 §13。

1. **第一轮（仅运行参数）被接受的优化**：对 diffusion 模块显式启用 flash attention。
   端到端 **−3.3 %**（1024²，124.0 → 119.9 s，n=5 vs n=5），DiT 采样段 **−3.6 %**；
   在 1.49 MP 上 **−3.6 %**（184.1 → 177.5 s）；
   在 2 MP 及以上，它不只是加速，而是让任务**可运行**（auto 路径在 1408×1408 上 8 步采样 8 分钟仍未结束，被安全停止）。
   代价：无（VRAM 峰值在噪声范围内相同）。这是纯运行参数，不修改任何代码或权重。

2. **瓶颈已定量定位**：DiT 采样占端到端 96 %，而采样几乎全部是 INT8 矩阵乘。
   把 Vulkan 整数点积关掉后，单步从 3.15 s 变为 90.5 s（512²，**29×**）。
   该内核位于 `ggml/`。第一轮微基准（§13.1）测得旧内核仅 **3.97 TFLOP/s ≈ dp4a 峰值的 4 %**，
   因此**任何 src/ 层的调度、卷积、注意力开关都不可能有量级收益**；第二轮直接重写该内核后
   达到 12.5 TFLOP/s（≈24 %），端到端减半。

3. **被实验否定的方向**（均有数据）：EasyCache（8 步内 0 次跳过）、conv-direct 关闭（慢 2.2 %）、
   强制单体图执行（慢 1.8 %）、verbose 日志（无可测开销）、以及在 1024² 让 auto 探针自己选（选错，见 §7）。

4. **显存上限是新发现**：auto 配置在 1.49 MP 时峰值已达 10.08 GB（余量 2.79 GB）；
   2 MP 用 auto 跑不完；开 flash attention 后 2 MP 峰值 10.43–10.47 GB、余量 2.40 GB；
   配方默认 1440×1920（2.76 MP）峰值 8.02 GB、余量 4.85 GB（可安全运行）。
   4 MP 未尝试，判为 `SKIPPED_UNSAFE`。

---

## 2. 环境（运行时实测）

| 项目 | 值 |
|---|---|
| CPU | 12th Gen Intel i5-12400F（6C/12T） |
| 内存 | 31 GiB；实验期间 MemAvailable 最低 19.6 GiB；swap 已用 6 GiB（实验前既有） |
| OS / kernel | CachyOS，7.3.0-4-cachyos |
| GPU | card1 = AMD Radeon RX 6700 XT（RADV NAVI22，PCI 1002:73df），VRAM 12 868 124 672 B |
| 驱动 | RADV（Vulkan），ROCm 7.2.4 工具链（`amd-smi` 26.2.2） |
| 设备能力 | `int dot: 1`，`fp16: dot2`，`bf16: 0`，`matrix cores: none`，shared mem 64 KiB |
| 桌面占用 | 空闲时 VRAM 811 MB，GPU busy 6 % |
| 推理时 | GPU busy 均值 93–95 %，p90 99 %，sclk 2462 MHz，mclk 1000 MHz，功耗 153 W |
| 编译 | `/usr/bin/c++`，`-O3 -DNDEBUG`，`GGML_NATIVE=ON`，ccache 生效 |

模型分配（auto-fit，1024²）：DiT 5162 MiB → VRAM；Qwen3.5-9B 5355 MiB → CPU 参数；
VAE 237 MiB → CPU 参数。总计 9846 MB。

---

## 3. 配方 → 命令的映射（已在代码中核对）

配方由 `sd-server` 读取（`examples/server/model_catalog.cpp`，`apply_recipe_defaults`），
`models` 槽位与 `defaults` 逐项映射到 `SDGenerationParams`。等价的 `sd-cli` 调用为：

```bash
./build-vulkan/bin/sd-cli \
  --diffusion-model /home/trave/models/diffusion_models/mossRefinedRealism_v208steps_int8.safetensors \
  --llm /home/trave/models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --tokenizer /home/trave/models/tokenizers/qwen35-tokenizer.json \
  --vae /home/trave/models/vae/MOSS-Image-SFT-vae.safetensors \
  -W <宽> -H <高> --steps 8 --cfg-scale 1.0 --sampling-method euler --flow-shift 5.0 \
  -s 42 -p "<固定提示词>" -o <输出>
```

- **配方本身不含提示词与随机种子**；实验统一固定提示词与 `-s 42` 作为实验控制项（非配方内容，已在 §9 列出）。
- 选择 `sd-cli` 作为测量入口的原因：它写完文件后进程退出，能给出真正的端到端墙钟；
  `sd-server` 不落盘图片，无法满足"输出文件写入并正常退出"的口径。
- 未使用 `--offload-to-cpu`：auto-fit 已经给出等价布局（TE/VAE 参数在 RAM，DiT 在 VRAM）。

---

## 4. 测量方法

`benchmarks/moss-image-autotune/bench.py` 包装一次 `sd-cli` 运行：

- **端到端**：`time.monotonic()` 从 `fork` 到进程退出；校验输出 PNG 存在且非空。
- **阶段耗时**：解析 `sd-cli` 自身日志（`get_learned_condition` / `sampling` / `decode_first_stage` / `generate_image`），
  无需改造推理代码。
- **GPU 显存**：独立线程每 **20 ms** 读 `/sys/class/drm/card1/device/mem_info_vram_used`；
  口径为**整卡占用（含桌面合成器约 0.81 GB）**，同时记录相对运行前基线的增量与最小余量。
- **安全闸**：显存余量低于 1.5 GiB 时自动终止本次子进程并标记 `safety_abort`（本轮未触发）。
- 同时采集 `gpu_busy_percent`、sclk/mclk、power1_average、`MemAvailable`、子进程 `ru_maxrss`。

原始记录：`raw-records.jsonl`（每次运行一条，含完整命令行）／汇总：`summary.csv`。

**测量局限（必须与数值一起看）**

1. 20 ms 采样的 sysfs 计数器仍是**下界**：注意力/激活的瞬态分配可能短于采样周期，未被捕获。
   直接证据：同一 1024² 配置在不同批次出现 6.84–8.60 GB 的峰值差异，
   而非单调随分辨率变化（1888×1056 只有 7.53 GB，1408×1408 是 10.43 GB），说明 peak 受 auto-fit
   分段计划与采样时刻影响。故显存结论只用于**同尺寸 A/B 比较**与**安全余量判断**，不作绝对上界。
2. **同配置重跑不是逐位可复现**：同种子两次 1024² 运行图像 PSNR 仅 29.26 dB（MAE 6.8，max 34）。
   因此像素级 A/B 无法把"优化引入的数值差"与"运行间抖动"分开；本轮对 flash attention 的质量结论
   依据是算法等价性（同一 softmax 注意力的分块实现）+ 无异常统计，而非逐像素相同。
3. 端到端时间噪声：同配置 1024² 的 wall 在 120.5–125.9 s、sampling 在 116.7–119.0 s 之间，
   故只有 ≥3 % 且方向一致的差异才被接受。

---

## 5. 基线（HEAD，auto 配置）

| 分辨率 | 像素 | 重复 | 端到端中位 | DiT 采样 | TE | VAE | 显存峰值(口径: 整卡) | 最小余量 |
|---|---|---|---|---|---|---|---|---|
| 512×512（冒烟） | 0.26 MP | 1 | 32.2 s | 28.1 s | 2.77 s | 0.58 s | 7.04 GB | 5.83 GB |
| 1024×1024 | 1.05 MP | 5 | 124.0 s | 119.0 s | 2.8 s | 1.6–3.3 s | 8.44 GB | 4.27 GB |
| 1328×752 | 1.00 MP | 1 | 114.7 s | 108.5 s | 2.77 s | 2.66 s | 6.88 GB | 5.99 GB |
| 1408×1056 | 1.49 MP | 1 | 184.1 s | 178.1 s | 2.78 s | 2.28 s | 10.08 GB | 2.79 GB |
| 1632×912 | 1.49 MP | 1 | 183.1 s | 177.1 s | 2.77 s | 2.29 s | 10.09 GB | 2.78 GB |
| 1408×1408 | 1.98 MP | 1（无效） | 485.7 s 被杀 | 未完成 | – | – | 7.51 GB（中途） | 5.36 GB |

单步耗时：14.5 s（1024²，16384 image tokens）、3.15 s（512²），即采样 ≈ 96 % 端到端。

---

## 6. 瓶颈证据（每一条都有对照实验）

| # | 假设 | 实验 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | 采样由 INT8 矩阵乘主导 | `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` @512² | 3.15 → **90.5 s/it** | **CONFIRMED**，dp4a 内核即全部热点 |
| 2 | 是否权重带宽受限 | 512² vs 1024² 单步（权重固定 5.16 GB） | 3.15 → 14.5 s，随 token 数而非权重走 | **CONFIRMED 计算受限**，非带宽受限 |
| 3 | GPU 是否被喂满 | 20 ms 采样 `gpu_busy_percent` | 均值 94 %，p90 99 % | **CONFIRMED** 非提交/同步瓶颈 |
| 4 | 注意力占比 | 微基准测得的每层 GEMM 时间 × 42 层，与实测单步对比 | 1024²：GEMM ≈ 9.8 s / 单步 14.5 s（68 %）；512²：≈ 2.45 s / 3.15 s（78 %） | 注意力+norm 约占 22–32 %；注意力长度决定 2 MP 的显存可行性 |
| 5 | 图切分开销 | `--disable-segmented-compute` @512² | 28.6 s vs 28.1 s（慢 1.8 %） | **REJECTED**，切分没有开销 |
| 6 | 日志开销 | `-v` @512² | 28.0 s vs 28.1 s | 无影响 |

有效算力（微基准实测，非推算）：1024² 的 token 数为 **4096**（64×64 latent），
单步 INT8 线性层总量约 39 TFLOP，旧内核 3.97 TFLOP/s（dp4a 峰值约 26.6 TMAC/s 的 **4 %**），
新内核 12.5 TFLOP/s（**24 %**）。差距在内核内部，本轮已在 §13 处理。

---

## 7. 实验记录与判定

### A1 — Flash attention（**ACCEPTED**）

`--flash-attn diffusion=on` 对比 `auto`（auto 在 1024² 解析为 off）。

| 分辨率 | auto 端到端/采样 | flash on 端到端/采样 | 端到端变化 | 显存变化 |
|---|---|---|---|---|
| 1024×1024（n=5/5） | 124.0 / 119.0 s | **119.9 / 114.7 s** | **−4.1 s（−3.3 %）** | 8.44 → 8.44 GB（噪声内） |
| 1328×752 | 114.7 / 108.5 s | 114.6 / 108.3 s | 0（auto 本就 on） | 0 |
| 1408×1056 | 184.1 / 178.1 s | **177.5 / 171.6 s** | **−6.6 s（−3.6 %）** | 10.08 → 9.81 GB |

方向在 5 组独立 A/B 中全部一致，且 flash-on 的 sampling 方差极小（114.7–114.8 s），
auto 的为 116.7–119.0 s（auto 的探针路径带来额外抖动）。

**同时发现的真实缺陷**：`flash attention auto` 在 1024² 用一次微探针（normal 2.567 ms vs flash 2.772 ms）
判定"flash 更慢"，但端到端实测 flash 更快 3.3 %。即 auto 探针在 MOSS-Image 上**系统性选错**。
本轮不修改探针（会影响所有模型、收益未在其它模型验证），只把"显式 pin"作为配置建议上报。

### A2 — Direct convolution（**REJECTED**）

`--conv-direct diffusion=off` @1024²：119.3 s 采样 vs auto 119.0 s（慢 2.2 %）。
auto 在 1024² 已选 on，保持默认即可。

### B1 — VAE 分块（**不适用**）

VAE 解码仅 0.6–3.3 s（≤2.6 % 端到端），且显存峰值出现在 DiT 阶段（见 §8 峰值时刻），
不是峰值来源，因此不引入 tiling（只会更慢）。

### B2 — 后端/Offload（**不采用**）

auto-fit 已把 TE+VAE 参数放 CPU、DiT 放 VRAM；显存峰值不来自可卸载的模块。

### D — 原生推理缓存（**REJECTED**）

`--cache-mode easycache`（默认 threshold 0.2 / start 0.15 / end 0.95）@1024²，2 次：
采样 119.0 s，与基线逐位相同；日志明确输出 **"EasyCache completed without skipping steps"**。
在 8 步、cfg 1.0 的配方下该阈值从不触发，无收益。未尝试更激进的阈值：那会改变数值结果，
属于需要单独授权的质量/速度权衡，不属于本轮"同等质量"的范畴。

---

## 8. 第一轮结果表（旧内核，每分辨率）

> 本节数据采集于 **GEMM 内核改动之前**，用于对比 flash attention 的作用；
> 全部尺寸在新内核下的最新结果见 §13.5。

"auto" 为 HEAD 默认配置；"最终" 为 `--flash-attn diffusion=on`。显存口径为整卡占用（含桌面 ~0.81 GB），20 ms 采样。

| 分辨率 | 像素 | 重复(auto/最终) | 基线端到端 | 最终端到端 | 变化 | 基线显存 | 最终显存 | 变化 | 最低余量 | 稳定性 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512×512 | 0.26 MP | 1 / – | 32.2 s | –（auto 已 on） | 0 | 7.04 GB | 7.04 GB | 0 | 5.83 GB | 成功 | 一致，无需改动 |
| 1024×1024 | 1.05 MP | 5 / 5 | 124.0 s | **119.9 s** | **−3.3 %** | 8.44 GB | 8.44 GB | 0 | 4.27 GB | 成功 | **接受** |
| 1328×752 | 1.00 MP | 1 / 1 | 114.7 s | 114.6 s | −0.1 % | 6.88 GB | 6.88 GB | 0 | 5.99 GB | 成功 | 无差异（auto 已 on） |
| 1408×1056 | 1.49 MP | 1 / 1 | 184.1 s | **177.5 s** | **−3.6 %** | 10.08 GB | 9.81 GB | −0.27 GB | 3.05 GB | 成功 | **接受** |
| 1632×912 | 1.49 MP | 1 / – | 183.1 s | 未测 | – | 10.09 GB | – | – | 2.78 GB | 成功 | 证据不足（同像素数 4:3 已接受） |
| 1408×1408 | 1.98 MP | 1(无效) / 1 | 未完成 | **247.3 s** | 不可比 | 7.51 GB(中途) | 10.43 GB | – | 2.43 GB | 成功（最终配置） | auto 不可用；最终**可运行** |
| 1888×1056 | 1.99 MP | – / 1 | 未测（安全停止） | 250.4 s | – | – | 7.53 GB | – | 5.34 GB | 成功 | – |
| 1152×1728 | 1.99 MP | – / 1 | 未测（安全停止） | 247.9 s | – | – | 10.47 GB | – | 2.40 GB | 成功 | – |
| 1440×1920（配方默认 2.76 MP） | 2.76 MP | – / 1 | 未测 | 385.4 s | – | – | 8.02 GB | – | 4.85 GB | 成功 | – |
| 2048×1952 | 4.00 MP | – / – | – | – | – | – | – | – | – | **SKIPPED_UNSAFE** | 2 MP 已用到 10.4 GB；4 MP 的激活与注意力长度会突破 1.5 GiB 余量 |

阶段耗时（最终配置，s）：TE 2.5–3.2；DiT 采样 114.7（1024²）／171.6（1.49 MP）／240.5–241.4（2 MP）／372.7（2.76 MP）；
VAE 解码 1.6–3.3；进程启动+落盘 ≈ 4–13 s（含权重首次读取，冷/热页缓存差异）。

---

## 9. 可复现执行记录

- 单次运行包装：`run_variant.sh <label> <W> <H> [额外参数]`；批处理：`batch_*.sh`。
- 每次运行的完整命令行、Git commit、返回值、阶段耗时都在 `raw-records.jsonl` 里。
- 实验控制项（配方中没有、为可复现而固定）：
  提示词 `a photograph of a young woman standing on a city street at golden hour, shallow depth of field, 85mm lens, natural skin texture, highly detailed`，种子 `-s 42`。
- 配方文件全程只读，SHA-256 前后一致（§1）。

---

## 10. 推荐配置

```bash
./build-vulkan/bin/sd-cli \
  --diffusion-model /home/trave/models/diffusion_models/mossRefinedRealism_v208steps_int8.safetensors \
  --llm /home/trave/models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --tokenizer /home/trave/models/tokenizers/qwen35-tokenizer.json \
  --vae /home/trave/models/vae/MOSS-Image-SFT-vae.safetensors \
  --flash-attn diffusion=on \
  -W 1440 -H 1920 --steps 8 --cfg-scale 1.0 --sampling-method euler --flow-shift 5.0 \
  -p "<提示词>" -o out.png
```

- **建议启用**：`--flash-attn diffusion=on`（唯一有可靠收益的开关；2 MP 以上是必需项）。
- **保持默认**：`--conv-direct`（auto 已选对）、`--vae-tiling`（关闭，VAE 不是瓶颈）、
  `--cache-mode`（8 步下不触发）、`--disable-segmented-compute`（更慢）、线程数（默认=物理核）。
- **不需要按分辨率切换**：同一配置在 1 MP→2.76 MP 全部一致变快或不劣化；不引入尺寸相关分支。
- **速度/显存权衡**：本方案不牺牲任一指标；`--cache-mode` 一类近似缓存被实测无收益（8 步），
  若未来要试，必须单独做质量实验。
- **可信度**：1024² 与 1.49 MP 的结论有 5 次/2 次重复与一致方向支撑；
  2 MP 及以上只有单次运行（自动路径不可用，无法做 A/B），属"可运行性证据"而非严格加速证据。

---

## 11. 未解决问题与后续方向

1. **INT8 GEMM 内核效率**（真正的瓶颈）：`ggml-vulkan` 的 `mul_mat_i8_tensorwise` 只有 dp4a 峰值的约 12 %。
   Navi22 无矩阵核心，coopmat int8 变体不可用。要突破必须改 `ggml/`（本轮按规则未动）；
   可行方向：分块尺寸/寄存器压强、把 `quantize_i8_convrot` 与 matmul 融合、按行分块提高权重复用。
   建议先向上游提 issue 并附本报告 §6 的 29× 对照数据。
2. **auto 探针选错**：flash attention 探针（`src/core/probe/flash_attn.cpp`）以单算子微秒级对比决定开关，
   在 MOSS-Image 上给出与端到端相反的选择。修法有两种：按模型/尺寸给经验默认，或把探针改成端到端可代表形状；
   两者都会影响其它模型，需要单独评估，本轮未改。
3. **4 MP 与 1440×1920 的 auto 路径**：自动配置在 2 MP 已不可用；如需在 auto 下支持大图，
   需要自动选择 flash attention（与第 2 点合并解决）。
4. **运行间不可复现**（PSNR 29.3 dB）：影响所有像素级质量验证。若后续要评估近似算法，
   建议先定位非确定来源（候选：auto-fit 依据实时空闲显存生成的分段计划、以及注意力浮点归约顺序），
   或在 `--rng cpu` 下复测。

---

## 12. 交付物

| 文件 | 说明 |
|---|---|
| `bench.py` | 单次运行测量（端到端/阶段/20 ms 显存/安全闸/GPU 遥测），输出 JSONL |
| `run_variant.sh` | 配方等价的单变量运行器 |
| `batch_a.sh` `batch_cd.sh` `batch_f.sh` | 各轮批处理（A/B 交错） |
| `summarize.py` | 按变体聚合中位数/极值 |
| `compare_png.py` | PNG 像素差（MAE/RMSE/PSNR），用于质量门 |
| `raw-records.jsonl` | 全部 36 次运行的原始记录 |
| `summary.csv` | 轻量汇总 |
| `ggml_i8bench.cpp` | INT8 GEMM 微基准（真实形状、latency/TOPS、`--dump` 逐位指纹） |
| `STATUS.md` | 执行期检查点（进度、瓶颈证据、下一步） |
| `REPORT.md` | 本报告 |

---

## 13. 第二轮：INT8 GEMM 内核优化（已获授权修改 `ggml/`）

### 13.1 微基准（先测后改）

`ggml_i8bench.cpp` 用模型真实形状（rows=4096，即 1024² 的 64×64 latent token 数）直接调用
`ggml_quantize_i8_convrot` + `ggml_mul_mat_i8_tensorwise`：

| 形状 | 旧内核 | 新内核 | 加速 | 旧 TOPS | 新 TOPS |
|---|---|---|---|---|---|
| qkv 3072→9216 | 58.14 ms | **18.55 ms** | 3.13× | 3.99 | 12.50 |
| ffn_up_gate 3072→16384 | 103.65 ms | **32.61 ms** | 3.18× | 3.98 | 12.64 |
| ffn_down 8192→3072 | 52.22 ms | **17.19 ms** | 3.04× | 3.95 | 11.99 |

换算为 MAC 吞吐：约 2.0 → 6.3 TMAC/s，即 dp4a 峰值的 **约 4 % → 约 24 %**（6700 XT 峰值约
26.6 TMAC/s @2462 MHz）。

### 13.2 旧内核为什么慢

```glsl
// 旧：16×16 输出 tile，1 个累加器/线程，每 32 个 int8 一次 barrier×2
shared int weights[16*8]; shared int activations[16*8];
for (k_base += 8) { 载入 128+128 int；barrier(); 8 次 dp4a；barrier(); }
```
- 每线程只有 1 条依赖链 → dp4a 延迟无法被掩盖；
- 每次 barrier 只服务 16×16×32 = 8192 MAC，K=3072 时共 96 次（192 个 barrier）；
- 每个权重/激活 slice 只被 16 个输出复用 → 共享内存访问指令 : MAC ≈ 1:4。

### 13.3 新内核设计（`mul_mat_i8_tensorwise.comp`）

- 输出 tile **64 列 × 32 行**（`TILE_N=64, TILE_R=32`），16×16 线程，每线程 **2 行 × 4 列 = 8 累加器**；
- K 阶段 `KC=8`（32 个 int8），载入 512 个权重 int + 256 个激活 int（256 线程各 2/1 个）；
- 内层：每个 packed k 取 4 个权重 + 2 个激活，做 8 次 `dotPacked4x8EXT` → 每条共享内存指令对应
  5.3 个 MAC（旧为 4）；
- 每 barrier 服务 64×32×32 = 65536 MAC（旧的 8×），barrier 总次数不变但摊薄 8 倍；
- 边界处理：越界列/行以 0 填充且写入前判界，`n`/`rows` 非 64/32 倍数仍正确；
- 分发侧：`ggml_vk_create_pipeline(..., {64, 32, 1}, ...)`（原 `{16,16,1}`），dispatch 仍传
  `{n, rows}`（`CEIL_DIV` 换算），因此 C++ 侧只改 1 行。

实测扫描过的变体（均逐位正确，均更慢，已回退）：

| 变体 | qkv ms | 结论 |
|---|---|---|
| 64×32，KC=8，8 acc（**采用**） | 18.55 | 最优 |
| 64×64，16 acc | 22.71 | 寄存器压力，慢 22 % |
| 64×32，KC=16 | 20.02 | barrier 不是瓶颈 |
| 128×32，16 acc | 25.00 | 慢 35 % |

### 13.4 正确性验证（三重）

1. **逐位相同**：同一输入下新旧内核输出 FNV-1a 指纹完全一致
   （qkv `cec6e8140dd9cd29`、upgate `142a8989aae8a75f`、down `7be4752580175e41`），
   `max_abs = 0`。原因是 int32 累加是精确的，最终只做一次 `float(sum)*ascale*wscale`，
   故与累加顺序无关。
   （中途发现：无条件 `+0.0f` 会把 `-0.0` 变成 `+0.0`，产生 41.8 万个"仅符号零"差异；
   改为仅在 `has_bias` 时加 bias 后完全一致。）
2. **仓库自带测试**：`ggml/tests/test-int8-convrot.cpp`（独立编译，`--vulkan`）
   → `INT8 convrot Vulkan test passed`。
3. **端到端图像**：新旧 PNG 的像素差异（PSNR 19.4 dB）与"同一配置两次运行之间"的差异
   （PSNR 18.6 dB）同量级，说明差异来自既有的运行间非确定性（见 §4），不是本次改动引入。

### 13.5 端到端结果（全部 `--flash-attn diffusion=on`）

| 分辨率 | 像素 | 端到端(旧内核) | 端到端(新内核) | 变化 | 采样(旧→新) | 显存峰值 |
|---|---|---|---|---|---|---|
| 512×512 | 0.26 MP | 32.2 s | **16.3 s** | −49 % | 28.1 → 12.5 s | 6.86 GB |
| 1024×1024 | 1.05 MP | 119.9 s | **55.1 s** | −54 % | 114.7 → 49.1 s | 6.75–8.37 GB |
| 1328×752 | 1.00 MP | 114.6 s | **51.3 s** | −55 % | 108.3 → 46.3 s | 8.29 GB |
| 1632×912 | 1.49 MP | 183.1 s¹ | **77.5 s** | −58 %¹ | 177.1 → 72.1 s | 9.67 GB |
| 1408×1408 | 1.98 MP | 247.3 s | **119.8 s** | −52 % | 240.5 → 113.4 s | 10.59 GB / 余 2.28 |
| 1152×1728 | 1.99 MP | 247.9 s | **122.7 s** | −51 % | 241.1 → 113.9 s | 7.38 GB |
| 1888×1056 | 1.99 MP | 250.4 s | **120.2 s** | −52 % | 241.4 → 114.0 s | 10.33 GB / 余 2.54 |
| 1440×1920（配方默认） | 2.76 MP | 385.4 s | **192.8 s** | −50 % | 372.7 → 181.8 s | 7.87 GB / 余 4.99 |

¹ 1632×912 的"旧"数据是 auto（flash off）配置，因此该行同时含两个变量，仅作量级参考。

**显存**：内核改动不影响驻留权重，峰值在既有噪声区间内（同尺寸重跑可差 1.6 GB），
最低余量仍出现在 2 MP（2.28 GB，高于 1.5 GiB 安全线）。

### 13.6 仍未利用的优化空间

新内核约在 dp4a 峰值的 24 %，剩余瓶颈推测为共享内存访问指令占比与 LDS bank 冲突
（`w_sh[kk][4*lx+c]` 为 2 路冲突）。未继续深入的原因：本轮已取得 3.1× 内核 / 2× 端到端的
可验证收益，进一步调优需要 RGP/`rgp` 级别的性能计数器，而本机 RADV 环境没有可用的 GPU
profiler；盲调信号不足。

后续可选（按预期收益排序）：
1. `ivec4` 向量化权重共享内存读取（预计 +20–30 % 指令密度）；
2. 共享内存 padding 消除 2 路 bank 冲突；
3. 用 `subgroupBroadcast` 让同一 warp 内复用权重（需重构映射，风险较高）；
4. 上游化：该内核在 `ggml/` 子模块，建议以 15 行级 diff 提 PR，并附 §13.1 的微基准与
   逐位一致性证据。

---

## 14. 第二轮迭代：共享内存向量化 / bank 冲突 / warp 广播

### 14.1 硬件计数器是否可得：不可得（已尝试）

用户安装了 `radeon-gpu-profiler`，但该包**只有 GUI 与 `radeonreg`/`radeontool`，
没有 RGP 采集 CLI（`rgp`）也没有 Vulkan 采集层**（全盘搜索
`*RadeonDeveloperLayer*` / `libVkLayer_*radon*` 均无结果，`/usr/share/vulkan` 下无对应 json）。
`RADV_DEBUG=shaderstats`（Mesa 26.2.5 中确有该 token）在本环境无任何输出。
因此本轮**没有任何硬件计数器可用**，判定依据为：微基准 A/B + SPIR-V 静态指令计数 + 逐位一致性。

### 14.2 三个方向的实验结果

| 变体 | qkv ms | upgate ms | down ms | qkv TFLOPS | 判定 |
|---|---|---|---|---|---|
| 原始内核（16×16，1 累加器） | 58.14 | 103.65 | 52.22 | 3.99 | 基线 |
| 64×32，8 累加器，KC=8（§13） | 18.51 | 32.58 | 17.26 | 12.53 | 接受 |
| ① + `ivec4/ivec2` 共享内存读取 | 23.01 | 41.22 | 21.97 | 10.08 | **REJECTED** |
| ② 线程列映射 `col = 16*c + lx`（消除 2 路冲突） | **15.48** | **26.92** | **14.80** | **14.98** | **ACCEPTED** |

**① 为什么失败**：向量化读取要求每个 lane 一次取 4 个**连续列**，而全局权重布局是
`w[(col)*k_packed + k]`，4 个连续列相距 `k_packed*4 B`，于是每个 `ivec4` 需要 4 次
分离的 32 B 扇区读取 → 权重流的实际取入量放大约 8 倍（每 stage 2 KB 有效数据取 8 KB）。
节省的内层 6→2 条 LDS 指令抵不过这一步的取数损失。结论：**在该布局下，staging 的合并访问
比内层指令密度更重要**。

**② 为什么有效**：原来 16 个 lane 的列索引是 `4*lx + c`（步长 4），
任一 32 位 LDS 的 16 个 lane 只落在 8 个 bank 上 → 恒定 2 路冲突。
把每线程的 4 列改成 `{lx, lx+16, lx+32, lx+48}` 后，同一指令的 16 个 lane 读
**连续 16 个字** → 无冲突；写回也仍然按行合并（固定 c 时 16 个 lane 写连续 16 列）。
纯映射调整，零成本，微基准 +20 %。

**③ warp 内广播复用权重：分析后放弃**。当前映射下一个 warp 的 lx 为 0..15 重复两次
（ly=0/1），两半确实读同一组权重，但替换为 `subgroupBroadcast` 只是把 1 条 LDS 换成
1 条 shuffle，指令数不变；而真正能提高权重复用率的"更多累加器"方向（64×64 / 16 累加器、
128×32 / 16 累加器）已实测更慢（22.7 / 25.0 ms），原因是 VGPR 压力。
在 8 累加器的最优配置下，warp 级广播没有可兑现的收益。

### 14.3 端到端结果（`--flash-attn diffusion=on` + 最终内核）

| 分辨率 | 原始内核 | 最终内核 | 变化 | 采样(原始→最终) | 显存峰值 |
|---|---|---|---|---|---|
| 1024×1024 | 119.9 s | **51.5 s** | **−57 %** | 114.7 → 45.3 s | 8.46 GB |
| 1408×1408（2 MP） | 247.3 s | **113.1 s** | **−54 %** | 240.5 → 105.7 s | 10.29 GB |
| 1440×1920（配方默认） | 385.4 s | **183.4 s** | **−52 %** | 372.7 → 171.1 s | 7.90 GB |

（512² 仅在 §13 的中间内核下复测过：32.2 → 16.3 s。）

### 14.4 正确性

- 全部四个变体的输出 FNV-1a 与原始内核一致（`cec6e814…` / `142a8989…` / `7be47525…`），
  `max_abs = 0`；
- `test-int8-convrot --vulkan` 通过；
- 端到端 PNG 正常产出。

### 14.5 当前所处位置与后续

内核累计 **3.5–3.9×**（58.1 → 15.5 ms 等），约 dp4a 峰值的 **28 %**（7.5 / 26.6 TMAC/s）。
按内层指令配比（每 14 条指令 8 次 dp4a）算，纯发射上限约 57 %，即目前已达发射上限的一半；
剩余损失来自 staging、barrier 与延迟/占用率。

若要继续，优先级：
1. 共享内存布局改为 `[k][col]` 且每列 9-int 奇数步长 + 逐列 staging（全局合并、读无冲突），
   配合 `ivec4` 需要 16 B 对齐会退化到 2 路冲突 —— 需要 trade-off 实测；
2. 尝试 32×8 线程组（warp 全部落在 x 方向，权重读取天然连续）；
3. 双缓冲 staging（用 `barrier()` 之外的异步拷贝或 ping-pong 共享缓冲隐藏取数），
   代价是共享内存翻倍（3 KB → 6 KB，仍远低于 64 KB）。

---

## 15. 第三轮迭代：profiler 可用性、双缓冲、KC 复核

### 15.1 Profiler 可用性：全部 BLOCKED（逐项尝试记录）

| 工具 | 状态 | 证据 |
|---|---|---|
| RGP GUI 包 (`radeon-gpu-profiler`) | 无采集层 | 全盘无 `*RadeonDeveloperLayer*` / `libVkLayer_*radon*`，`/usr/share/vulkan` 无对应 json，包内无 `rgp` CLI |
| Radeon Developer Tool Suite (`~/tools/RadeonDeveloperToolSuite-2026-10-02-1872`) | 有 `rga`(2.15)、GUI 版 `RadeonGPUProfiler`、`RadeonDeveloperPanelCLI`，仍**无 `rgp` CLI 与采集层** | `rga -s vk-offline -c gfx1031 --comp … --isa … -a …` 只打印目标 GPU 列表便退出，不产出 ISA/stats；`-c navi22` 报 `could not detect target GPU` |
| `RADV_DEBUG=shaderstats` | 无输出 | Mesa 26.2.5 二进制内含该 token，但运行无任何 stderr |
| `RADV_DEBUG=shaders` | 不落盘 | 工作目录无 `shader_*` 文件 |

结论：本机无法取得任何硬件或静态计数器，判定依据只能是微基准 A/B + 逐位一致性 + 静态指令配比。

### 15.2 实验结果

| 变体 | qkv ms | upgate ms | down ms | qkv TFLOPS | 判定 |
|---|---|---|---|---|---|
| 原始（16×16，1 acc） | 58.14 | 103.65 | 52.22 | 3.99 | 基线 |
| 64×32，8 acc，KC=8 | 18.51 | 32.58 | 17.26 | 12.53 | 接受 |
| ① `ivec4/ivec2` 读取 | 23.01 | 41.22 | 21.97 | 10.08 | 拒绝（staging 取数放大 8×） |
| ② 冲突无关列映射（`16*c+lx`） | 15.48 | 26.92 | 14.80 | 14.98 | 接受（+20 %） |
| 双缓冲 staging（①②之上） | **14.95** | **26.09** | **14.25** | **15.52** | 接受（+3.5 %，内核级） |
| 双缓冲 + KC=16 | 16.23 | 28.59 | 14.87 | 14.29 | 拒绝 |

累计（对原始内核）：qkv **3.89×**、upgate **3.97×**、down **3.66×**，约 dp4a 峰值的 **29 %**。

### 15.3 端到端（最终内核，`--flash-attn diffusion=on`）

| 分辨率 | 原始内核 | 最终内核 | 变化 | 采样 | 显存峰值（本次） |
|---|---|---|---|---|---|
| 1024×1024 | 119.9 s | **51.5 / 53.7 s** | −55 ~ −57 % | 114.7 → 44.9–45.3 s | 7.04–8.46 GB |
| 1408×1408（2 MP） | 247.3 s | **113.1 / 113.6 s** | −54 % | 240.5 → 104.5–105.7 s | 7.65–10.29 GB |
| 1440×1920（配方默认） | 385.4 s | **182.3 / 183.4 s** | −52 % | 372.7 → 169.8–171.1 s | 7.90–8.15 GB |

**诚实说明**：双缓冲带来的 3.5 % 内核级提升**低于端到端噪声**（同配置重跑 wall 波动约 ±4 %，
51.5 vs 53.7 s 无法区分）。保留它的理由是与②同向、逐位一致、无回归，而不是可测的端到端收益。

### 15.4 剩余空间

内核现约 dp4a 峰值 29 %；按内层配比（每 14 条指令 8 次 dp4a）发射上限约 57 %，
已达上限的一半。要继续需要计数器（判断是 VGPR 限制、LDS 带宽还是 latency），
而本机三条获取路径全部不可用，因此**在没有 profiler 的情况下盲调已进入收益/风险不划算的区间**。

## 16. 预览 TAE 的 backend 放置（GPU vs CPU，2026-10-30）

### 16.1 问题与改动

原有代码里 `tae` 只是 `vae` 模块的别名，preview TAE 与正式 VAE 共用
`SDBackendModule::VAE` 的 runtime backend，因此不存在"只把预览搬到 CPU"的开关
（`--backend vae=cpu` 会连最终解码一起搬）。本次给预览 TAE 增加独立模块
`preview_vae`：未显式赋值时完全回退到 `vae`（默认行为不变），赋值后只影响预览。

```shell
sd-cli ... --taesd taef2.safetensors --taesd-preview-only --backend tae-preview=cpu
```

实现：`SDBackendModule::PREVIEW_VAE` + 解析别名（`tae-preview` / `vae-preview` /
`preview`），`module_assignment_name()` 回退到 `vae`；`model_builders.cpp` 用该模块
创建 preview TAE；`diffusion_engine.cpp` 用该模块注册 params 与解析 FA/conv 探测。

### 16.2 测量（sd-cli，8 步，`--preview tae --preview-interval 1` = 8 次预览）

| 分辨率 | 组合 | sampling s | generate s | 显存峰值 | 最低空闲 | 每次预览增量 |
|---|---|---|---|---|---|---|
| 1024² | 不预览（3 次） | 48.00 / 49.44 / 48.34 | 55.4 / 57.8 / 54.0 | 9.77 / 9.80 / 10.07 GB | 1.92–2.21 GB | — |
| 1024² | 预览=proj | 47.65 | 55.01 | 9.82 GB | 2.16 GB | ≈0 |
| 1024² | 预览=TAE @ GPU | **安全中止 @14.1 s** | — | ≥10.66 GB | **1.32 GB** | 未完成 |
| 1024² | 预览=TAE @ CPU（新） | 97.15 / 94.76 | 103.0 / 100.3 | 10.08 / 10.07 GB | 1.91 / 1.92 GB | ≈5.9 s |
| 1024² | `--backend vae=cpu`（旧） | 94.57 | 142.35 | 9.77 GB | 2.21 GB | 5.8 s + 最终解码 +40 s |
| 768² | 不预览 / GPU / CPU | 25.05 / 26.40 / 51.93 | 30.3 / 31.0 / 57.6 | 8.50 / 9.00 / 8.52 GB | 3.49 / 2.99 / 3.46 GB | — / 0.17 s / 3.36 s |
| 512² | 不预览 / GPU / CPU | 12.26 / 13.18 / 23.23 | 16.9 / 17.4 / 27.5 | 9.37 / 9.37 / 8.50 GB | 2.61 / 2.61 / 3.48 GB | — / 0.12 s / 1.37 s |

每张预览图 8 次写盘（512/768/1024）确认生成，1024² TAE 预览为全分辨率图，`proj` 仅 128²。

### 16.3 结论

1. **可行**：`--backend tae-preview=cpu` 只把预览放 CPU，DiT/TE/VAE 不动；默认不设置时预览仍跟随 `vae`，
   512/768 的默认 GPU 预览回归正常。
2. **CPU 预览的代价**：1024² 每次约 **5.9 s**（8 步 × interval 1 → 采样 48.6 → 96.0 s，翻倍）；
   768² 约 3.4 s，512² 约 1.4 s，基本随像素线性（≈5.6 µs/px）。`proj` 预览则无可测开销。
3. **GPU 预览的代价**：768² 峰值 +0.50 GB；512² 无可见增量；1024² 在桌面占 ~3 GB 时
   把空闲显存压到 1.32 GB，触发本实验的 1.5 GiB 安全门禁，**未能完成一次预览解码**
   （按面积外推约 0.18 s/次，属估算而非测量）。
4. **CPU 预览不降低峰值**，它的收益是"预览不再叠加 ~0.5–1 GB"：1024² 峰值与不预览持平
   （10.07 vs 9.77–10.07 GB），空闲显存保持在 1.9 GB 以上。
5. 旧的 `--backend vae=cpu` 能达到同样的预览放置，但最终解码也上 CPU：generate 142.35 s vs
   新开关的 103.0/100.3 s（GPU 解码 3.35 s，CPU 解码约 +40 s）。

实用建议：要 **显存余量**（高分辨率/与桌面共存）→ `--backend tae-preview=cpu`，并把
`--preview-interval` 调到 4（1024² 约 +12 s 而非 +47 s）；要 **速度** → 保持默认（GPU），
前提是显存够；只想看进度 → `--preview proj`，开销≈0。

## 17. 预览 TAE 的"大与慢"来自 im2col，不是 TAE 本身（对 §16 结论的更正）

§16 的数据没有错，但结论方向需要更正：把预览搬到 CPU 只是绕开问题。真正的成本是
ggml 默认 conv 路径（im2col + matmul）在 VAE/TAE 高分辨率层上物化巨大的 im2col 矩阵。

### 17.1 证据

| 组合（1024²，8 步，8 次预览） | sampling | 每次预览 | 显存峰值 | 最低空闲 | 最终 VAE 解码 |
|---|---|---|---|---|---|
| 不预览，im2col（§16 A） | 48.0 / 49.4 / 48.3 s | — | 9.77–10.07 GB | 1.92–2.21 GB | 3.35 s |
| 预览 @ GPU，im2col（§16 B） | 14.1 s 被安全门禁中止 | 未完成 | ≥10.66 GB | **1.32 GB** | — |
| 预览 @ GPU，**direct conv** | 49.39 / 51.13 s | **0.19–0.41 s** | 9.85–10.01 GB | 1.98–2.13 GB | **1.59 s** |
| 不预览，direct conv | 47.88 s | — | 9.44 GB | 2.54 GB | 1.59 s |
| 预览 @ CPU，direct conv | 97.91 s | 5.9 s | 9.26 GB | 2.73 GB | 1.59 s |

- CPU 预览开不开 direct 都是 ~5.9 s/次（96.0 → 97.9 s），说明 CPU 侧瓶颈是 F32 算力（每次预览约几 GFLOP），
  不是 im2col。
- GPU 侧相反：im2col 在 1024² 要额外 ~0.6–1 GB 临时显存并被门禁拦下；direct conv 后预览 0.2–0.4 s/次、
  峰值不涨，**最终解码也从 3.35 s 降到 1.59 s**。
- 2 MP（1408²，`--flash-attn diffusion=on`）：预览 @ GPU + direct conv 采样 109.9 s
  （基线 104.5–105.7 s，≈0.5–0.6 s/次），最终解码 2.89 s（基线 5.59 s），峰值 9.83 GB、空闲 2.16 GB。
  （同尺寸不带 `--flash-attn` 时 auto 关闭 FA，实测 244 s/步，属配置错误，已中止。）

### 17.2 为什么 auto 没选 direct

`src/core/probe/conv_direct.cpp` 的 Vulkan 探针只用 32×32/64ch 与 16×16/128ch 两个小 shape，
此处 im2col 与 direct 几乎一样快（实测 0.362 vs 0.380 ms），于是判 off。VAE 的真实卷积在
512²–1024²、64ch 量级，im2col 要物化 9× 数据，direct 明显更优——探针 shape 与用途不匹配。
这是可修的上游候选（把 VAE 典型尺寸加入探针），但会改变所有 Vulkan 用户的默认行为，需单独评估。

### 17.3 结论

- 预览 TAE 不需要（也不应）搬 CPU：保持默认 GPU，并给 VAE 模块开 `--conv-direct vae=on`
  （预览模块未显式赋值时继承 vae）。1024² 预览 0.2–0.4 s/次、2 MP 0.5–0.6 s/次，显存峰值不变。
- `--backend tae-preview=cpu` 只在显存被其它常驻占用挤爆、且不在乎预览延迟时使用，
  并配合 `--preview-interval 4`（1024² 约 +12 s）。
- 预览回调仍是同步的（`step_callback` 在采样循环内），ggml-vulkan 单队列下 DiT 与 TAE 无法真并发；
  修好 conv 路径后预览仅占采样的 2–4 %，并行的收益不值得引入第二个执行流。

## 18. auto 决策审计：还有哪些 auto 会选到更差的路径

本仓库里带自动决策的开关共 10 处。探针型只有两个（`flash_attn`、`conv_direct`），
其余是规则型（auto-fit、LoRA 应用方式、VAE 格式、图像预处理、自动切图、失败重试等）。
逐项核对 MOSS-Image / 1024²–2 MP / RDNA2 Vulkan 的实际负载：

| auto | 决策规则 | 本次判定 | 实测（1024²，8 步） | 结论 |
|---|---|---|---|---|
| `--flash-attn auto`（默认） | 2 个探针 shape：512 tok×4 heads×dim64、1024 tok×4 heads×dim128；5 次交替取中位，合计快 ≥5 % 才 on | **off**（2.460 vs 2.735 ms） | diffusion 关 FA：采样 48.0/49.4/48.3 s；`diffusion=on`：**43.38 s**（≈−10 %）。2 MP：无 FA **244 s/步**（已中止）vs 13.7 s/步 | **误判** |
| `--conv-direct auto`（默认，2D） | 2 个探针 shape：32²×64ch、16²×128ch | **off**（0.362 vs 0.380 ms） | 最终解码 3.35→**1.59 s**；预览从"≥1 GB scratch、触发 1.5 GiB 门禁"变为 **0.2–0.4 s/次**、峰值不涨；2 MP 解码 5.59→**2.89 s** | **误判** |
| `--conv-direct auto`（3D） | 同上，16³ 级小 shape | 未测（本模型无 3D conv） | 视频 VAE（Wan/LTX/Hunyuan）会走同一探针，很可能同样误判 | 待验证 |
| `--flash-attn auto`（TE 模块） | 同上探针 | **off** | TE 关 FA 2.17 s vs 开 FA 3.16 s | **判断正确** |
| `--auto-fit on`（默认） | 规则：compute GPU（留 ~2 GiB）→ RAM → 其他 GPU → 磁盘 | diffusion=GPU，te/vae params=CPU | 与显式 `--offload-to-cpu` 对比：采样 43.38 vs 43.90 s（≈1 %），峰值 8.40 vs 7.46 GB | 正常 |
| `--lora-apply-mode auto` | LoRA 存在时选择立即/运行时应用 | 无 LoRA | — | 不适用 |
| `--vae-format auto` | 仅当模型 trait 允许覆盖 | MOSS 未启用覆盖 | — | 不适用 |
| `--image-preprocess …auto` | 图像输入预处理规则 | 文生图不触发 | — | 不适用 |
| `--max-vram`（负值自动预算） | 空闲 −N 作为预算 | 未使用 | — | 不适用 |
| 自动切图 / 分段执行 | 每次 compute 生成图切分计划 | 启用 | 此前实测优于 `--disable-segmented-compute`（见 §7） | 正常 |
| VAE 解码失败自动 tiling | 仅失败后重试 | 未触发 | — | 正常 |

### 18.1 两个误判的共同成因

`sd_run_backend_probe` 的规则本身没问题（交替顺序、5 次中位、合计需快 5 %、1 s 预算）。
问题在 **shape 与真实负载不匹配**：

- FA 探针是 4 heads、512/1024 tokens。MOSS 在 1024² 是 **24 heads、4096 tokens**（2 MP 时 11 568），
  注意力工作量差 24–72 倍；小 shape 下两种实现都被 launch 开销主导（2.46 vs 2.74 ms），
  而真实尺寸下 FA 省的是中间张量物化。
- conv 探针是 32²/16²。VAE 的真实卷积在 512²–1024²、64ch 量级，im2col 要物化 9× 数据，
  两个小 shape 完全覆盖不到这个区间。
- 反例：同一 FA 探针对 TE（LLM 解码段）判 off 是对的（2.17 vs 3.16 s）——不是探针错，
  而是它只对"和探针尺寸同量级"的负载有效。

### 18.2 建议

1. 服务端显式固定：`--flash-attn on`（或 `diffusion=on`）＋ `--conv-direct vae=on`。
   两者都是"验证过更好"的显式覆盖，不要再依赖 auto。
2. 上游候选修复（需单独评估，会改变所有 Vulkan 用户的默认）：
   - FA 探针用模型真实 head 数/head_dim，并至少包含一个 4096-token 的 case；
   - conv 探针加入 256²/512² × 64ch 的 case，让合计金额由真实区间主导。
3. `--auto-fit` 在 12 GB + 桌面 ~1–3 GB 的机器上判断合理（与显式 offload 打平，
   显式 offload 只多换 ~1 GB 余量）；不需要覆盖。

## 19. 端到端逐算子分解（GGML_VK_PERF_LOGGER，2026-10-30）

`GGML_VK_PERF_LOGGER=1` 用 Vulkan timestamp query 池逐节点计时（RGP 不可用时的替代），
1024²、8 步、`--flash-attn diffusion=on --conv-direct vae=on`，8 个采样图完全一致。
每步 GPU 计时合计 **5.02 s**，同配置采样墙钟 5.42 s/步 → **0.40 s/步（7 %）在 GPU 计时之外**
（CPU 建图/派发/同步）。原始日志：`raw/perf-1024-vk-logger.log`。

| 算子 | 每步 ms | 占比 | 每次调用 | 效率 |
|---|---|---|---|---|
| FLASH_ATTN_EXT（4096 tok, 24 heads, d=128） | 1604 | **32.0 %** | 38.2 ms × 44 | 5.45 TFLOP/s ≈ fp16 峰值的 43 % |
| MUL_MAT i8 ffn_up_gate (16384×4120×3072) | 1076 | 21.4 % | 25.7 ms × 42 | 16.4 TFLOP/s |
| MUL_MAT i8 qkv (9216×4120×3072) | 612 | 12.2 % | 14.6 ms × 42 | 16.2 TFLOP/s |
| MUL_MAT i8 ffn_down (3072×4120×8192) | 554 | 11.0 % | 13.2 ms × 42 | 15.4 TFLOP/s |
| MUL_MAT i8 proj.o (3072×4120×3072) | 210 | 4.2 % | 5.1 ms × 42 | 15.3 TFLOP/s |
| **i8 GEMM 小计** | **2452** | **48.8 %** | — | **≈ 32 % of dp4a 峰值** |
| MUL（调制/逐元素） | 196 | 3.9 % | 0.76 ms × 257 | 内存受限 |
| RMS_NORM_MUL | 188 | 3.7 % | — | 内存受限 |
| SCALE | 138 | 2.7 % | 0.78 ms × 176 | 内存受限 |
| ADD | 102 | 2.0 % | 0.37 ms × 271 | 内存受限 |
| ROPE_APPLY / TANH / SIGMOID（situ） | 172 | 3.4 % | — | — |
| CONT / CPY（布局转换） | 98 | 2.0 % | — | — |
| QUANTIZE_I8_CONVROT | 53 | 1.0 % | 0.31 ms × 168 | 可忽略 |

### 19.1 GEMM 的 32 % 不是带宽限制

`i8bench` 行数扫描（rows 1024→8192，同形状）：

| rows | qkv TFLOP/s | upgate TFLOP/s | down TFLOP/s |
|---|---|---|---|
| 1024 | 15.5 | 16.1 | 14.0 |
| 2048 | 16.1 | 16.2 | 14.7 |
| 4096 | 16.2 | 16.4 | 14.9 |
| 8192 | 16.3 | 15.7 | 15.4 |

行数增加 8 倍（权重复用次数 8 倍）吞吐不变 → 不是权重带宽受限，而是内核执行效率平台
（LDS/指令发射/延迟隐藏）。按内层配比（每 8 次 dp4a 约 14 条指令）理论发射上限约 57 %，
实测 32 %，即**只用到发射上限的 55 %**。

### 19.2 现在的瓶颈排序

1. **i8 GEMM 49 %**（32 % 峰值）——下一个可动的量是内层指令配比：增加每线程行数（RPM 2→4）
   可把 dp4a 占比从 ~57 % 提到 ~73 %，若用 `ivec4` 一次加载 4 行激活（`a_sh[kk][4*ly..4*ly+3]`
   连续且 16 B 对齐）还能再省 3 条 LDS；代价是 8→? 累加器与 VGPR 压力（16 累加器此前实测 −22~35 %）。
2. **Attention 32 %**（43 % fp16 峰值）——6 倍于 GEMM 的每 FLOP 成本，但改进需要动 ggml 的
   FA shader（tile 配置对 head_dim=128 未必最优），风险与工作量都大。
3. **逐元素 18 %**（多数在 ~130–200 GB/s，低于 384 GB/s 峰值）——调制 `y + y*p1 + p0` 每块 6 次
   全尺寸读写；融合成一次仿射需要新算子。
4. **GPU 计时外 7 %/步**（CPU 建图/派发/同步）。

## 20. 四项瓶颈实验（2026-10-30 晚）

### 20.1 GEMM 内层：RPM 2→4 + `ivec4` 激活加载（接受）

内核改为 64 行 × 32 列 tile、每线程 4 行 × 2 列（仍 8 累加器），激活的 4 行在 shared 中连续且
16 B 对齐，用一次 `LDS.128` 取回；内层从每 8 次 dp4a / 14 条指令变为 / 11 条。

| 指标 | 旧内核 | 新内核 |
|---|---|---|
| FNV-1a（qkv/upgate/down） | cec6e814 / 142a8989 / 7be47525 | **完全相同** |
| qkv TFLOP/s（rows 4096，5 次 ×2 轮） | 16.23 / 16.25 | **16.77 / 16.73** |
| upgate | 16.38 / 16.40 | **16.97 / 17.02** |
| down | 14.97 / 15.06 | **15.41 / 15.48** |
| 1024² 采样（FA on, direct conv） | 43.38 s | **42.40 / 42.49 s** |
| 1024² wall | 47.16 s | **46.17 / 46.26 s** |

内核 +3.1~3.7 %，逐位一致；端到端 −2.2 %（与 GEMM 占 49 % × 3.4 % 的预期一致）。
理论发射上限 73 % 只兑现了 3 %，说明瓶颈主要不是指令条数，而是 LDS 延迟/占用率。

### 20.2 Flash attention：Br 8→16（拒绝，端到端更慢）

`get_fa_tuning_params_scalar` 对 AMD RDNA2（hsk=hsv=128、n_rows>=64）给出 Br=8。把它调到 16 后：

| 指标 | Br=8 | Br=16 |
|---|---|---|
| FA 单次（perf logger，跨会话各测一次） | 38.24 ms | 35.56 ms |
| 与 CPU FA 参考的最大绝对误差（L=4120） | 3.35e-05 | 3.35e-05（H=1）/ 4.35e-05（H=24） |
| 同 prompt 端到端图像 sha256 | d641c169… | d641c169…（相同，逐位一致） |
| 端到端采样，**交叉 A/B**（两个二进制轮换，各 2 次） | **39.89 / 39.93 s** | 41.58 / 41.58 s |

结论：Br=16 数值等价，但**端到端慢 4.1 %**（两次交叉一致）。perf-logger 的“FA −7 %”是跨会话
时钟漂移造成的假象——两条单算子数据不在同一时段，交叉 A/B 才是可靠判据。已完全回退
（含为 Br=16 放宽的 occupancy 限制器条件）。Br=32 会塌陷（554 ms/次），也不可用。

### 20.3 逐元素：situ 是最大单体，但需要新算子（未实现）

按 §19 的分解把逐元素项归属到具体表达式：

| 表达式 | 构成 | 每步耗时 |
|---|---|---|
| situ（`4·tanh(g/4)·σ(g) · 25·tanh(h/25)`） | 88 TANH(68 ms) + 44 SIGMOID(33) + 176 SCALE(138) + 88 MUL(67) | **≈306 ms（6.1 %）** |
| 调制 `y + y·p1 + p0` | 84 ADD + 84 MUL（每块 2 处） | ≈95 ms（1.9 %） |
| q/k RMSNorm（128 宽行，84 次） | 84 × 1.99 ms | 167 ms（3.3 %） |

- 融合 situ 成单趟内核：读 g+h（270 MB）+ 写 135 MB ≈ 1.05 ms/块 → **约 44 ms/步**，
  即每步可省 ≈260 ms（≈5 %）。需要新增 `GGML_OP_SITU`（枚举、CPU 实现、Vulkan shader+dispatch、
  RPC op 计数断言），属架构级改动，本轮未做。
- 调制项可把 `1+p1` 每步预计算一次，省 84 个 ADD ≈ 31 ms（0.6 %），但改模型代码且收益小。
- q/k RMSNorm 行宽仅 128，84 次调用共 167 ms，带宽利用率约 51 GB/s，属小行开销问题，
  同样需要专用内核。

### 20.4 GPU 计时外的 7 %：主要是探针开销与首步预热（无收益）

在 `MOSSImageRunner::compute` 内临时打点（已还原）：

| 项 | 每步 |
|---|---|
| 建图（`new_graph_custom` + 42 块 forward + build_forward_expand） | **0.3 ms** |
| `GGMLRunner::compute` 全耗时 | 4.82 s |
| 采样循环 | 5.16 s（进度条从第 3 步起稳定显示 4.82 s/it） |

即建图可忽略；采样循环比 compute 多出的 ~2.7 s 全部集中在首步（pipeline 编译、图切分计划、PE 缓存）。
§19 里"每步 0.40 s 在 GPU 之外"主要是 `GGML_VK_PERF_LOGGER` 自身的 timestamp/同步开销
（同一内核：开 logger 5.02 s/步 vs 不开 4.82 s/步 ≈ 4 %），没有可回收的 CPU 瓶颈。
