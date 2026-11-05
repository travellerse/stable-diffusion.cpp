# INT8 QK 原型（P1）阶段记录

日期：2026-10-30 · 设备：RX 6700 XT（gfx1031 / RDNA2 / RADV）

本阶段只做**独立实验**：不改默认 Attention、不合并任何近似路径。所有测量都是 isolate 级别。

## 1. ALU 吞吐前提（dp4a vs dot2）

`alu_bench.comp` + `alu_bench.cpp`：纯寄存器、8 条独立累加链、无内存读写，只比较两种内层指令。

| 变体 | 中位 | 最优 | MAC 率 | 相当于 |
|---|---|---|---|---|
| fp16 `v_dot2_f32_f16` | 4.31–4.41 ms | 3.60 ms | 8.8 TMAC/s | 17.6 TFLOP/s（理论 25 的 70 %） |
| int8 `v_dot4_i32_i8`（dp4a） | 3.05–3.69 ms | 2.68 ms | 16.2 TMAC/s | 32.4 TFLOP/s（理论 50 的 65 %） |

**结论：dp4a 的 MAC 吞吐是 dot2 的 1.8×**，不是 2×，但确实存在可观余量。

## 2. QK 微内核（FA 的真实访问模式）

`qk_micro.comp` + `qk_micro.cpp`：从 `flash_attn.comp` 抽出 QK 段，保持 Br=8 / Bc=64 /
D_split=8 / 128 线程 / wave32、Q 走 shared、K 走 global、D_split 维 shuffle 归约；
去掉 softmax 与 PV。真实数据：layer 0、head 0、1440×1920、L=10812（Q/K 均按 FA 的方式转 f16；
int8 变体用 per-row 对称量化）。

| 变体 | 中位（3 进程 × 30 轮） | 最优 |
|---|---|---|
| fp16（dot2） | 4.12–4.41 ms | 3.59 ms |
| int8（dp4a） | 3.05–3.69 ms | 2.68 ms |
| **比值** | **≈1.4×** | **≈1.34×** |

数值核对（workgroup 0 的 8×KV 分数 vs numpy 参考）：

| 对比 | max_abs | rel |
|---|---|---|
| f16 kernel vs f16 参考 | 4.46e-03 | 8.5e-04 |
| int8 kernel vs int8 参考（同 per-row scale） | 2.53e-02 | 2.2e-03 |
| int8 kernel vs f16 参考 | 5.66e-02 | **1.33e-02** |

最后一行与第一阶段离线结论（per-row INT8 输出相对误差 1.2 %）一致，说明微内核的量化语义正确。

## 3. 收益外推（尚未实测）

- 单次 FA：QK 段 24 heads × 4.3 ms ≈ 103 ms（f16）→ 24 × 3.05 ≈ 73 ms（int8），节省 ≈30 ms；
  当前 FA 实测 ≈190 ms → 预计 ≈160 ms（−16 %），端到端 ≈ −7 %。
- 量化前处理：Q/K 各一次读写（≈200 MB/step）≈0.6 ms/step，可忽略（尚未实测）。
- 未计入：PV 段不变、softmax 不变、int8 路径的寄存器/占用率变化。

## 4. 尚未完成（P1 的剩余部分）

1. 把 int8 QK 做成**独立 FA 管线**（新 .comp + 额外 scale 绑定 + env 开关），在 `attn_check` 上做
   isolated FA 计时（≥10 轮对照）与 CPU 参考校验。
2. 若 isolated FA 有明显收益，再做完整模型的质量/性能 A/B（需要 P0 的确定性基线）。
3. 质量门槛：先给 Pareto 曲线，不预设可接受损失。

## 5. 判定

- **前提成立**：dp4a 在本卡上有 1.8× 的 MAC 吞吐，FA 的 QK 段在真实访问模式下实测快 1.3–1.4×。
- **尚未证明**：完整 FA 的端到端收益（softmax/PV/量化前处理/占用率未计入）。
- 下一步按 §4.1 继续做独立 FA 管线；在拿到 isolated FA 数据前不进入完整模型测试。

---

## 6. P1-B：独立 INT8 QK FA 管线（已实现，**未通过验证**）

### 6.1 已落地的组件（全部默认关闭）

| 组件 | 文件 | 说明 |
|---|---|---|
| INT8 QK FA 着色器 | `vulkan-shaders/flash_attn_int8qk.comp` | 固定配置 D=128/Br=8/Bc=64/Dsplit=8/128 线程/shmem 归约；Q/K 为 per-row INT8（4 值/word），`S_ij = dot8(Q_i,K_j)·sQ_i·sK_j·scale`，softmax/PV/归约/回写与 `flash_attn.comp` 保持一致（f16 PV、f32 softmax） |
| 量化内核 | `vulkan-shaders/quantize_qk_i8.comp` | 每个 (head,row) 一个 workgroup，128 线程求 amax → 对称 INT8 + scale；另编译 `DATA_F16` 变体用于 f16 的 K |
| 管线注册 | `vulkan-shaders-gen.cpp`、`ggml-vulkan-types.h`、`ggml-vulkan-push-constants.h` | 新增 3 条 pipeline 与 2 个 push-constant 结构 |
| 派发 | `ggml-vulkan.cpp` `ggml_vk_flash_attn` | `GGML_VK_FA_INT8QK=1` 时走独立分支：量化 Q/K → 同步 → int8 FA；条件为 mask/sink 为空、HSK=HSV=128、q F32、k/v F16、batch=1、gqa_ratio=1、n_rows≥256、heads≥16、张量连续 |
| 布局自适应 | 同上 | MOSS 的 q/k/v/dst 是 `[d, head, L]`，通用 FA 是 `[d, L, head]`；按 `q->ne[1] == heads` 判定并选择对应 stride |

### 6.2 当前状态：**已修复（2026-10-31），算子级正确性通过**

> 根因：`vk_subbuffer.offset` 已在 descriptor 中生效，原型又在 push constant 里重复加了
> `a_base`/`v_base`，导致 K/V 读到错误内存。修复后 `attn_check` rms 8.16e-06（f16 为 5.94e-06），
> 真实数据 int8 vs f16 rel 0.198 %，输出覆盖率 100 %。性能：FA 185.9→154.5 ms（1.20×），
> 端到端 1024² −4.5 %、1440×1920 −8.3 %。详见 [INT8_QK_DISPATCH_DEBUG.md](INT8_QK_DISPATCH_DEBUG.md) §8。

> 2026-10-31 更新：dispatch 契约、push constant ABI、buffer 绑定与 stride 语义已逐项审计完毕
> （见 [INT8_QK_DISPATCH_DEBUG.md](INT8_QK_DISPATCH_DEBUG.md)）。用"main 末尾无条件 marker"证明
> 写回覆盖率是 **0 %**（不是此前误判的 1.09 %），即内核根本没有执行写回；根因在 shader 执行/绑定层，
> 不在索引数学。

- 用真实 Q/K/V（layer 0、L=10812）对比 int8 与 f16 的 FA 输出：int8 输出恒为 0（`rel = 1.000`，
  rmse 恰等于参考 rms）。
- sentinel 测试（把写回值固定为 1.0）：只有约 1.09 % 的元素被写为 1，且集中在缓冲区起始的一段连续
  vec4 上 —— 说明写回地址/派发范围仍然不对（stride 或 binding 接线），而不是数值问题。
- 因此**集成的 123 ms vs 181 ms 计时不作数**（很可能少做了工作）。
- 已确认默认路径不受影响：`GGML_VK_FA_INT8QK` 未设置时，1024² 8 步运行与 P0 基线**逐位一致**。

### 6.3 下一步（未开始）

1. 定位写回 bug（sentinel 已给出可复现的最小实验；优先检查 `dst` 的 subbuffer range 与
   `gl_WorkGroupID.y` 到 head 的映射）。
2. 修好后按 §4.5 的门槛验收（CPU 参考、≥10 轮交错、显存、默认路径不变）。
3. 只有 isolated FA 有确定收益才进入整模型质量/性能 A/B。

### 6.4 仍然成立的有效数据（与本节的失败无关）

- ALU：dp4a 是 dot2 的 **1.8×**（16.2 vs 8.8 TMAC/s）。
- QK 微内核（真实数据、FA 真实访问模式）：**1.3–1.4×**；数值与 numpy 参考一致（int8 vs f16 参考 1.33 %）。

---

## 7. P1-C：质量 / 性能 Pareto（2026-10-31）

详见 [INT8_QK_QUALITY_PARETO.md](INT8_QK_QUALITY_PARETO.md)。

- 门禁：L=10812 与 L=4138 各 1000 次调用逐位一致、覆盖率 100 %、无 NaN/Inf；CPU 参考 rms 8.16e-06。
- 性能（10 对交错）：1024² 38.48→36.64 s（−4.79 %）；1440×1920 132.85→121.75 s（−8.35 %）。
- 确定性：FP16 A/A 与 INT8 A/A 各自逐位一致；FP16 vs INT8 固定 24.59 dB / SSIM 0.944（1024²）。
- Teacher-forced：同一输入下 attention 输出相对误差 L0 0.16–0.19 %、L20 0.45–0.67 %、L41 0.57–0.88 %。
- 多 seed/3 类 prompt（1024²，15 对）：PSNR 均值 22.50 dB（13.33–32.02），文字类 prompt 差异最大；
  1440×1920（3 对）27.63 dB。LPIPS/CLIP 本机不可用，未测。
- 建议（未实施）：若采用，按分辨率/形状启用（1440×1920 及以上更划算）。
