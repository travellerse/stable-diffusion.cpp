# INT8 QK FA：Dispatch 契约审计与写回覆盖率诊断

日期：2026-10-31 · 设备：RX 6700 XT（gfx1031 / RDNA2 / RADV）· 默认路径未改动

本轮只做正确性定位，不做性能。所有 INT8 FA 计时（123/129/140 ms 等）作废，不得引用。

## 1. 原始 FP16 FA 的执行契约（从代码提取）

| 项目 | 原 FA |
|---|---|
| Pipeline 创建 | `ggml_vk_create_pipeline(..., parameter_count = 8, push_constant_size = sizeof(vk_flash_attn_push_constants), wg_denoms = {Br,1,1}, spec_constants = get_fa_spec_constants(), ...)` |
| Local size | `layout(local_size_x_id = 0)`，由 spec constant `WorkGroupSize` 决定；本形状为 128 |
| Dispatch 元素 | `{ workgroups_x = neq1, workgroups_y = neq2, workgroups_z = neq3 }` |
| 实际 workgroup 数 | 由 `ggml_vk_dispatch_pipeline` 计算：`wg0 = CEIL_DIV(elements[0], wg_denoms[0])` → `CEIL_DIV(neq1, Br)` = 查询 tile 数 |
| Binding | 0=Q, 1=K, 2=V, 3=mask（无 mask 时绑 q_buf 占位）, 4=sinks, **5=dst**, 6=mask_opt, 7=sparse |
| Push constant | `vk_flash_attn_push_constants`（N, KV, ne1..ne3, neq2/3, nek2/3, nev2/3, nem1/2/3, nb01..nb23, scale, max_bias, logit_softcap, mask_n_head_log2, m0, m1, gqa_ratio, split_kv, k_num） |
| `p.ne1` 语义 | **`dst->ne[1]`**（不是 q 的行数）：MOSS 的输出 view 是 `[d, head, L]`，所以 `ne1 = 24` |
| 输出索引公式 | `o = (iq2*HSV + (i*Br+row)*p.ne1*HSV)/4 + d*D_split + d_tid`，即 `head*32 + row*(ne1*32) + dvec` |
| 输出张量（MOSS 实测） | `ne = [128, 24, 4108]`，`nb = (4, 512, 12288, 50479104)` → 行距 768 vec4、head 距 32 vec4 |
| 输出 buffer | subbuffer offset 664,289,280、range 132,857,856 B（完整张量） |

## 2. INT8 FA 的契约（当前实现）

| 项目 | INT8 FA | 与原 FA 一致 |
|---|---|---|
| Local size | 128（写死） | 是 |
| Dispatch 元素 | `{ Tr, heads, 1 }` = `{1352, 24, 1}`（实测 trace） | 是（原 FA 的 elements 是 `{neq1, neq2, 1}`，再除以 Br） |
| wg_denoms | `{1,1,1}`，因此 wg0 = 1352 | 等价 |
| Binding | 0=Q8, 1=K8, 2=V, 3=scales, 4=dst（5 个） | 否（独立布局，参数数量 5 vs 8） |
| Push constant | 自建 `vk_fa_int8qk_push_constants`：12×uint + float = **52 B** | 布局经 SPIR-V 反汇编核对：`OpTypeStruct` = 12 uint + 1 float，无 padding 偏移 |
| 输出索引 | `o = head*o_head_stride + row*o_row_stride + dvec`，stride 由 **dst 自身** `ne[1]==heads` 判定后取 `nb[1]/16`、`nb[2]/16` | 是（MOSS: 32 / 768；harness `[d,L,h]`: L*32 / 32） |
| 输出 buffer | 与前一次 trace 一致（offset 664,289,280，range 132,857,856） | 是 |
| Q/K 量化缓冲 | q8、k8 各 24×10812×128 B（offset 0），scales 24×(N+KV)×4 B | 新增 |

## 3. 关键证据：写回覆盖率为 0（不是 1.09 %）

先用固定浮点 sentinel 得到的 "1.09 %" 是**误导性**结果（部分写入被后续写覆盖）。本轮改用**唯一值编码**：

在 `main()` 末尾（所有正常写回之后）对 `WG(0,0,0)/lane0` 无条件写 `data_d[0] = 123.0`：

- 结果：输出仍然**全零**，`data_d[0]` 也没有变成 123。
- 结论：**内核根本没有执行写回**（或完全没有被派发到 GPU）。任何关于 stride/索引的推测都不是当前瓶颈。

## 4. 已排除项（本轮）

| 假设 | 证据 | 结论 |
|---|---|---|
| Dispatch 数量错误 | trace：`wg=(1352,24,1)`，`elems=(1352,24,1)`，`denoms=(1,1,1)` | 正确 |
| Push constant 大小/布局 | SPIR-V `OpTypeStruct` = 12 uint + 1 float；C++ `sizeof` = 52；dispatch 断言通过 | 正确 |
| Buffer binding 数量 | trace：5 个 buffer，`parameter_count = 5` 断言通过 | 正确 |
| dst buffer/offset/range | trace：offset 664,289,280、range 132,857,856 | 与张量匹配 |
| 输出 stride 语义 | 原 FA 用 `dst->ne[1]`（=heads）作行距；INT8 版改为由 dst 自身维度推导 | 已修正（但不是本次根因） |
| D_split 归约缺失 | 已补 `subgroupShuffleXor` 归约 | 已修正（但不是本次根因） |
| ggml 默认路径受影响 | 用 P0 二进制（无 INT8 代码）与当前二进制各跑一次 1024²/8 步：**逐位一致** | 默认路径未被改动 |

## 5. 剩余嫌疑（下一轮按顺序验证）

1. **Pipeline 是否真正创建成功**：在 `ggml_vk_load_shaders` 后打印
   `device->pipeline_fa_int8qk->initialized`、`parameter_count`、`push_constant_size`；
   若是 false，则为创建失败（此时代码仍会进入分支并"派发"）。
2. **Descriptor set 是否写入**：在 `ggml_vk_dispatch_pipeline` 内对 int8qk 打印
   `descriptor_set_idx` 与 `ctx->descriptor_sets.size()`，并在 `vkUpdateDescriptorSets` 后读回校验。
3. **独立 harness 验证 shader 本身**：把 `/tmp/fa_i8.spv` 放进 `qk_micro.cpp` 的最小 Vulkan 程序里
   （单 binding、单 dispatch、已知 buffer），确认 shader 在 ggml 之外能写出数据。若不能，问题在 shader/
   SPIR-V；若能，问题在 ggml 的绑定或提交路径。
4. **提交路径**：确认我的分支没有因为缺少 `ggml_vk_sync_buffers`/`prealloc_*_need_sync` 之类的状态
   标记而导致所属 command buffer 未被提交（对照 split_k 分支与普通分支的收尾动作）。

## 6. 本轮附带确认的事实

- **1024² 的残余不确定性是"会话/构建稳定"的**：同一二进制连跑 2 次逐位一致；P0 旧二进制与当前
  二进制也逐位一致；但都不同于更早一次会话的基线。说明它不是每跑一次就随机，而是随构建/驱动缓存
  状态一次性改变 —— 这与 P0 报告中"1/6 次分歧"的记录合并起来，指向分配布局敏感的非确定性。
- 默认路径在 `GGML_VK_FA_INT8QK` 未设置时与 P0 二进制**逐位相同**，INT8 代码未污染正式路径。

## 7. 结论

- INT8 FA 当前**不可用于任何性能或质量评估**（写回覆盖率为 0）。
- 已确认 dispatch 契约、push constant ABI、buffer 绑定参数与 stride 语义均已与原 FA 对齐；根因位于
  "shader 是否执行/写入"这一层，下一步按 §5 的 1→4 顺序排查。
- 默认路径保持正确且未被改动。

---

## 8. 第二轮：执行链路审计结果（2026-10-31）

### 8.1 Pipeline 生命周期

| 检查 | 结果 |
|---|---|
| `ggml_vk_create_pipeline` 记录的元数据 | `initialized=1, parameter_count=5, push_constant_size=52, wg_denoms=(1,1,1)` |
| 创建时 `VkPipeline` / `VkPipelineLayout` | **均为 nil** —— ggml 的 pipeline 是**惰性编译**，只有被 `ggml_pipeline_request_descriptor_sets` 请求时才真正创建 |
| dispatch 时句柄 | **非空**（`pipeline=0x556a9ac4e630`，name=`flash_attn_int8qk`），且是本次派发实际绑定的对象 |
| 嵌入 SPIR-V 与源码同步 | 生成物 `flash_attn_int8qk_fp32_len = 30496`；`spirv-val`/`spirv-dis` 检查通过；push constant `OpTypeStruct` = 12×uint + 1×float（52 B），与 C++ 结构逐字段一致 |
| GPU 实际执行证据 | `GGML_VK_PERF_LOGGER` 显示 DiT 的 FA 节点执行 42 次（`dst(128,24,4108) q(128,4108,24)`），单次 ~20.3 ms |

结论：pipeline 创建、请求、绑定、执行全部正常。问题不在"是否执行"。

### 8.2 三阶段诊断（改为写回之后输出，避免被正常写回覆盖）

在 `main()` 末尾（所有写回之后）由 `WG(0,0,0)/lane0` 写出 KV / N / Lf / Mf / Of / 首个 q scale / 首个 k scale：

| 阶段 | 修复前 | 修复后 |
|---|---|---|
| KV / N | 10812 / 10812（push constant 正确） | 同左 |
| q amax（量化器探针） | 5.685（Q 读取正常） | 5.685 |
| **k amax（量化器探针）** | **0 ← K 读到全零** | **4.25** |
| Lf[0] | 1351.69（= KV·e^-2.079，全 0 分数导致均匀 softmax） | 0.125 |
| Of[0][0] | **0** | -0.593 |
| 输出全零比例 | 100 % | 0 % |

### 8.3 根因：subbuffer offset 被重复应用

`ggml_vk_tensor_subbuffer()` 返回的 `vk_subbuffer.offset` **已经写进 `VkDescriptorBufferInfo.offset`**，shader 里的 `data_a[0]` 就是张量的第一个元素。而原型在 push constant 里又传了一次 `a_base = buf.offset / elem_size`、`v_base = v_buf.offset / 8`，导致指针二次偏移：

- Q（f32）二次偏移后仍落在同一大 buffer 的其他数据上 → amax 看起来"正常"（其实是别的数据）；
- K（f16，offset 81,558,016 B）二次偏移后落到未初始化区域 → amax=0；
- V 同理 → PV 累加为 0。

**修复**：`a_base`、`v_base` 一律置 0（descriptor 已带偏移）。改动仅 3 行，不涉及其它数学。

### 8.4 修复后的验证

| 检查 | 结果 |
|---|---|
| `attn_check` vs CPU 参考（L=10842, H=24） | int8 `rms 8.159e-06 / max_abs 7.99e-05`；f16 为 `5.941e-06 / 5.42e-05` → int8 误差仅为 f16 的 1.37× |
| 真实 Q/K/V 对比 f16 FA（L=10812） | `max 7.42e-02, rms 1.19e-03, rel 0.198 %`，输出无零 |
| 输出覆盖率 | 100 %（zero_frac = 0） |
| 默认路径隔离 | `GGML_VK_FA_INT8QK` 未设置时，`[i8qk]` 分支从不进入（调试计数 = 0） |

### 8.5 性能（修复后，真实值）

| 场景 | f16 FA | int8 QK FA | 加速 |
|---|---|---|---|
| L=10842, H=24（isolated） | 185.9 ms | **154.5 ms** | **1.20×** |
| L=4120, H=24（isolated） | 31.4 ms | **26.7 ms** | **1.17×** |
| 1024² 端到端 8 步采样 | ~38.7 s | **37.0 s** | −4.5 % |
| 1440×1920 端到端 8 步采样 | 133.2 s | **122.2 s** | −8.3 % |

此前的 123/129/140 ms 是错误路径（少读 K/V）的产物，已作废。

### 8.6 仍开放

1. **图像级质量**：int8 与 f16 的 latent 在 step 1 就有 ~26 % 差异（混沌放大），不能用 latent 差判定质量；需要多 seed 的 Pareto 曲线。
2. **1024² 的 A/A 残余不确定性**：本轮观察到同一二进制两次运行在 `pred` 第 1 步即不同（此前 3/3 一致），说明 P0 记录的残余问题是**逐次随机**而非构建相关，仍待定位。
3. 量化前处理的显存/带宽未单独计入性能表（已包含在上述 FA 计时内）。
