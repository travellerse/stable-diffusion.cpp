# MOSS-Image autotune — final state

Updated: 2026-10-29 06:10 local. Full report: `REPORT.md` (§13 = INT8 GEMM round).

## Best configuration (verified)

`--flash-attn diffusion=on`, plus the rewritten INT8 GEMM kernel in `ggml/` (see below).

| resolution | before (auto + old kernel) | after (flash on + new kernel) |
|---|---|---|
| 512×512 | 32.2 s | **16.3 s** |
| 1024×1024 | 124.0 s | **55.1 s** |
| 1328×752 | 114.7 s | **51.3 s** |
| 1632×912 | 183.1 s | **77.5 s** |
| 1408×1408 (2 MP) | not runnable (auto) | **119.8 s** |
| 1152×1728 (2 MP) | not runnable (auto) | **122.7 s** |
| 1888×1056 (2 MP) | 250.4 s | **120.2 s** |
| 1440×1920 (recipe default) | 385.4 s | **192.8 s** |

VRAM unchanged within run noise; minimum free VRAM 2.28 GB at 2 MP (above the 1.5 GiB gate).

## INT8 GEMM kernel (ggml/, uncommitted)

- `mul_mat_i8_tensorwise.comp`: 16×16 tile / 1 accumulator / 2 barriers per 32 int8
  → **64×32 tile / 8 accumulators per thread / 2 barriers per 64 int8**.
- `ggml-vulkan.cpp`: pipeline `wg_denoms` `{16,16,1}` → `{64,32,1}` (1 line).
- Microbenchmark: 58.14→18.55 ms (qkv), 103.65→32.61 (upgate), 52.22→17.19 (down) = **3.0–3.2×**;
  3.97 → 12.5 TFLOP/s (4 % → 24 % of dp4a peak).
- Bit-exact vs old kernel (FNV-1a equal, max_abs 0), `test-int8-convrot --vulkan` passes.
- Rejected variants: 64×64/16 acc (22.7 ms), KC=16 (20.0 ms), 128×32/16 acc (25.0 ms).

## Measurement caveats

- Same-seed reruns are not bit-identical (PSNR 18.6–29.3 dB between runs of the same config);
  cause is suspected auto-fit plan variation, not the kernel (kernel proven bit-exact).
- 20 ms sysfs VRAM sampling still yields lower bounds; peaks vary up to 1.6 GB run to run.

## Next candidates

1. Vectorized (`ivec4`) shared-memory weight loads and bank-conflict padding (est. +20–30 %).
2. Upstream the kernel change as a PR (needs user decision: commit inside submodule + push).

## ggml rebase onto v0.28.0 (2026-10-29)

`ggml` replayed `dev` (25 commits) onto `facdc44a` (origin/master v0.28.0). Four picks
conflicted, five conflict blocks, all resolved by ownership analysis rather than side-picking:

| pick | file | resolution |
|---|---|---|
| 9525036d add INT8 convrot | ggml-rpc.h, ggml.c ×2 | `GGML_OP_COUNT == 104` (measured), RPC patch 1→2 |
| 9525036d | ggml-cuda.cu | keep upstream `TOP_K_MASK` + our `REGULAR_HADAMARD`/`QUANTIZE_I8_CONVROT` cases |
| 2d27a714 drop Hadamard | ggml-rpc.h, ggml.c ×2 | `103` (measured) |
| 2d27a714 | ggml-cuda.cu | keep upstream `TOP_K_MASK` + our `QUANTIZE` case, delete the `HADAMARD` case |
| c3a676c sol attention | ggml-rpc.h | `105` (measured; the pick's own 104 was stale) |
| 6a0eafd / e7dabd4 CUDA fusions | ggml-cuda.cu | union: our dependency registrations first, then upstream's GLU block |

**Silent auto-merge defect found and fixed**: the `ROPE_APPLY` pick's assert delta (104→105)
landed on a tree whose count was already 105, so the tip asserted 105 while the enum had 106.
Added `a3a2ecb7 ggml : fix GGML_OP_COUNT asserts after rebase onto v0.28.0`.
Rule adopted: `GGML_OP_COUNT` is always taken from a compiled probe of the merged enum, never
from either conflict side.

Verification: 67/69 touched files have byte-identical old-vs-new patch deltas (the 2 exceptions
are the op-count asserts themselves); build OK; `test-int8-convrot --vulkan` passes; kernel
microbench output FNV-1a identical to pre-rebase (`cec6e814…` / `142a8989…` / `7be47525…`) with
14.6/25.9/13.9 ms; end-to-end 1024² = 51.1 s (pre-rebase 51.5 s).
Rollback: `git -C ggml reset --hard 3c846059` (old tip, still reachable).

## Preview TAE backend placeholder (2026-10-30)

`--backend tae-preview=cpu` now moves only the step-preview TAE to the CPU; DiT/TE/VAE are
untouched and an unset assignment still follows `vae` (default behaviour unchanged).

- 1024², 8 steps, `--preview tae --preview-interval 1`: sampling 48.6 s → 96.0 s (≈5.9 s per
  preview), peak VRAM unchanged vs no preview, min free 1.91–1.92 GB.
- Preview TAE on the GPU at 1024² trips the 1.5 GiB safety gate with a ~3 GB desktop footprint
  (free dropped to 1.32 GB); at 768² it costs +0.50 GB peak and 0.17 s per preview.
- `--preview proj` is free (sampling 47.65 s) but renders a 128² colour projection.
- Old `--backend vae=cpu` also moves the final decode: generate 142.4 s vs 103.0/100.3 s.

## Preview TAE: im2col is the cost (2026-10-30, supersedes the CPU-offload advice)

Keep the preview TAE on the GPU and add `--conv-direct vae=on` (the preview module inherits it):
1024² preview costs 0.19–0.41 s per step (vs ~1 GB of im2col scratch and a safety abort on the
default path), final VAE decode 3.35 → 1.59 s, peak unchanged. 2 MP: 0.5–0.6 s per preview,
decode 5.59 → 2.89 s. `--backend tae-preview=cpu` is only for VRAM-constrained runs (5.9 s per
preview at 1024², unaffected by `--conv-direct`). The Vulkan conv probe benchmarks only 32×32 and
16×16 shapes, which is why `auto` misses this; changing the probe needs a separate evaluation.
