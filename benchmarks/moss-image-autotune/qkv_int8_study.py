#!/usr/bin/env python3
"""Offline INT8 QK study on real MOSS-Image Q/K/V dumps.

Inputs come from the diagnostic dump (SD_DUMP_STEPS + SD_DUMP_ATTN_LAYERS):
  dbg_q_l<L>  [d_head, Lq, n_head, 1] f32  (post-RoPE, exactly what FA consumes)
  dbg_k_l<L>  [d_head, Lk, n_head, 1] f32
  dbg_v_l<L>  [d_head, n_head, Lk, 1] f32  (forwarded through FA as f16)

Reference and every quantized variant follow the deployed computation chain:
  QK^T in f32 accumulation -> f32 softmax -> PV with f16 V, f32 accumulation.
K and V are cast to f16 first, matching `ggml_cast(k, F16)` in ggml_extend.cpp.

usage: qkv_int8_study.py DUMP_DIR LAYER [--heads 0,1,2] [--qchunk 1024] [--maxq 0]
"""
import argparse
import json
import os

import numpy as np


def find(dump_dir, tag, call=None):
    for line in open(os.path.join(dump_dir, "manifest_gpu.jsonl")):
        r = json.loads(line)
        if r["tag"] == tag and (call is None or r["call"] == call):
            path = os.path.join(dump_dir, f"{tag}_c{r['call']:04d}.bin")
            dt = np.float16 if r["type"] == "f16" else np.float32
            return np.fromfile(path, dtype=dt).reshape(r["shape"]), r
    raise SystemExit(f"missing {tag} in {dump_dir}")


def quant_per_row(x, axis_last=True):
    """Symmetric int8 per row (last axis), returns (int8 array, scale per row)."""
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale[..., 0]


def quant_per_keyblock(x, block):
    """One symmetric int8 scale per block of `block` keys, shared across d.

    x: [n_keys, d] -> (int8 q, per-key scale [n_keys]) so that a score can be
    dequantised as dot(q_i, q_j) * s_q[i] * s_k[j].
    """
    n, d = x.shape
    pad = (-n) % block
    xp = np.pad(x, ((0, pad), (0, 0))) if pad else x
    view = xp.reshape(-1, block, d)
    amax = np.max(np.abs(view), axis=(1, 2))
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    q = np.clip(np.rint(view / scale[:, None, None]), -127, 127).astype(np.int8).reshape(-1, d)
    if pad:
        q = q[:n]
    ks = scale.repeat(block)[:n]
    return q, ks


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("layer", type=int)
    ap.add_argument("--heads", default="0")
    ap.add_argument("--qchunk", type=int, default=1024)
    ap.add_argument("--maxq", type=int, default=0, help="limit query rows (0 = all)")
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    q_t, _ = find(args.dump_dir, f"dbg_q_l{args.layer}", args.call or None)
    k_t, _ = find(args.dump_dir, f"dbg_k_l{args.layer}", args.call or None)
    v_t, _ = find(args.dump_dir, f"dbg_v_l{args.layer}", args.call or None)
    q_t = q_t.astype(np.float32)
    k_t = k_t.astype(np.float32)
    v_t = v_t.astype(np.float32)

    d_head, Lq, n_head = q_t.shape[0], q_t.shape[1], q_t.shape[2]
    Lk = k_t.shape[1]
    # k/v as FA sees them: f16
    k_t = k_t.astype(np.float16).astype(np.float32)
    v_t = v_t.astype(np.float16).astype(np.float32)

    scale = np.float32(1.0 / np.sqrt(float(d_head)))
    heads = [int(h) for h in args.heads.split(",") if h != ""]
    q_limit = args.maxq if args.maxq else Lq

    variants = ["fp16_ref", "row_int8", "row_int8_kmean", "block64_int8", "block128_int8",
                "row_int8_blockmean"]
    acc = {name: {"logit_sq": 0.0, "logit_max": 0.0, "out_sq": 0.0, "out_ref_sq": 0.0,
                  "out_max": 0.0, "tv": 0.0, "n": 0} for name in variants[1:]}

    for h in heads:
        q = np.ascontiguousarray(q_t[:, :q_limit, h, 0])   # [d, Lq]
        k = np.ascontiguousarray(k_t[:, :, h, 0])          # [d, Lk]
        v = np.ascontiguousarray(v_t[:, h, :, 0])          # [d, Lk]

        # global K mean (SageAttention smoothing subtracts the per-channel K mean)
        k_mean_global = k.mean(axis=1, keepdims=True)
        kc_global = k - k_mean_global

        for start in range(0, q_limit, args.qchunk):
            end = min(start + args.qchunk, q_limit)
            qc = q[:, start:end]
            ref_logits = (qc.T @ k) * scale
            ref_p = softmax(ref_logits)
            ref_out = v @ ref_p.T

            def report(name, logits):
                p = softmax(logits)
                out = v @ p.T
                dlog = np.abs(logits - ref_logits)
                dout = np.abs(out - ref_out)
                a = acc[name]
                a["logit_sq"] += float(np.sum((logits - ref_logits) ** 2))
                a["logit_max"] = max(a["logit_max"], float(dlog.max()))
                a["out_sq"] += float(np.sum((out - ref_out) ** 2))
                a["out_ref_sq"] += float(np.sum(ref_out ** 2))
                a["out_max"] = max(a["out_max"], float(dout.max()))
                a["tv"] += float(np.sum(np.abs(p - ref_p))) / 2.0
                a["n"] += int(logits.size)

            # 1) per-row INT8 on Q and K
            qq, qs = quant_per_row(qc.T)
            kq, ks = quant_per_row(k.T)
            logits = ((qq.astype(np.int32) @ kq.astype(np.int32).T).astype(np.float32)
                      * qs[:, None] * ks[None, :]) * scale
            report("row_int8", logits)

            # 2) per-row INT8 with global K mean smoothing
            qq, qs = quant_per_row(qc.T)
            kq, ks = quant_per_row(kc_global.T)
            logits = ((qq.astype(np.int32) @ kq.astype(np.int32).T).astype(np.float32)
                      * qs[:, None] * ks[None, :]) * scale
            logits = logits + (qc.T @ k_mean_global) * scale
            report("row_int8_kmean", logits)

            # 3) one scale per block of keys (Bc = 64 / 128), shape unchanged
            for blk, name in ((64, "block64_int8"), (128, "block128_int8")):
                qq, qs = quant_per_row(qc.T)
                kq, ks = quant_per_keyblock(k.T, blk)
                logits = ((qq.astype(np.int32) @ kq.astype(np.int32).T).astype(np.float32)
                          * qs[:, None] * ks[None, :]) * scale
                report(name, logits)

            # 4) per-row INT8 with per-block K mean smoothing (Bc=64)
            blk = 64
            kc = np.empty_like(k)
            for s in range(0, Lk, blk):
                e = min(s + blk, Lk)
                kc[:, s:e] = k[:, s:e] - k[:, s:e].mean(axis=1, keepdims=True)
            qq, qs = quant_per_row(qc.T)
            kq, ks = quant_per_row(kc.T)
            logits = ((qq.astype(np.int32) @ kq.astype(np.int32).T).astype(np.float32)
                      * qs[:, None] * ks[None, :]) * scale
            report("row_int8_blockmean", logits)

    print(f"layer={args.layer} heads={heads} Lq={q_limit} Lk={Lk} d={d_head}")
    print(f"{'variant':20s} {'logit_rmse':>11s} {'logit_max':>10s} {'out_rmse':>10s} "
          f"{'out_rel':>9s} {'out_max':>9s} {'softmax_TV':>11s}")
    for name in variants[1:]:
        a = acc[name]
        n = a["n"]
        lr = np.sqrt(a["logit_sq"] / n)
        ore = np.sqrt(a["out_sq"] / max(1, a["n"]))
        orel = np.sqrt(a["out_sq"] / max(1e-12, a["out_ref_sq"]))
        print(f"{name:20s} {lr:11.3e} {a['logit_max']:10.3e} {ore:10.3e} {orel:9.3e} "
              f"{a['out_max']:9.3e} {a['tv']/max(1,len(heads)*((q_limit+args.qchunk-1)//args.qchunk)):11.3e}")


if __name__ == "__main__":
    main()
