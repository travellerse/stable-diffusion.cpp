#!/usr/bin/env python3
"""Block-sparsity statistics on real MOSS-Image attention.

Streaming: attention logits are produced one query chunk at a time; the full
Lq x Lk matrix is never materialised.

Reports, per head:
  * softmax mass coverage of the top KV blocks (50/75/90/95/99 %)
  * routing quality of proxy scores (SpargeAttn mean-similarity, Sol-Attn
    mean+max variants) versus the exact block mass ranking
  * output error when only the top-k blocks are computed:
      - drop    : skipped blocks contribute nothing (renormalised inside)
      - meancor : skipped blocks contribute their exact mass times the block
                  mean of V (the FlashPrefill V2 correction, covariance-only error)

usage: block_sparsity.py DUMP_DIR LAYER [--heads 0,1] [--qchunk 256] [--maxq 0]
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
            return np.fromfile(path, dtype=dt).reshape(r["shape"])
    raise SystemExit(f"missing {tag} in {dump_dir}")


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("layer", type=int)
    ap.add_argument("--heads", default="0")
    ap.add_argument("--qchunk", type=int, default=256)
    ap.add_argument("--maxq", type=int, default=0)
    ap.add_argument("--call", type=int, default=0)
    ap.add_argument("--bc", default="32,64,128")
    args = ap.parse_args()

    q_t = find(args.dump_dir, f"dbg_q_l{args.layer}", args.call or None).astype(np.float32)
    k_t = find(args.dump_dir, f"dbg_k_l{args.layer}", args.call or None).astype(np.float32)
    v_t = find(args.dump_dir, f"dbg_v_l{args.layer}", args.call or None).astype(np.float32)
    k_t = k_t.astype(np.float16).astype(np.float32)
    v_t = v_t.astype(np.float16).astype(np.float32)

    d_head, Lq, n_head = q_t.shape[0], q_t.shape[1], q_t.shape[2]
    Lk = k_t.shape[1]
    scale = np.float32(1.0 / np.sqrt(float(d_head)))
    heads = [int(h) for h in args.heads.split(",") if h != ""]
    q_limit = args.maxq if args.maxq else Lq

    cover_targets = [0.50, 0.75, 0.90, 0.95, 0.99]
    for bc in [int(b) for b in args.bc.split(",")]:
        n_blk = (Lk + bc - 1) // bc
        # exact coverage: fraction of blocks needed to reach target mass
        need = {t: [] for t in cover_targets}
        proxy_hit = {0.1: [], 0.2: [], 0.3: [], 0.5: []}
        drop_err = {0.1: [], 0.2: [], 0.3: [], 0.5: []}
        corr_err = {0.1: [], 0.2: [], 0.3: [], 0.5: []}
        dens_err = []

        for h in heads:
            q = np.ascontiguousarray(q_t[:, :q_limit, h, 0])
            k = np.ascontiguousarray(k_t[:, :, h, 0])
            v = np.ascontiguousarray(v_t[:, h, :, 0])
            # block means for the proxy scores and the mean correction
            kblk = np.zeros((d_head, n_blk), dtype=np.float32)
            vblk = np.zeros((d_head, n_blk), dtype=np.float32)
            for b in range(n_blk):
                s, e = b * bc, min((b + 1) * bc, Lk)
                kblk[:, b] = k[:, s:e].mean(axis=1)
                vblk[:, b] = v[:, s:e].mean(axis=1)

            for start in range(0, q_limit, args.qchunk):
                end = min(start + args.qchunk, q_limit)
                qc = q[:, start:end]
                logits = (qc.T @ k) * scale
                p = softmax(logits)
                dense = v @ p.T

                # exact block mass
                mass = np.zeros((end - start, n_blk), dtype=np.float64)
                for b in range(n_blk):
                    s, e = b * bc, min((b + 1) * bc, Lk)
                    mass[:, b] = p[:, s:e].sum(axis=1)
                order = np.argsort(-mass, axis=1)
                csum = np.cumsum(np.take_along_axis(mass, order, axis=1), axis=1)
                for t in cover_targets:
                    need[t].append(np.mean((csum < t).sum(axis=1) + 1) / n_blk)

                # proxy: mean-similarity (pooled Q dot pooled K)
                qmean = qc.mean(axis=1)
                proxy = np.tile((kblk.T @ qmean) * scale, (end - start, 1))
                porder = np.argsort(-proxy, axis=1)

                for frac in (0.1, 0.2, 0.3, 0.5):
                    kk = max(1, int(round(frac * n_blk)))
                    exact_sel = order[:, :kk]
                    proxy_sel = porder[:, :kk]
                    hit = np.mean([len(set(a) & set(b)) / kk for a, b in zip(exact_sel, proxy_sel)])
                    proxy_hit[frac].append(hit)

                    # sparse output with the *proxy* selection (what a real
                    # implementation would use), renormalise inside the kept mass
                    out_drop = np.zeros_like(dense)
                    out_corr = np.zeros_like(dense)
                    for row in range(end - start):
                        sel = proxy_sel[row]
                        cols = np.concatenate([np.arange(b * bc, min((b + 1) * bc, Lk)) for b in sel])
                        pr = p[row, cols]
                        norm = pr.sum()
                        if norm <= 0:
                            continue
                        out_drop[:, row] = (v[:, cols] @ pr) / norm
                        kept = v[:, cols] @ pr
                        skipped_mass = 1.0 - norm
                        skipped_mean = vblk[:, sel].mean(axis=1)
                        out_corr[:, row] = kept + skipped_mass * skipped_mean
                    drop_err[frac].append(np.sqrt(np.mean((out_drop - dense) ** 2)) /
                                          (np.sqrt(np.mean(dense ** 2)) + 1e-12))
                    corr_err[frac].append(np.sqrt(np.mean((out_corr - dense) ** 2)) /
                                          (np.sqrt(np.mean(dense ** 2)) + 1e-12))
                dens_err.append(np.sqrt(np.mean(dense ** 2)))

        print(f"\n=== Bc={bc} (n_blocks={n_blk}) layer={args.layer} heads={heads} Lq={q_limit} Lk={Lk}")
        print("  coverage (fraction of blocks needed):")
        for t in cover_targets:
            v = np.mean(need[t])
            print(f"    {int(t*100):3d}% mass -> {v*100:5.1f}% of blocks")
        print(f"  {'keep':>6s} {'proxy_topk_recall':>18s} {'drop_rel_err':>13s} {'meancor_rel_err':>16s}")
        for frac in (0.1, 0.2, 0.3, 0.5):
            print(f"  {frac*100:5.0f}% {np.mean(proxy_hit[frac]):18.3f} "
                  f"{np.mean(drop_err[frac]):13.3e} {np.mean(corr_err[frac]):16.3e}")


if __name__ == "__main__":
    main()
