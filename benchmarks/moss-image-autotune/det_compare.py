#!/usr/bin/env python3
"""Compare two SD_DUMP_STEPS runs step by step.

usage: det_compare.py DIR_A DIR_B [DIR_C ...]
Prints, for every dumped tensor (tag, step), the FNV hash equality and the
MAE / RMSE / max-abs / relative error against the first directory.
"""
import json
import os
import sys

import numpy as np


def load_manifest(d):
    entries = {}
    path = os.path.join(d, "manifest.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            entries[(r["tag"], r["step"])] = r
    return entries


def load_tensor(d, tag, step, shape):
    path = os.path.join(d, f"{tag}_s{step:03d}.bin")
    data = np.fromfile(path, dtype=np.float32)
    return data.reshape(shape)


def main():
    dirs = sys.argv[1:]
    if len(dirs) < 2:
        print(__doc__)
        return 1
    manifests = [load_manifest(d) for d in dirs]
    keys = sorted(manifests[0].keys(), key=lambda k: (k[0], k[1]))
    print(f"{'tag':6s} {'step':>4s} " + " ".join(f"{'hash'+str(i):>9s} {'rmse':>10s} {'maxabs':>10s} {'rel':>9s}" for i in range(1, len(dirs))))
    first_div = {}
    for tag, step in keys:
        ref_entry = manifests[0][(tag, step)]
        shape = ref_entry["shape"]
        ref = load_tensor(dirs[0], tag, step, shape)
        cols = []
        for i in range(1, len(dirs)):
            e = manifests[i][(tag, step)]
            cur = load_tensor(dirs[i], tag, step, shape)
            same = e["fnv"] == ref_entry["fnv"]
            diff = cur - ref
            rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
            mx = float(np.max(np.abs(diff)))
            denom = float(np.mean(np.abs(ref))) + 1e-12
            rel = float(np.mean(np.abs(diff))) / denom
            cols.append(f"{'same' if same else 'DIFF':>9s} {rmse:10.3e} {mx:10.3e} {rel:9.3e}")
            if not same and tag not in first_div:
                first_div[tag] = step
        print(f"{tag:6s} {step:4d} " + " ".join(cols))
    print()
    for tag, step in sorted(first_div.items()):
        print(f"first divergence: {tag} at step {step}")
    if not first_div:
        print("all dumped tensors are bit-identical across runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
