#!/usr/bin/env python3
"""Summarize /tmp/moss-autotune/records.jsonl into per-label statistics."""
import collections
import json
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moss-autotune/records.jsonl"
recs = [json.loads(l) for l in open(path)]

groups = collections.defaultdict(list)
for r in recs:
    label = r["label"]
    base = label.rsplit("-r", 1)[0] if "-r" in label else label
    groups[base].append(r)

print(f"{'variant':22s} {'n':>2s} {'wall_s med(min..max)':>26s} {'samp_s med':>11s} "
      f"{'te_s':>6s} {'vae_s':>6s} {'vram_GB med':>11s} {'minfree_GB':>10s} {'busy%':>6s} {'rc':>3s}")
for base, rs in sorted(groups.items()):
    ok = [r for r in rs if r["returncode"] == 0 and r["out_bytes"] > 0]
    walls = sorted(r["wall_s"] for r in ok)
    samp = sorted(r["stages"].get("sampling_s", 0) for r in ok)
    te = [r["stages"].get("te_ms", 0) for r in ok]
    vae = [r["stages"].get("vae_decode_s", 0) for r in ok]
    vrams = [r["vram_peak"] / 1e9 for r in ok if r.get("vram_peak")]
    free = [r["vram_min_free"] / 1e9 for r in ok if r.get("vram_min_free")]
    busy = [r.get("gpu_busy_mean") for r in ok if r.get("gpu_busy_mean")]
    wall_s = f"{statistics.median(walls):.1f} ({min(walls):.1f}..{max(walls):.1f})" if walls else "-"
    print(f"{base:22s} {len(ok):2d} {wall_s:>26s} "
          f"{(f'{statistics.median(samp):.1f}' if samp else '-'):>11s} "
          f"{(f'{statistics.median(te):.2f}' if te else '-'):>6s} "
          f"{(f'{statistics.median(vae):.2f}' if vae else '-'):>6s} "
          f"{(f'{statistics.median(vrams):.2f}' if vrams else '-'):>11s} "
          f"{(f'{min(free):.2f}' if free else '-'):>10s} "
          f"{(f'{statistics.median(busy):.0f}' if busy else '-'):>6s} "
          f"{rs[-1]['returncode']:3d}")
