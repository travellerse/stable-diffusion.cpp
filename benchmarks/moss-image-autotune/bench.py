#!/usr/bin/env python3
"""MOSS-Image end-to-end benchmark harness.

Wraps one sd-cli process run and records wall time, stage times parsed from the
sd-cli log, peak device VRAM (sysfs sampling), peak child RSS and minimum
system MemAvailable into a JSONL record.

VRAM accounting: /sys/class/drm/card1/device/mem_info_vram_used is the total
device-wide VRAM in use, including the desktop compositor. Reported peak is
absolute device occupancy plus the delta over the pre-run idle baseline.
"""
import argparse
import glob
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time

DRM = "/sys/class/drm/card1/device"
SAMPLE_INTERVAL = 0.1  # seconds


def find_amdgpu_hwmon():
    for d in glob.glob(f"{DRM}/hwmon/hwmon*"):
        try:
            if open(f"{d}/name").read().strip() == "amdgpu":
                return d
        except OSError:
            pass
    return None


HWMON = find_amdgpu_hwmon()

STAGE_PATTERNS = [
    ("load", re.compile(r"load model from .* taking ([\d.]+)s|loaded .* in ([\d.]+)s", re.I)),
    ("te_ms", re.compile(r"get_learned_condition completed, taking ([\d.]+)s", re.I)),
    ("sampling_s", re.compile(r"sampling completed, taking ([\d.]+)s", re.I)),
    ("vae_decode_s", re.compile(r"decode_first_stage completed, taking ([\d.]+)s", re.I)),
    ("generate_s", re.compile(r"generate_image completed in ([\d.]+)s", re.I)),
]


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except OSError:
        return None


def read_meminfo():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0])  # kB
    except OSError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--logdir", default="/tmp/moss-autotune/raw")
    ap.add_argument("--record", default="/tmp/moss-autotune/records.jsonl")
    ap.add_argument("--min-free-gb", type=float, default=1.5,
                    help="abort the run if device free VRAM drops below this")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given")

    os.makedirs(args.logdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.logdir, f"{args.label}-{stamp}.log")

    vram_before = read_int(f"{DRM}/mem_info_vram_used")
    gtt_before = read_int(f"{DRM}/mem_info_gtt_used")
    mem_before = read_meminfo()

    vram_samples = []
    gtt_samples = []
    avail_samples = []
    busy_samples = []
    sclk_samples = []
    mclk_samples = []
    power_samples = []
    stop_sampler = threading.Event()

    def fast_vram_sampler():
        t_start = time.monotonic()
        while not stop_sampler.is_set():
            v = read_int(f"{DRM}/mem_info_vram_used")
            if v is not None:
                vram_samples.append((round(time.monotonic() - t_start, 3), v))
            time.sleep(0.02)

    t0 = time.monotonic()
    sampler = threading.Thread(target=fast_vram_sampler, daemon=True)
    sampler.start()
    vram_total = read_int(f"{DRM}/mem_info_vram_total") or 0
    guard_bytes = int(args.min_free_gb * (1 << 30))
    safety_abort = False
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        pid = proc.pid
        while proc.poll() is None:
            g = read_int(f"{DRM}/mem_info_gtt_used")
            v_now = read_int(f"{DRM}/mem_info_vram_used")
            if v_now is not None and vram_total and vram_total - v_now < guard_bytes:
                safety_abort = True
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            a = read_meminfo().get("MemAvailable")
            b = read_int(f"{DRM}/gpu_busy_percent")
            s = read_int(f"{HWMON}/freq1_input") if HWMON else None
            m = read_int(f"{HWMON}/freq2_input") if HWMON else None
            p = read_int(f"{HWMON}/power1_average") if HWMON else None
            if g is not None:
                gtt_samples.append(g)
            if a is not None:
                avail_samples.append(a)
            if b is not None:
                busy_samples.append(b)
            if s is not None:
                sclk_samples.append(s)
            if m is not None:
                mclk_samples.append(m)
            if p is not None:
                power_samples.append(p)
            time.sleep(SAMPLE_INTERVAL)
        rc = proc.wait()
    stop_sampler.set()
    sampler.join(timeout=1.0)
    wall = time.monotonic() - t0

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_mib = usage.ru_maxrss / 1024.0

    text = open(log_path, errors="replace").read()
    stages = {}
    for name, pat in STAGE_PATTERNS:
        vals = []
        for m in pat.findall(text):
            v = next((g for g in (m if isinstance(m, tuple) else (m,)) if g), None)
            if v:
                vals.append(float(v))
        if vals:
            stages[name] = vals[-1]

    out_path = None
    for i, a in enumerate(cmd):
        if a in ("-o", "--output"):
            out_path = cmd[i + 1]
    out_bytes = os.path.getsize(out_path) if out_path and os.path.exists(out_path) else 0

    vram_peak = max((v for _, v in vram_samples), default=None)
    vram_peak_t = max(vram_samples, key=lambda s: s[1])[0] if vram_samples else None
    rec = {
        "label": args.label,
        "start": stamp,
        "returncode": rc,
        "safety_abort": safety_abort,
        "wall_s": round(wall, 2),
        "stages": stages,
        "vram_used_before": vram_before,
        "vram_peak": vram_peak,
        "vram_peak_t": vram_peak_t,
        "vram_peak_delta": (vram_peak - vram_before) if (vram_peak and vram_before) else None,
        "vram_min_free": (read_int(f"{DRM}/mem_info_vram_total") - vram_peak) if vram_peak else None,
        "gtt_peak": max(gtt_samples) if gtt_samples else None,
        "gtt_before": gtt_before,
        "mem_available_min_mib": (min(avail_samples) / 1024.0) if avail_samples else None,
        "mem_available_before_mib": mem_before.get("MemAvailable", 0) / 1024.0,
        "swap_used_before_mib": (mem_before.get("SwapTotal", 0) - mem_before.get("SwapFree", 0)) / 1024.0,
        "child_peak_rss_mib": round(peak_rss_mib, 1),
        "samples": len(vram_samples),
        "gpu_busy_mean": round(sum(busy_samples) / len(busy_samples), 1) if busy_samples else None,
        "gpu_busy_p90": (sorted(busy_samples)[int(0.9 * (len(busy_samples) - 1))] if busy_samples else None),
        "gpu_busy_max": max(busy_samples) if busy_samples else None,
        "sclk_max_mhz": round(max(sclk_samples) / 1e6) if sclk_samples else None,
        "mclk_max_mhz": round(max(mclk_samples) / 1e6) if mclk_samples else None,
        "power_max_w": round(max(power_samples) / 1e6) if power_samples else None,
        "out_bytes": out_bytes,
        "log": log_path,
        "cmd": cmd,
    }
    with open(args.record, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))
    sys.exit(0 if rc == 0 and out_bytes > 0 else 1)


if __name__ == "__main__":
    main()
