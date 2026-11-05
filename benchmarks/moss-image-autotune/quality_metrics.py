#!/usr/bin/env python3
"""Image quality metrics for paired FP16 / INT8-QK outputs.

usage: quality_metrics.py DIR_F16 DIR_I8 PREFIX [PREFIX ...]
Computes PSNR / MAE / RMSE / SSIM (gaussian window) per pair and writes a CSV.
LPIPS and CLIP are not available offline on this machine and are reported as n/a.
"""
import csv
import os
import sys

import numpy as np
from PIL import Image


def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr(a, b, peak=255.0):
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(peak * peak / mse)


def ssim(a, b, peak=255.0):
    # 11x11 gaussian (sigma 1.5) applied with FFT convolution
    def gkern(size=11, sigma=1.5):
        x = np.arange(size) - size // 2
        g = np.exp(-(x ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def filt(img):
        k = gkern()
        K = np.outer(k, k)
        H, W = img.shape
        pad = 5
        p = np.zeros((H + 2 * pad, W + 2 * pad))
        p[pad:pad + H, pad:pad + W] = img
        p[pad:pad + H, :pad] = img[:, :1]
        p[pad:pad + H, -pad:] = img[:, -1:]
        p[:pad, :] = p[pad:pad + 1, :]
        p[-pad:, :] = p[-pad - 1:-pad, :]
        fh, fw = p.shape
        fk = np.zeros_like(p)
        fk[:K.shape[0], :K.shape[1]] = K
        fk = np.roll(fk, (-(K.shape[0] // 2), -(K.shape[1] // 2)), axis=(0, 1))
        out = np.real(np.fft.ifft2(np.fft.fft2(p) * np.fft.fft2(fk)))
        return out[pad:pad + H, pad:pad + W]

    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    vals = []
    for c in range(a.shape[2]):
        x = a[:, :, c]
        y = b[:, :, c]
        mx, my = filt(x), filt(y)
        mxx, myy, mxy = filt(x * x), filt(y * y), filt(x * y)
        vx = mxx - mx * mx
        vy = myy - my * my
        vxy = mxy - mx * my
        s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
        vals.append(s.mean())
    return float(np.mean(vals))


def main():
    dst_root, src_root = sys.argv[1], sys.argv[2]
    prefixes = sys.argv[3:]
    rows = []
    for p in prefixes:
        fa = os.path.join(dst_root, p + ".png")
        fb = os.path.join(src_root, p + ".png")
        if not (os.path.exists(fa) and os.path.exists(fb)):
            print(f"missing pair for {p}", file=sys.stderr)
            continue
        a, b = load(fa), load(fb)
        d = np.abs(a - b)
        rows.append({
            "pair": p,
            "psnr": psnr(a, b),
            "mae": float(d.mean()),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "ssim": ssim(a, b),
            "lpips": "n/a",
        })
        print(f"{p:24s} PSNR={rows[-1]['psnr']:6.2f} SSIM={rows[-1]['ssim']:.4f} "
              f"MAE={rows[-1]['mae']:6.2f} RMSE={rows[-1]['rmse']:6.2f}")
    if rows:
        out = os.path.join(dst_root, "metrics.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)
        arr = {k: [r[k] for r in rows if isinstance(r[k], float)] for k in ("psnr", "mae", "rmse", "ssim")}
        for k, v in arr.items():
            if v:
                print(f"mean {k}: {np.mean(v):.4f}  min {np.min(v):.4f}  max {np.max(v):.4f}")


if __name__ == "__main__":
    main()
