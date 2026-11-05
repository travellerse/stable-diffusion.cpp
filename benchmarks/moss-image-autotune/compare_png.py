#!/usr/bin/env python3
"""Pixel comparison between two PNGs: MAE, RMSE, PSNR, max abs diff.
Uses PIL when available, falls back to a pure-Python PNG decode via zlib for
8-bit RGB/RGBA non-interlaced files."""
import sys
import zlib
import struct


def decode_png(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos = 8
    idat = b""
    w = h = bitdepth = colortype = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            w, h, bitdepth, colortype, comp, filt, interlace = struct.unpack(">IIBBBBB", chunk)
            assert bitdepth == 8 and interlace == 0, f"unsupported PNG bitdepth={bitdepth} interlace={interlace}"
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[colortype]
        elif ctype == b"IDAT":
            idat += chunk
        elif ctype == b"IEND":
            break
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = w * channels
    out = bytearray(h * stride)
    prev = bytearray(stride)
    p = 0
    for y in range(h):
        ft = raw[p]
        p += 1
        line = bytearray(raw[p:p + stride])
        p += stride
        if ft == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return w, h, channels, bytes(out)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    wa, ha, ca, da = decode_png(a_path)
    wb, hb, cb, db = decode_png(b_path)
    if (wa, ha) != (wb, hb):
        print(f"SIZE MISMATCH {wa}x{ha} vs {wb}x{hb}")
        return
    n = min(len(da), len(db))
    step = 4 if ca == 4 else 3
    tot = 0
    sq = 0
    mx = 0
    count = 0
    for i in range(0, n - step + 1, step):
        for k in range(3):
            d = abs(da[i + k] - db[i + k])
            tot += d
            sq += d * d
            mx = max(mx, d)
            count += 1
    mae = tot / count
    mse = sq / count
    psnr = 10 * __import__("math").log10(255 * 255 / mse) if mse > 0 else float("inf")
    print(f"{a_path} vs {b_path}: {wa}x{ha} MAE={mae:.3f} RMSE={mse ** 0.5:.3f} PSNR={psnr:.2f}dB maxabs={mx}")


if __name__ == "__main__":
    main()
