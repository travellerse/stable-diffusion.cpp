#!/usr/bin/env bash
# Round 2: 1024 without GPU preview (it trips the 1.5 GiB safety gate),
# plus 512/768 where the GPU preview fits.
set -uo pipefail
REPO=/home/trave/code/stable-diffusion.cpp
RUN=$REPO/benchmarks/moss-image-autotune/run_variant.sh
TAE=/home/trave/models/vae_approx/taef2.safetensors
PV=/tmp/moss-autotune/preview
mkdir -p "$PV" /tmp/moss-autotune/raw

for round in 3 4; do
  bash "$RUN" "tae-a${round}-1024" 1024 1024 || true
  bash "$RUN" "tae-c${round}-1024" 1024 1024 --preview tae --preview-interval 1 \
    --taesd "$TAE" --taesd-preview-only --backend tae-preview=cpu \
    --preview-path "$PV/c${round}_%03d.png" || true
done
bash "$RUN" "tae-d2-1024" 1024 1024 --preview tae --preview-interval 1 \
  --taesd "$TAE" --taesd-preview-only --backend vae=cpu \
  --preview-path "$PV/d2_%03d.png" || true

for size in 512 768; do
  bash "$RUN" "tae-a5-${size}" "$size" "$size" || true
  bash "$RUN" "tae-b5-${size}" "$size" "$size" --preview tae --preview-interval 1 \
    --taesd "$TAE" --taesd-preview-only \
    --preview-path "$PV/b5_${size}_%03d.png" || true
  bash "$RUN" "tae-c5-${size}" "$size" "$size" --preview tae --preview-interval 1 \
    --taesd "$TAE" --taesd-preview-only --backend tae-preview=cpu \
    --preview-path "$PV/c5_${size}_%03d.png" || true
done
