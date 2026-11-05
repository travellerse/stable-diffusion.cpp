#!/usr/bin/env bash
# A/B: preview TAE on GPU vs CPU, at the recipe's 1024x1024 / 8 steps.
#   a = no preview, b = preview on Vulkan, c = preview on CPU (new flag),
#   d = whole vae module on CPU (existing flag, reference)
set -euo pipefail
REPO=/home/trave/code/stable-diffusion.cpp
RUN=$REPO/benchmarks/moss-image-autotune/run_variant.sh
TAE=/home/trave/models/vae_approx/taef2.safetensors
PV=/tmp/moss-autotune/preview
mkdir -p "$PV" /tmp/moss-autotune/raw

for round in 1 2; do
  bash "$RUN" "tae-a${round}" 1024 1024
  bash "$RUN" "tae-b${round}" 1024 1024 --preview tae --preview-interval 1 \
    --taesd "$TAE" --taesd-preview-only --preview-path "$PV/b${round}_%03d.png"
  bash "$RUN" "tae-c${round}" 1024 1024 --preview tae --preview-interval 1 \
    --taesd "$TAE" --taesd-preview-only --backend tae-preview=cpu \
    --preview-path "$PV/c${round}_%03d.png"
done
bash "$RUN" "tae-d1" 1024 1024 --preview tae --preview-interval 1 \
  --taesd "$TAE" --taesd-preview-only --backend vae=cpu \
  --preview-path "$PV/d1_%03d.png"
