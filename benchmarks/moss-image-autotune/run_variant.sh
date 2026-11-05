#!/usr/bin/env bash
# Run one MOSS-Image benchmark variant through the recipe-faithful sd-cli path.
# usage: run_variant.sh LABEL W H [extra sd-cli args...]
set -euo pipefail

REPO=/home/trave/code/stable-diffusion.cpp
BIN=$REPO/build-vulkan/bin/sd-cli
LABEL=$1; W=$2; H=$3; shift 3

exec python3 "$REPO/benchmarks/moss-image-autotune/bench.py" --label "$LABEL" -- \
  "$BIN" \
  --diffusion-model /home/trave/models/diffusion_models/mossRefinedRealism_v208steps_int8.safetensors \
  --llm /home/trave/models/text_encoders/Qwen3.5-9B-Q4_K_M.gguf \
  --tokenizer /home/trave/models/tokenizers/qwen35-tokenizer.json \
  --vae /home/trave/models/vae/MOSS-Image-SFT-vae.safetensors \
  -W "$W" -H "$H" --steps 8 --cfg-scale 1.0 --sampling-method euler --flow-shift 5.0 -s 42 \
  -p "a photograph of a young woman standing on a city street at golden hour, shallow depth of field, 85mm lens, natural skin texture, highly detailed" \
  -o "/tmp/moss-autotune/raw/$LABEL.png" \
  "$@"
