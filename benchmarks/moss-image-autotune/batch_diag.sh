#!/usr/bin/env bash
# Diagnostic runs at 512x512 to localise where the DiT step time goes.
set -uo pipefail
R=/home/trave/code/stable-diffusion.cpp/benchmarks/moss-image-autotune/run_variant.sh

bash "$R" diag-512-base 512 512 >/dev/null 2>&1; echo "done base"
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1 bash "$R" diag-512-noidp 512 512 >/dev/null 2>&1; echo "done noidp"
bash "$R" diag-512-monolithic 512 512 --disable-segmented-compute >/dev/null 2>&1; echo "done monolithic"
bash "$R" diag-512-vlog 512 512 -v >/dev/null 2>&1; echo "done vlog"
echo "BATCH_DIAG_COMPLETE"
