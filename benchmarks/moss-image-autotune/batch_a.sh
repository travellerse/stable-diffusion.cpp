#!/usr/bin/env bash
# Round A: single-variable comparisons at 1024x1024, interleaved to spread drift.
set -uo pipefail
R=/home/trave/code/stable-diffusion.cpp/benchmarks/moss-image-autotune/run_variant.sh
LABELS=()
run() { bash "$R" "$1" 1024 1024 "${@:2}" >/dev/null 2>&1; echo "done $1"; }
for rep in 1 2; do
  run a1-base-r$rep
  run a1-faon-r$rep  --flash-attn diffusion=on
  run a2-cdoff-r$rep --conv-direct diffusion=off
done
echo "BATCH_A_COMPLETE"
