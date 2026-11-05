#!/usr/bin/env bash
R=/home/trave/code/stable-diffusion.cpp/benchmarks/moss-image-autotune/run_variant.sh
bash "$R" f-2mp-faon 1408 1408 --flash-attn diffusion=on >/dev/null 2>&1; echo "done 2mp-faon"
bash "$R" f-easycache-r1 1024 1024 --cache-mode easycache >/dev/null 2>&1; echo "done easycache1"
bash "$R" b-base-r3 1024 1024 >/dev/null 2>&1; echo "done base3"
bash "$R" e-faoff-r1 1024 1024 --flash-attn diffusion=off >/dev/null 2>&1; echo "done faoff1"
bash "$R" f-easycache-r2 1024 1024 --cache-mode easycache >/dev/null 2>&1; echo "done easycache2"
echo "BATCH_F_COMPLETE"
