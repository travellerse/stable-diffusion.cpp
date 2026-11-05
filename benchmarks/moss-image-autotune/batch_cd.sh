#!/usr/bin/env bash
# Round C: resolution matrix screening. Round D: native cache evaluation.
R=/home/trave/code/stable-diffusion.cpp/benchmarks/moss-image-autotune/run_variant.sh
run() { bash "$R" "$1" "$2" "$3" "${@:4}" >/dev/null 2>&1; echo "done $1 rc-file"; }

run c-1328x752  1328 752
run c-1408x1056 1408 1056
run c-1632x912  1632 912
run c-1408x1408 1408 1408
run c-1888x1056 1888 1056
run c-1152x1728 1152 1728
run c-1440x1920 1440 1920
run d-easycache-1024 1024 1024 --cache-mode easycache
run d-easycache-1024b 1024 1024 --cache-mode easycache
run d-easycache-1888 1888 1056 --cache-mode easycache
echo "BATCH_CD_COMPLETE"
