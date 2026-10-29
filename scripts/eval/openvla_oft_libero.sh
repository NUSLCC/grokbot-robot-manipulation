#!/usr/bin/env bash
# OpenVLA-OFT released LIBERO checkpoints (~16 GB VRAM for LIBERO inference per README).
# Default 500 trials (10 tasks x 50) is the paper protocol; we default to 5/task for a quick run.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${SUITE:=libero_spatial}"; : "${TRIALS:=5}"; : "${SEED:=7}"
: "${CKPT:=moojink/openvla-7b-oft-finetuned-${SUITE//_/-}}"
PY="$RSP_ENVS/conda/openvla-oft/bin/python"; SRC="$RSP_SRC/openvla-oft"
[ -x "$PY" ] || die "Run scripts/12_install_openvla_oft.sh first"
OUT="$RSP_LOGS/eval_oft_libero_$(stamp)"; mkdir -p "$OUT"
cd "$SRC"
gpu_monitor_start "$OUT/gpu.csv"
"$PY" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$CKPT" --task_suite_name "$SUITE" \
  --num_trials_per_task "$TRIALS" --seed "$SEED" --center_crop True 2>&1 | tee "$OUT/eval.log"
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Log in $OUT (upstream also writes to experiments/logs)"
