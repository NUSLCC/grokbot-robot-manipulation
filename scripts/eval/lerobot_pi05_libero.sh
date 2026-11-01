#!/usr/bin/env bash
# Evaluate LeRobot's pi0.5 LIBERO checkpoint (protocol from LeRobot docs: n_action_steps=10).
# Needs HF login + accepted license for google/paligemma-3b-pt-224 (gated tokenizer).
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env
: "${POLICY:=lerobot/pi05_libero_finetuned_v044}"   # 'lerobot/pi05_libero_finetuned' redirects here (checked 2026-10-29)
: "${SUITES:=libero_spatial}"; : "${N_EPISODES:=2}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-eval" ] || die "Run scripts/10_install_lerobot.sh first"
select_mujoco_gl "$ENV/bin/python"
libero_config "$ENV/bin/python" lerobot   # no first-import prompt (EOFError headless)   # EGL, or OSMesa fallback on WSL2
OUT="$RSP_LOGS/eval_pi05_lerobot_libero_$(stamp)"
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-eval" --output_dir="$OUT" --env.type=libero --env.task="$SUITES" \
  --eval.batch_size=1 --eval.n_episodes="$N_EPISODES" --policy.path="$POLICY" \
  --policy.n_action_steps=10 --env.max_parallel_tasks=1
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Results in $OUT"
