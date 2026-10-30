#!/usr/bin/env bash
# Evaluate the released SmolVLA LIBERO checkpoint (lerobot/smolvla_libero) with lerobot-eval.
# FIRST VALIDATION RUN. Defaults are a quick check (libero_object, 2 episodes/task). For a benchmark-protocol
# run use: SUITES=libero_spatial,libero_object,libero_goal,libero_10 N_EPISODES=10.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${POLICY:=lerobot/smolvla_libero}"; : "${SUITES:=libero_object}"; : "${N_EPISODES:=2}"; : "${BATCH:=1}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-eval" ] || die "Run scripts/10_install_lerobot.sh first"
select_mujoco_gl "$ENV/bin/python"   # EGL, or OSMesa fallback on WSL2
OUT="$RSP_LOGS/eval_smolvla_libero_$(stamp)"
# The checkpoint's config.json expects observation.images.camera1/camera2 while the LIBERO env emits
# observation.images.image/image2 -> rename map (mapping inferred from config.json; verify on first run).
RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-eval" \
  --policy.path="$POLICY" \
  --env.type=libero --env.task="$SUITES" \
  --eval.batch_size="$BATCH" --eval.n_episodes="$N_EPISODES" \
  --env.max_parallel_tasks=1 \
  --rename_map="$RENAME" \
  --output_dir="$OUT"
echo "Peak GPU memory (whole GPU): $(gpu_monitor_stop) MiB"; ok "Results in $OUT (see eval_info.json)"
