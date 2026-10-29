#!/usr/bin/env bash
# Train Diffusion Policy from scratch on PushT (2D sim) with periodic sim eval.
# LeRobot HW guide: diffusion ~8-14 GB peak at batch 8 (indicative).
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${STEPS:=50000}"; : "${BATCH:=64}"; : "${EVAL_FREQ:=10000}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-train" ] || die "Run scripts/10_install_lerobot.sh first"
OUT="$RSP_CKPT/diffusion_pusht_$(stamp)"
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-train" \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --steps="$STEPS" --batch_size="$BATCH" --eval_freq="$EVAL_FREQ" --save_freq="$EVAL_FREQ" \
  --eval.n_episodes=50 --eval.batch_size=50 \
  --output_dir="$OUT" --job_name=diffusion_pusht --wandb.enable=false
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Checkpoints: $OUT/checkpoints"
warn "BATCH=64 is our default (upstream examples vary); drop to 8-32 if you hit OOM."
