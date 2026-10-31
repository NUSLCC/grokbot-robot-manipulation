#!/usr/bin/env bash
# Train Diffusion Policy from scratch on PushT (2D sim) with periodic sim eval.
# LeRobot HW guide: diffusion ~8-14 GB peak at batch 8 (indicative).
#   STEPS=50000 BATCH=64 EVAL_FREQ=10000 N_EVAL=50   (PushT renders with pygame; SDL_VIDEODRIVER=dummy keeps it windowless)
#   EVAL_ONLY=1 [CKPT=<run>/checkpoints/last/pretrained_model]   re-evaluate the newest (or given) checkpoint, no training
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${STEPS:=50000}"; : "${BATCH:=64}"; : "${EVAL_FREQ:=10000}"; : "${N_EVAL:=50}"; : "${EVAL_ONLY:=0}"
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"
# Peak GPU memory is mostly PyTorch's cache + cuDNN autotune workspaces (see TROUBLESHOOTING.md); keep the cache compact.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-train" ] || die "Run scripts/10_install_lerobot.sh first"
# The eval at the end needs gym_pusht: check it now (seconds), not after the training (minutes).
lerobot_sim_ensure "$ENV/bin/python" pusht PushT-v0
lerobot_train_compat "$ENV/bin/lerobot-train"   # flag names differ across LeRobot releases
if [ "$EVAL_ONLY" = 1 ]; then
  : "${CKPT:=$(lerobot_latest_ckpt diffusion_pusht)}"
  [ -n "$CKPT" ] && [ -f "$CKPT/config.json" ] || die "EVAL_ONLY=1: no checkpoint found under $RSP_CKPT/diffusion_pusht_*/checkpoints/last (set CKPT=...)"
  OUT="$RSP_LOGS/eval_diffusion_pusht_$(stamp)"; log "EVAL_ONLY=1: evaluating $CKPT ($N_EVAL episodes)"
  gpu_monitor_start "$OUT.gpu.csv"
  "$ENV/bin/lerobot-eval" --policy.path="$CKPT" --policy.device=cuda --env.type=pusht \
    --eval.n_episodes="$N_EVAL" --eval.batch_size="$N_EVAL" "${LR_EVAL_SYNC[@]}" --output_dir="$OUT"
  echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Results in $OUT (see eval_info.json)"
  exit 0
fi
OUT="$RSP_CKPT/diffusion_pusht_$(stamp)"
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-train" \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/pusht --env.type=pusht \
  --steps="$STEPS" --batch_size="$BATCH" "$LR_EVAL_FREQ_FLAG=$EVAL_FREQ" --save_freq="$EVAL_FREQ" \
  --eval.n_episodes="$N_EVAL" --eval.batch_size="$N_EVAL" "${LR_EVAL_SYNC[@]}" \
  --output_dir="$OUT" --job_name=diffusion_pusht "${LR_TRACKER_OFF[@]}"
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Checkpoints: $OUT/checkpoints ; re-evaluate without training: EVAL_ONLY=1 bash scripts/train/lerobot_diffusion_pusht.sh"
warn "BATCH=64 is our default (upstream examples vary); drop to 8-32 if you hit OOM."
