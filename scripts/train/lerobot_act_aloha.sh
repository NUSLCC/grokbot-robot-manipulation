#!/usr/bin/env bash
# Train ACT from scratch on the ALOHA sim transfer-cube human demos, with periodic sim eval.
# LeRobot HW guide: ACT ~2-6 GB peak at batch 8; ~30-60 min for 5 epochs of a ~45k-frame dataset on a 4090 (indicative).
#   STEPS=20000 BATCH=8 EVAL_FREQ=5000 N_EVAL=10   (eval renders ALOHA in MuJoCo: EGL, or OSMesa on WSL2)
#   EVAL_ONLY=1 [CKPT=<run>/checkpoints/last/pretrained_model]   re-evaluate the newest (or given) checkpoint, no training
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${STEPS:=20000}"; : "${BATCH:=8}"; : "${EVAL_FREQ:=5000}"; : "${N_EVAL:=10}"; : "${EVAL_ONLY:=0}"
TASK=AlohaTransferCube-v0
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-train" ] || die "Run scripts/10_install_lerobot.sh first"
select_mujoco_gl "$ENV/bin/python"
# The eval at the end needs gym_aloha: check it now (seconds), not after the training (minutes).
lerobot_sim_ensure "$ENV/bin/python" aloha "$TASK"
lerobot_train_compat "$ENV/bin/lerobot-train"   # flag names differ across LeRobot releases
if [ "$EVAL_ONLY" = 1 ]; then
  : "${CKPT:=$(lerobot_latest_ckpt act_aloha_transfer_cube)}"
  [ -n "$CKPT" ] && [ -f "$CKPT/config.json" ] || die "EVAL_ONLY=1: no checkpoint found under $RSP_CKPT/act_aloha_transfer_cube_*/checkpoints/last (set CKPT=...)"
  OUT="$RSP_LOGS/eval_act_aloha_$(stamp)"; log "EVAL_ONLY=1: evaluating $CKPT ($N_EVAL episodes)"
  gpu_monitor_start "$OUT.gpu.csv"
  "$ENV/bin/lerobot-eval" --policy.path="$CKPT" --policy.device=cuda \
    --env.type=aloha --env.task="$TASK" \
    --eval.n_episodes="$N_EVAL" --eval.batch_size="$N_EVAL" "${LR_EVAL_SYNC[@]}" --output_dir="$OUT"
  echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Results in $OUT (see eval_info.json)"
  exit 0
fi
OUT="$RSP_CKPT/act_aloha_transfer_cube_$(stamp)"
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-train" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task="$TASK" \
  --steps="$STEPS" --batch_size="$BATCH" "$LR_EVAL_FREQ_FLAG=$EVAL_FREQ" --save_freq="$EVAL_FREQ" \
  --eval.n_episodes="$N_EVAL" --eval.batch_size="$N_EVAL" "${LR_EVAL_SYNC[@]}" \
  --output_dir="$OUT" --job_name=act_aloha_transfer_cube "${LR_TRACKER_OFF[@]}"
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"
ok "Checkpoints: $OUT/checkpoints ; re-evaluate without training: EVAL_ONLY=1 N_EVAL=50 bash scripts/train/lerobot_act_aloha.sh"
