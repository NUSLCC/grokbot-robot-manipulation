#!/usr/bin/env bash
# Train ACT from scratch on the ALOHA sim transfer-cube human demos, with periodic sim eval.
# LeRobot HW guide: ACT ~2-6 GB peak at batch 8; ~30-60 min for 5 epochs of a ~45k-frame dataset on a 4090 (indicative).
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${STEPS:=20000}"; : "${BATCH:=8}"; : "${EVAL_FREQ:=5000}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-train" ] || die "Run scripts/10_install_lerobot.sh first"
OUT="$RSP_CKPT/act_aloha_transfer_cube_$(stamp)"
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-train" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --steps="$STEPS" --batch_size="$BATCH" --eval_freq="$EVAL_FREQ" --save_freq="$EVAL_FREQ" \
  --eval.n_episodes=10 --eval.batch_size=10 \
  --output_dir="$OUT" --job_name=act_aloha_transfer_cube --wandb.enable=false
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"
ok "Checkpoints: $OUT/checkpoints ; re-evaluate: $ENV/bin/lerobot-eval --policy.path=$OUT/checkpoints/last/pretrained_model --env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=50 --eval.batch_size=10"
