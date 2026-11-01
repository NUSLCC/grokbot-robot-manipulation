#!/usr/bin/env bash
# EXPERIMENTAL on 24 GB: pi0.5 (LeRobot PyTorch) fine-tune with frozen VLM (train_expert_only), grad checkpointing, bf16.
# LeRobot docs size the full recipe for one 80 GB GPU; HW guide says large VLAs are "24 GB tight at BS 1".
# Whether this fits a 4090 at BATCH=1-4 is UNVERIFIED - watch the peak-memory line.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env
: "${STEPS:=5000}"; : "${BATCH:=2}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-train" ] || die "Run scripts/10_install_lerobot.sh first"
OUT="$RSP_CKPT/pi05_libero_expert_$(stamp)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
lerobot_train_compat "$ENV/bin/lerobot-train"   # flag names differ across LeRobot releases
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-train" \
  --dataset.repo_id=lerobot/libero --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_libero_base \
  --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
  --policy.n_action_steps=10 --policy.empty_cameras=1 \
  --policy.freeze_vision_encoder=true --policy.train_expert_only=true \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name=pi05_libero_expert --batch_size="$BATCH" --num_workers=8 \
  --steps="$STEPS" --save_freq=1000 --seed=1000 "${LR_TRACKER_OFF[@]}"
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Checkpoints: $OUT"
