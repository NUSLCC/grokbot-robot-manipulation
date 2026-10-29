#!/usr/bin/env bash
# Fine-tune SmolVLA (450M) from lerobot/smolvla_base on lerobot/libero (1.9 GB video dataset), then eval.
# LeRobot docs use batch 64 (multi-GPU class); HW guide: small VLA ~10-16 GB at batch 8 -> default 8 here.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${STEPS:=20000}"; : "${BATCH:=8}"; : "${SAVE_FREQ:=5000}"
ENV="$RSP_ENVS/lerobot"; [ -x "$ENV/bin/lerobot-train" ] || die "Run scripts/10_install_lerobot.sh first"
OUT="$RSP_CKPT/smolvla_libero_$(stamp)"
gpu_monitor_start "$OUT.gpu.csv"
"$ENV/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda \
  --dataset.repo_id=lerobot/libero --dataset.video_backend=torchcodec \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --steps="$STEPS" --batch_size="$BATCH" --save_freq="$SAVE_FREQ" \
  --policy.scheduler_decay_steps="$STEPS" \
  --output_dir="$OUT" --job_name=smolvla_libero --wandb.enable=false
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"
ok "Evaluate: POLICY=$OUT/checkpoints/last/pretrained_model bash scripts/eval/lerobot_smolvla_libero.sh"
