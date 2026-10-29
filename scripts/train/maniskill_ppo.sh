#!/usr/bin/env bash
# ManiSkill3 PPO baselines (GPU-parallel). MODE=state (PushCube, minutes) or MODE=rgb (PickCube; README: settings < 15 GB).
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${MODE:=state}"
PY="$RSP_ENVS/maniskill/bin/python"; B="$RSP_SRC/ManiSkill/examples/baselines/ppo"
[ -x "$PY" ] || die "Run scripts/14_install_maniskill.sh first"
cd "$B"; gpu_monitor_start "$RSP_LOGS/maniskill_ppo_$MODE.gpu.csv"
if [ "$MODE" = state ]; then
  "$PY" ppo.py --env_id="${ENV_ID:-PushCube-v1}" --num_envs=2048 --update_epochs=8 --num_minibatches=32 \
    --total_timesteps="${TOTAL:-2_000_000}" --eval_freq=10 --num-steps=20
else
  "$PY" ppo_rgb.py --env_id="${ENV_ID:-PickCube-v1}" --num_envs=256 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps="${TOTAL:-10_000_000}"
fi
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Runs under $B/runs"
