#!/usr/bin/env bash
# ManiSkill3 PPO baselines (GPU-parallel). MODE=state (PushCube, minutes) or MODE=rgb (PickCube; README: settings < 15 GB).
#   MS_RENDER=auto|vulkan|none   auto (default): probe SAPIEN Vulkan rendering first; if it does not work (normal on
#                                WSL2) run state-PPO with rendering disabled (render_backend="none", no eval videos).
#                                MODE=rgb needs rendering, so without Vulkan it exits with code 77 (SKIP).
#   TOTAL=2_000_000  ENV_ID=PushCube-v1  NUM_ENVS=2048
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection
: "${MODE:=state}"; : "${MS_RENDER:=auto}"
PY="$RSP_ENVS/maniskill/bin/python"; B="$RSP_SRC/ManiSkill/examples/baselines/ppo"
[ -x "$PY" ] || die "Run scripts/14_install_maniskill.sh first"
PROBE="$RSP_KIT_DIR/scripts/lib/maniskill_probe.py"; NORENDER="$RSP_KIT_DIR/scripts/lib/maniskill_ppo_norender.py"

RENDER="$MS_RENDER"
if [ "$RENDER" = auto ]; then
  log "Probing SAPIEN Vulkan rendering (vulkaninfo: $(vulkan_status))"
  if timeout 300 "$PY" "$PROBE" gpu; then RENDER=vulkan
  elif timeout 300 "$PY" "$PROBE" none; then
    RENDER=none
    warn "SAPIEN rendering is not available$(is_wsl && echo ' (expected on WSL2: no NVIDIA Vulkan driver)'). Falling back to state-only PPO with rendering disabled (no eval videos)."
  else
    skip_exit "ManiSkill cannot create an env even without rendering (see the probe error above)"
  fi
fi
if [ "$MODE" = rgb ] && [ "$RENDER" = none ]; then skip_exit "MODE=rgb needs SAPIEN rendering (Vulkan), which is not available here"; fi

cd "$B"; gpu_monitor_start "$RSP_LOGS/maniskill_ppo_$MODE.gpu.csv"
if [ "$MODE" = state ]; then
  ARGS=(--env_id="${ENV_ID:-PushCube-v1}" --num_envs="${NUM_ENVS:-2048}" --update_epochs=8 --num_minibatches=32
        --total_timesteps="${TOTAL:-2_000_000}" --eval_freq=10 --num-steps=20)
  if [ "$RENDER" = none ]; then
    "$PY" "$NORENDER" ppo.py "${ARGS[@]}" --no-capture-video
  else
    "$PY" ppo.py "${ARGS[@]}"
  fi
else
  "$PY" ppo_rgb.py --env_id="${ENV_ID:-PickCube-v1}" --num_envs="${NUM_ENVS:-256}" --update_epochs=8 --num_minibatches=8 \
    --total_timesteps="${TOTAL:-10_000_000}"
fi
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Runs under $B/runs (render=$RENDER)"
