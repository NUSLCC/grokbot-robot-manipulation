#!/usr/bin/env bash
# Isaac Lab 3.0 RL on built-in Franka manipulation tasks, using the EXISTING Isaac env (no syncing/installs).
#   TASK: Isaac-Lift-Franka | Isaac-Reach-Franka | Isaac-Reach-Franka-OSC | Isaac-Open-Drawer-Franka |
#         Isaac-Open-Drawer-Franka-Direct | Isaac-Lift-KukaAllegro | Isaac-Reorient-KukaAllegro   (IDs from release/3.0.0 source)
#   PHYSICS: empty = default (Newton) | isaacsim_physx (Isaac Sim 6.2 PhysX) | other presets shown by `list_envs`
#   RL_LIB: rsl_rl (default, installed by Isaac Lab) | skrl | sb3 | rl_games (only if that extra is installed)
# Runs headless: passes ISAACLAB_HEADLESS_FLAGS (default "--viz none --livestream 0"; Isaac Lab 3.0 rejects --headless).
# Add VIDEO=1 for --video recordings during play (needs the 'video' extra). SKIP_ISAAC=1 skips (exit 77).
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; apply_gpu_selection
source "$(dirname "$0")/../lib/isaaclab_env.sh"; isaac_skip_if_requested; isaaclab_wsl_note
: "${TASK:=Isaac-Lift-Franka}"; : "${NUM_ENVS:=1024}"; : "${MAX_ITERS:=1500}"; : "${RL_LIB:=rsl_rl}"
: "${PHYSICS:=}"; : "${SEED:=42}"; : "${PLAY:=1}"; : "${VIDEO:=0}"
RUN="$(dirname "$0")/../lib/isaaclab_run.sh"
EXTRA=(); [ -n "$PHYSICS" ] && EXTRA+=("physics=$PHYSICS")
LOG="$RSP_LOGS/isaaclab_${TASK}_$(stamp)"; mkdir -p "$LOG"
log "Training $TASK with $RL_LIB (num_envs=$NUM_ENVS, max_iterations=$MAX_ITERS) - VRAM scales with num_envs/cameras"
gpu_monitor_start "$LOG/gpu_train.csv"
"$RUN" cli train --rl_library "$RL_LIB" --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" \
  --seed "$SEED" ${EXTRA[@]+"${EXTRA[@]}"} $ISAACLAB_HEADLESS_FLAGS 2>&1 | tee "$LOG/train.log"
echo "Peak GPU memory (train): $(gpu_monitor_stop) MiB"
if [ "$PLAY" = 1 ]; then
  PARGS=(--rl_library "$RL_LIB" --task "$TASK" --num_envs 32); [ "$VIDEO" = 1 ] && PARGS+=(--video)
  log "Playing latest checkpoint (Isaac Lab picks the latest run under its logs/ dir by default)"
  "$RUN" cli play "${PARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} $ISAACLAB_HEADLESS_FLAGS 2>&1 | tee "$LOG/play.log" || warn "play failed; pass --checkpoint via PLAY_ARGS if needed"
fi
ok "Logs: $LOG ; Isaac Lab run dirs: \$ISAACLAB_PATH/logs/$RL_LIB/"
