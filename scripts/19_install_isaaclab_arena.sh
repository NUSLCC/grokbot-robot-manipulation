#!/usr/bin/env bash
# 19_install_isaaclab_arena.sh - OPTIONAL Isaac Lab-Arena (policy evaluation for GR00T / pi0.5 / custom servers).
# Arena v0.3.x is ALPHA (README: "Do not use this in production"); main targets Isaac Lab 3.0 + Isaac Sim 6.2.
# `uv sync` inside the Arena checkout creates ITS OWN .venv (Isaac Lab from source + Isaac Sim/PyTorch/Newton
# wheels). It does not touch your existing Isaac install, but it is a second multi-GB Isaac stack on disk.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv
[ "${I_UNDERSTAND_ARENA_IS_ALPHA:-0}" = 1 ] || die "Set I_UNDERSTAND_ARENA_IS_ALPHA=1 to proceed (second Isaac stack, alpha software)."
SRC="$RSP_SRC/IsaacLab-Arena"
git_sync https://github.com/isaac-sim/IsaacLab-Arena.git "$SRC" "$ISAACLAB_ARENA_REF"
cd "$SRC"
uv sync
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
.venv/bin/python isaaclab_arena/evaluation/policy_runner.py --policy_type zero_action --num_steps 20 cube_goal_pose
ok "Arena ready at $SRC (GR00T server-client mode: see Arena installation docs)"
