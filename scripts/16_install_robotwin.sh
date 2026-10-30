#!/usr/bin/env bash
# 16_install_robotwin.sh - RoboTwin 2.0 (50 bimanual tasks, SAPIEN) following the official install doc.
# Env: conda prefix $RSP_HOME/envs/conda/robotwin (Python 3.10; docs recommend CUDA 12.1).
# scripts/_install.sh builds CuRobo (and pytorch3d) -> needs nvcc; set CUDA_HOME to a CUDA 12.x toolkit.
# ROBOTWIN_DATA_TASKS="beat_block_hammer adjust_bottle" downloads pre-collected data for those tasks (optional).
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_conda
is_blackwell && warn "Blackwell GPU: RoboTwin's stack (CUDA 12.1 recommended, CuRobo/pytorch3d builds) is NOT expected to work on sm_120 without manual torch/CUDA upgrades."
is_wsl && warn "WSL2: RoboTwin renders with SAPIEN/Vulkan, which WSL2 does not provide for NVIDIA GPUs today."

PREFIX="$RSP_ENVS/conda/robotwin"; SRC="$RSP_SRC/RoboTwin"
conda_env_create "$PREFIX" "$ROBOTWIN_PY"
git_sync https://github.com/RoboTwin-Platform/RoboTwin.git "$SRC" "$ROBOTWIN_REF"
cd "$SRC"
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /usr/local/cuda-12.1 /usr/local/cuda-12.4 /usr/local/cuda-12.8 /usr/local/cuda; do [ -x "$c/bin/nvcc" ] && export CUDA_HOME="$c" && break; done
fi
log "CUDA_HOME=${CUDA_HOME:-unset}"
conda_activate "$PREFIX"
if [ ! -f "$SRC/.rsp_installed" ]; then
  bash scripts/_install.sh && touch "$SRC/.rsp_installed" || warn "_install.sh failed; see 'Manual Installation' in the RoboTwin install doc"
else ok "_install.sh already completed (delete $SRC/.rsp_installed to rerun)"; fi
if [ ! -d assets/embodiments ]; then bash scripts/_download_assets.sh; else ok "assets present"; fi
[ -n "${ROBOTWIN_DATA_TASKS:-}" ] && bash scripts/download_xpolicylab_data.sh $ROBOTWIN_DATA_TASKS
python -c "import sapien, torch; print('sapien', sapien.__version__, '| torch', torch.__version__, torch.cuda.is_available())"
ok "RoboTwin env: $PREFIX  (collect data: bash collect_data.sh beat_block_hammer demo_randomized 0)"
warn "Upstream note: mplib planner.py may need the 'or collide' edit described in the install doc if motion planning fails."
