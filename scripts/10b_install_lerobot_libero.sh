#!/usr/bin/env bash
# 10b_install_lerobot_libero.sh - dedicated LeRobot env for the LIBERO evals, with MuJoCo pinned to $LIBERO_MUJOCO (3.3.2).
# Why: MuJoCo >= 3.4 breaks libero_spatial task 5 (SmolVLA 0/10 -> 8/10 on 3.3.2). ACT/ALOHA/PushT keep envs/lerobot.
# Env: $RSP_HOME/envs/$LEROBOT_LIBERO_ENV (default lerobot-libero), same LeRobot source checkout as 10_install_lerobot.sh.
# Idempotent: rerunning only adds what is missing and re-applies the pins.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv
: "${LEROBOT_LIBERO_ENV:=lerobot-libero}"
SRC="$RSP_SRC/lerobot"; ENV="$RSP_ENVS/$LEROBOT_LIBERO_ENV"
git_sync https://github.com/huggingface/lerobot.git "$SRC" "$LEROBOT_REF"
check_disk_space 20 "$RSP_HOME"
[ -x "$ENV/bin/python" ] || uv venv --python "$LEROBOT_PY" "$ENV"
PY="$ENV/bin/python"
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
log "Installing lerobot[$LEROBOT_LIBERO_EXTRAS] + mujoco==$LIBERO_MUJOCO dm-control==$LIBERO_DM_CONTROL into $ENV"
uv pip install --python "$PY" --torch-backend "$LEROBOT_TORCH_BACKEND" -e "${SRC}[${LEROBOT_LIBERO_EXTRAS}]" \
  "mujoco==$LIBERO_MUJOCO" "dm-control==$LIBERO_DM_CONTROL"
select_mujoco_gl "$PY"
libero_config "$PY" lerobot-libero   # own config dir: envs/lerobot keeps its LIBERO config
"$PY" - "$LIBERO_MUJOCO" <<'PY'
import sys, importlib.metadata as m, torch, mujoco
from libero.libero import benchmark
print("lerobot", m.version("lerobot"), "| mujoco", mujoco.__version__, "| torch", torch.__version__, "| gpu ok:", torch.cuda.is_available())
print("LIBERO suites:", sorted(benchmark.get_benchmark_dict().keys()))
sys.exit(0 if mujoco.__version__ == sys.argv[1] else f"mujoco {mujoco.__version__} != {sys.argv[1]}")
PY
ok "LeRobot LIBERO env ready: $ENV (mujoco $LIBERO_MUJOCO)"
