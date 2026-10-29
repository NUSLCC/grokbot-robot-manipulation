#!/usr/bin/env bash
# 10_install_lerobot.sh - LeRobot (ACT, Diffusion Policy, SmolVLA, pi0/pi0.5 ports) + LIBERO/ALOHA/PushT sims.
# Env: $RSP_HOME/envs/lerobot (uv venv, Python 3.12, torch from the cu128 index). Idempotent.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv

SRC="$RSP_SRC/lerobot"; ENV="$RSP_ENVS/lerobot"
git_sync https://github.com/huggingface/lerobot.git "$SRC" "$LEROBOT_REF"

[ -x "$ENV/bin/python" ] || uv venv --python "$LEROBOT_PY" "$ENV"
# Preflight: disk space (LeRobot + CUDA wheels + datasets cache can exceed 20 GB)
check_disk_space 30 "$RSP_HOME"
PY="$ENV/bin/python"

log "Installing lerobot[$LEROBOT_EXTRAS] (torch backend $LEROBOT_TORCH_BACKEND)"
# --torch-backend picks the CUDA wheel variant (documented in LeRobot install guide for uv >= 0.6)
# Prefer GCC/g++ for building native extensions (egl-probe etc.) on Ubuntu where c++ may point to clang
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
uv pip install --python "$PY" --torch-backend "$LEROBOT_TORCH_BACKEND" -e "${SRC}[${LEROBOT_EXTRAS}]"

log "Verifying"
MUJOCO_GL=egl "$PY" - <<'PY'
import importlib.metadata as m, torch
print("lerobot", m.version("lerobot"), "| torch", torch.__version__, "cuda", torch.version.cuda, "| gpu ok:", torch.cuda.is_available())
for mod in ["lerobot.policies", "gym_pusht", "gym_aloha"]:
    try: __import__(mod); print("import ok:", mod)
    except Exception as e: print("import FAILED:", mod, e)
try:
    from libero.libero import benchmark  # provided by hf-libero
    print("LIBERO suites:", sorted(benchmark.get_benchmark_dict().keys()))
except Exception as e:
    print("LIBERO import FAILED (check EGL / hf-libero):", e)
PY
"$ENV/bin/lerobot-info" || true
ok "LeRobot env ready: source $ENV/bin/activate"
# Gated models notice
hf_check_access model google/paligemma-3b-pt-224 >/dev/null 2>&1 || true
warn "pi0/pi0.5 need the gated google/paligemma-3b-pt-224 tokenizer: accept its license on the Hub, then: $ENV/bin/hf auth login"
