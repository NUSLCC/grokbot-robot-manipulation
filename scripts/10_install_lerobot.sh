#!/usr/bin/env bash
# 10_install_lerobot.sh - LeRobot (ACT, Diffusion Policy, SmolVLA, pi0/pi0.5 ports) + LIBERO/ALOHA/PushT sims.
# Env: $RSP_HOME/envs/lerobot (uv venv, Python 3.12, torch from the $TORCH_CUDA index - cu128 by default,
# which also covers Blackwell / RTX 5090). Idempotent.
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
if is_blackwell; then
  case "$LEROBOT_TORCH_BACKEND" in cu12[89]|cu13*) ok "Blackwell: $LEROBOT_TORCH_BACKEND wheels include sm_120" ;;
    *) warn "Blackwell GPU but LEROBOT_TORCH_BACKEND=$LEROBOT_TORCH_BACKEND: sm_120 needs cu128 or newer (unset it to use the default)" ;; esac
fi
# --torch-backend picks the CUDA wheel variant (documented in LeRobot install guide for uv >= 0.6)
# Prefer GCC/g++ for building native extensions (egl-probe etc.) on Ubuntu where c++ may point to clang
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
uv pip install --python "$PY" --torch-backend "$LEROBOT_TORCH_BACKEND" -e "${SRC}[${LEROBOT_EXTRAS}]"

log "Verifying"
select_mujoco_gl "$PY"
# LIBERO asks a Y/N question on first import unless its config exists: write it now (defaults, no prompt).
libero_config "$PY" lerobot
"$PY" - <<'PY'
import importlib.metadata as m, os, torch
print("lerobot", m.version("lerobot"), "| torch", torch.__version__, "cuda", torch.version.cuda, "| gpu ok:", torch.cuda.is_available(), "| MUJOCO_GL", os.environ.get("MUJOCO_GL"))
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    print("gpu capability", cap, "| torch arch list", archs)
    if f"sm_{cap[0]}{cap[1]}" not in archs:
        print("WARNING: this torch build has no kernels for", f"sm_{cap[0]}{cap[1]}")
    (torch.ones(8, device="cuda") * 2).sum().item()
    print("cuda kernel launch ok")
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
# Which flag spellings this LeRobot uses (the train scripts detect them the same way at run time)
lerobot_train_compat "$ENV/bin/lerobot-train"
ok "LeRobot env ready: source $ENV/bin/activate"
# Gated models notice
hf_check_access model google/paligemma-3b-pt-224 >/dev/null 2>&1 || true
warn "pi0/pi0.5 need the gated google/paligemma-3b-pt-224 tokenizer: accept its license on the Hub, then: $ENV/bin/hf auth login"
