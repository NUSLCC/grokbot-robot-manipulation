#!/usr/bin/env bash
# 17_install_rlinf.sh - RLinf (phase 3: RL fine-tuning of VLAs: OpenVLA-OFT, pi0/pi0.5, GR00T on ManiSkill/LIBERO).
# Official options (verified, RLinf docs): (1) UV via requirements/install.sh  (2) Docker image.
#   RLINF_MODEL=openvla-oft|openvla|openpi   RLINF_ENV=maniskill_libero|libero   RLINF_USE_DOCKER=1
# VRAM on one 24 GB GPU is NOT documented by RLinf for these 7B / 3B models -> treat as experimental.
# This is separate from the Isaac Lab-integrated RLinf path (see docs/ISAAC_LAB.md), which modifies the Isaac env.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env
is_blackwell && warn "Blackwell GPU: RLinf's torch/CUDA per model is chosen by its own installer; sm_120 support is unverified."
: "${RLINF_MODEL:=openvla-oft}"; : "${RLINF_ENV:=maniskill_libero}"

if [ "${RLINF_USE_DOCKER:-0}" = 1 ]; then
  have docker || die "docker not installed"
  docker info 2>/dev/null | grep -qi nvidia || warn "NVIDIA container toolkit not detected"
  docker pull "$RLINF_DOCKER_IMAGE"
  cat <<MSG
Start the container (from RLinf docs):
  docker run -it --gpus all --shm-size 32g --net=host --name rlinf \\
    -e NVIDIA_DRIVER_CAPABILITIES=all -v $RSP_HOME:/workspace/rsp $RLINF_DOCKER_IMAGE /bin/bash
  # inside: git clone https://github.com/RLinf/RLinf.git && cd RLinf && source switch_env $RLINF_MODEL
(RLinf docs use --shm-size 100g; 32g is our reduced value for a workstation - raise if dataloaders crash.)
MSG
  exit 0
fi

ensure_uv
SRC="$RSP_SRC/RLinf"; VENV="$RSP_ENVS/rlinf-$RLINF_MODEL"
git_sync https://github.com/RLinf/RLinf.git "$SRC" "$RLINF_REF"
cd "$SRC"
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import rlinf" 2>/dev/null; then ok "RLinf venv exists: $VENV"
else
  log "bash requirements/install.sh embodied --model $RLINF_MODEL --env $RLINF_ENV --venv $VENV"
  bash requirements/install.sh embodied --model "$RLINF_MODEL" --env "$RLINF_ENV" --venv "$VENV"
fi
"$VENV/bin/python" -c "import rlinf; print('rlinf at', rlinf.__file__)"
"$VENV/bin/ray" --version || true
ok "RLinf ready. Next: scripts/rl/rlinf_maniskill_quickstart.sh"
