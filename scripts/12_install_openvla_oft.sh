#!/usr/bin/env bash
# 12_install_openvla_oft.sh - OpenVLA-OFT (7B) + LIBERO, following SETUP.md / LIBERO.md.
# Env: conda prefix $RSP_HOME/envs/conda/openvla-oft (Python 3.10; torch 2.2.0 + transformers fork pinned by pyproject).
# flash-attn 2.5.5 is needed for TRAINING only; it compiles if no matching wheel -> needs a CUDA 12.x nvcc
# (set CUDA_HOME=/usr/local/cuda-12.x). Set SKIP_FLASH_ATTN=1 for inference-only installs.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_conda

PREFIX="$RSP_ENVS/conda/openvla-oft"; SRC="$RSP_SRC/openvla-oft"; LIB="$RSP_SRC/LIBERO"
check_disk_space 40 "$RSP_HOME"
conda_env_create "$PREFIX" "$OFT_PY"
PY="$PREFIX/bin/python"

git_sync https://github.com/moojink/openvla-oft.git "$SRC" "$OFT_REF"
git_sync https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIB" "$LIBERO_REF"

log "pip install -e openvla-oft (pulls torch==2.2.0, custom transformers fork, tensorflow 2.15 for RLDS)"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -e "$SRC"

if [ "${SKIP_FLASH_ATTN:-0}" != 1 ]; then
  "$PY" -m pip install packaging ninja
  if "$PY" -c "import flash_attn" 2>/dev/null; then ok "flash-attn already installed"
  else
    if [ -z "${CUDA_HOME:-}" ]; then
      for c in /usr/local/cuda-12.1 /usr/local/cuda-12.4 /usr/local/cuda-12.8 /usr/local/cuda-12; do [ -x "$c/bin/nvcc" ] && export CUDA_HOME="$c" && break; done
    fi
    log "Installing flash-attn==$OFT_FLASH_ATTN (CUDA_HOME=${CUDA_HOME:-unset})"
    PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH" MAX_JOBS="${MAX_JOBS:-8}" \
      "$PY" -m pip install "flash-attn==$OFT_FLASH_ATTN" --no-build-isolation \
      || warn "flash-attn failed to build. Inference still works; for training install a CUDA 12.x toolkit (01_base_deps.sh INSTALL_CUDA_TOOLKIT=12-4) and retry."
  fi
fi

log "LIBERO + OFT LIBERO requirements"
"$PY" -m pip install -e "$LIB"
"$PY" -m pip install -r "$SRC/experiments/robot/libero/libero_requirements.txt"
printf 'N\n' | MUJOCO_GL=egl "$PY" -c "from libero.libero import benchmark; print('LIBERO ok')" || warn "LIBERO import failed (see troubleshooting)"
"$PY" -c "import torch, transformers; print('torch', torch.__version__, torch.version.cuda, 'gpu', torch.cuda.is_available(), '| transformers', transformers.__version__)"
ok "OpenVLA-OFT env: $PREFIX"
