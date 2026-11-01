#!/usr/bin/env bash
# 18_install_simplevla_rl.sh - SimpleVLA-RL (veRL + OpenVLA-OFT, LIBERO option) per SETUP.md.
# WARNING: upstream RL training was tested on 8x A800 80 GB (single node) and 16x A800. A single 24 GB RTX 4090
# cannot hold the 7B actor + rollout for RL training. Install this only to read the code, run evaluation
# (trainer.val_only=True) experimentally, or to prepare jobs for a bigger machine.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_conda

PREFIX="$RSP_ENVS/conda/simplevla"; W="$RSP_SRC/simplevla_ws"; mkdir -p "$W"
conda_env_create "$PREFIX" 3.10; PY="$PREFIX/bin/python"
warn_torch_arch "SimpleVLA-RL" "$SIMPLEVLA_TORCH"
"$PY" -m pip install "torch==$SIMPLEVLA_TORCH" --index-url https://download.pytorch.org/whl/cu124
git_sync https://github.com/volcengine/verl.git "$W/verl" "$VERL_BRANCH"
"$PY" -m pip install -e "$W/verl"
git_sync https://github.com/moojink/openvla-oft.git "$W/openvla-oft" "$OFT_REF"
"$PY" -m pip install -e "$W/openvla-oft"
"$PY" -m pip install packaging ninja
"$PY" -c "import flash_attn" 2>/dev/null || PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH" "$PY" -m pip install flash-attn --no-build-isolation \
  || warn "flash-attn build failed (needs CUDA 12.x nvcc matching torch cu124)"
git_sync https://github.com/Lifelong-Robot-Learning/LIBERO.git "$W/LIBERO" "$LIBERO_REF"
"$PY" -m pip install -e "$W/LIBERO"
"$PY" -m pip install -r "$W/openvla-oft/experiments/robot/libero/libero_requirements.txt"
libero_config "$PY" simplevla   # writes LIBERO's default config; export LIBERO_CONFIG_PATH=$RSP_HOME/config/libero/simplevla before its scripts
git_sync https://github.com/PRIME-RL/SimpleVLA-RL.git "$W/SimpleVLA-RL" "$SIMPLEVLA_REF"
"$PY" -c "import torch, verl; print('torch', torch.__version__, '| verl ok')"
ok "SimpleVLA-RL workspace: $W (edit examples/run_openvla_oft_rl_libero.sh: NUM_GPUS=1, SFT_MODEL_PATH, CKPT_PATH; set trainer.val_only=True for eval)"
