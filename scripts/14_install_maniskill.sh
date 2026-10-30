#!/usr/bin/env bash
# 14_install_maniskill.sh - ManiSkill3 (SAPIEN, GPU-parallel sim) + its PPO / Diffusion Policy baselines.
# Env: $RSP_HOME/envs/maniskill (uv venv, Python 3.11 - our choice; mani_skill requires >=3.9).
# torch comes from the $TORCH_CUDA index (cu128 default: covers Ampere, Ada and Blackwell).
# Rendering needs a real Vulkan GPU device (not on WSL2 today); the smoke test then falls back to no-render
# state mode, which is what train/maniskill_ppo.sh uses there (MS_RENDER=auto).
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv

ENV="$RSP_ENVS/maniskill"; SRC="$RSP_SRC/ManiSkill"
[ -x "$ENV/bin/python" ] || uv venv --python "$MANISKILL_PY" "$ENV"
PY="$ENV/bin/python"
uv pip install --python "$PY" --torch-backend "$TORCH_CUDA" "mani_skill==$MANISKILL_VERSION" torch
uv pip install --python "$PY" tyro tensorboard wandb  # used by examples/baselines/ppo
git_sync https://github.com/haosulab/ManiSkill.git "$SRC" "v$MANISKILL_VERSION"   # baselines live in the repo

[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -f /etc/vulkan/icd.d/nvidia_icd.json ] || warn "NVIDIA Vulkan ICD missing; ManiSkill rendering will fail$(is_wsl && echo ' (normal on WSL2: no-render state mode is used instead)')"
log "Smoke test: random actions in PickCube-v1"
if ! "$PY" -m mani_skill.examples.demo_random_action -e PickCube-v1; then
  warn "rendering demo failed; check Vulkan (vulkaninfo) - see TROUBLESHOOTING.md. Trying the no-render state mode:"
  "$PY" "$RSP_KIT_DIR/scripts/lib/maniskill_probe.py" none && ok "no-render state mode works (train/maniskill_ppo.sh will use it)" \
    || warn "no-render state mode also failed; see the error above"
fi
"$PY" -c "import mani_skill, torch; print('mani_skill', mani_skill.__version__, '| torch', torch.__version__, torch.cuda.is_available())"
ok "ManiSkill env: $ENV ; baselines: $SRC/examples/baselines"
