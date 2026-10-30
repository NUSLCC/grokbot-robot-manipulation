#!/usr/bin/env bash
# 14_install_maniskill.sh - ManiSkill3 (SAPIEN, GPU-parallel sim) + its PPO / Diffusion Policy baselines.
# Env: $RSP_HOME/envs/maniskill (uv venv, Python 3.11 - our choice; mani_skill requires >=3.9).
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv

ENV="$RSP_ENVS/maniskill"; SRC="$RSP_SRC/ManiSkill"
[ -x "$ENV/bin/python" ] || uv venv --python "$MANISKILL_PY" "$ENV"
PY="$ENV/bin/python"
uv pip install --python "$PY" --torch-backend cu128 "mani_skill==$MANISKILL_VERSION" torch
uv pip install --python "$PY" tyro tensorboard wandb  # used by examples/baselines/ppo
git_sync https://github.com/haosulab/ManiSkill.git "$SRC" "v$MANISKILL_VERSION"   # baselines live in the repo

[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -f /etc/vulkan/icd.d/nvidia_icd.json ] || warn "NVIDIA Vulkan ICD missing; ManiSkill rendering will fail"
log "Smoke test: random actions in PickCube-v1"
"$PY" -m mani_skill.examples.demo_random_action -e PickCube-v1 || warn "demo failed; check Vulkan (vulkaninfo) - see README troubleshooting"
"$PY" -c "import mani_skill, torch; print('mani_skill', mani_skill.__version__, '| torch', torch.__version__, torch.cuda.is_available())"
ok "ManiSkill env: $ENV ; baselines: $SRC/examples/baselines"
