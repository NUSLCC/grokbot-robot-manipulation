#!/usr/bin/env bash
# RLinf quickstart (docs: start/vla.rst): PPO RL fine-tuning of OpenVLA-OFT on ManiSkill3.
# Steps 1-2 run automatically; step 3 (YAML path edits) must be done by you, then rerun with CONFIG_EDITED=1.
# Single-RTX-4090 feasibility for a 7B actor + rollout is UNVERIFIED (upstream examples target multi-GPU).
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env
SRC="$RSP_SRC/RLinf"; VENV="$RSP_ENVS/rlinf-openvla-oft"; M="$RSP_CKPT/rlinf"
[ -x "$VENV/bin/python" ] || die "Run RLINF_MODEL=openvla-oft scripts/17_install_rlinf.sh first"
HF="$VENV/bin/hf"; mkdir -p "$M"
[ -d "$M/RLinf-OpenVLAOFT-ManiSkill-Base-Main" ] || "$HF" download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Main --local-dir "$M/RLinf-OpenVLAOFT-ManiSkill-Base-Main"
[ -d "$M/oft-sft/lora_004000" ] || "$HF" download RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora --local-dir "$M/oft-sft/lora_004000"
A="$SRC/rlinf/envs/sim/maniskill/assets"; [ -d "$A" ] || "$HF" download --repo-type dataset RLinf/maniskill_assets --local-dir "$A"
CFG="$SRC/examples/embodiment/config/maniskill_ppo_openvlaoft_quickstart.yaml"
if [ "${CONFIG_EDITED:-0}" != 1 ]; then
  cat <<MSG
Edit $CFG :
  rollout.model.model_path = $M/RLinf-OpenVLAOFT-ManiSkill-Base-Main
  actor.model.model_path   = $M/RLinf-OpenVLAOFT-ManiSkill-Base-Main
  actor.model.lora_path    = $M/oft-sft/lora_004000
  actor.model.is_lora      = True
  (and reduce env/rollout batch sizes / GPU placement for a single 24 GB GPU)
Then: CONFIG_EDITED=1 bash $0
MSG
  exit 0
fi
cd "$SRC"; source "$VENV/bin/activate"
gpu_monitor_start "$RSP_LOGS/rlinf_quickstart.gpu.csv"
bash examples/embodiment/run_embodiment.sh maniskill_ppo_openvlaoft_quickstart
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"
