#!/usr/bin/env bash
# Isaac Lab Mimic imitation-learning pipeline (release/3.0.0 docs: features/imitation-learning/teleop_imitation.rst):
#   1) download the official Franka cube-stacking human dataset (skip teleop)   2) annotate subtasks (--auto)
#   3) generate synthetic demos with Mimic   4) train robomimic BC-RNN   5) evaluate rollouts
# Uses the EXISTING Isaac env. Requires the 'isaacsim' + 'mimic' extras already installed there. This script will
# NOT install them unless ALLOW_ISAAC_ENV_CHANGES=1 (which runs `isaaclab.sh -i mimic` per the docs).
#   STAGE=all|annotate|generate|train|play   N_GEN=1000   GEN_ENVS=100   EPOCHS=1000
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
source "$(dirname "$0")/../lib/isaaclab_env.sh"
: "${STAGE:=all}"; : "${N_GEN:=1000}"; : "${GEN_ENVS:=100}"; : "${EPOCHS:=1000}"
RUN="$(dirname "$0")/../lib/isaaclab_run.sh"
isaaclab_detect; [ "$ISAACLAB_MODE" != none ] || die "Isaac Lab not detected (set ISAACLAB_PATH)"
D="$RSP_DATA/isaaclab_mimic"; mkdir -p "$D"
URL="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5"

if ! "$RUN" python -c "import isaaclab_mimic, robomimic" >/dev/null 2>&1; then
  if [ "${ALLOW_ISAAC_ENV_CHANGES:-0}" = 1 ]; then
    warn "Installing Isaac Lab 'mimic' extra into the Isaac env (you allowed changes)"
    (cd "$ISAACLAB_PATH" && ./isaaclab.sh -i mimic)
  else
    die "isaaclab_mimic/robomimic not importable in the Isaac env. Install them yourself (docs: './isaaclab.sh -i mimic' or 'uv sync --inexact --extra mimic'), or rerun with ALLOW_ISAAC_ENV_CHANGES=1."
  fi
fi

if [ ! -s "$D/dataset.hdf5" ]; then log "Downloading human demo dataset"; wget -c -O "$D/dataset.hdf5" "$URL"; fi
s() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
if s annotate; then
  "$RUN" python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --auto --input_file "$D/dataset.hdf5" --output_file "$D/annotated_dataset.hdf5"
fi
if s generate; then
  gpu_monitor_start "$D/gpu_generate.csv"
  "$RUN" python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --num_envs 20 --generation_num_trials 10 \
    --input_file "$D/annotated_dataset.hdf5" --output_file "$D/generated_dataset_small.hdf5"
  "$RUN" python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --num_envs "$GEN_ENVS" --generation_num_trials "$N_GEN" \
    --input_file "$D/annotated_dataset.hdf5" --output_file "$D/generated_dataset.hdf5"
  echo "Peak GPU memory (generate): $(gpu_monitor_stop) MiB"
fi
if s train; then
  "$RUN" python scripts/imitation_learning/robomimic/train.py --task IsaacContrib-Stack-Cube-Franka-IK-Rel --algo bc \
    --epochs "$EPOCHS" --dataset "$D/generated_dataset.hdf5"
  ok "robomimic models are saved under \$ISAACLAB_PATH/logs/robomimic"
fi
if s play; then
  CK="${CHECKPOINT:-$(ls -t "$ISAACLAB_PATH"/logs/robomimic/*/*/models/*.pth 2>/dev/null | head -n1 || true)}"
  [ -n "$CK" ] || die "No robomimic checkpoint found; set CHECKPOINT=/path/model.pth"
  "$RUN" python scripts/imitation_learning/robomimic/play.py --task IsaacContrib-Stack-Cube-Franka-IK-Rel \
    --num_rollouts 50 --checkpoint "$CK"
fi
ok "Mimic pipeline stage '$STAGE' done. Data in $D"
