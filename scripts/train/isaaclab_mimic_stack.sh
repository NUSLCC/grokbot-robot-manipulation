#!/usr/bin/env bash
# Isaac Lab Mimic imitation-learning pipeline (release/3.0.0 docs: features/imitation-learning/teleop_imitation.rst):
#   1) download the official Franka cube-stacking human dataset (skip teleop)   2) annotate subtasks (--auto)
#   3) generate synthetic demos with Mimic   4) train robomimic BC-RNN   5) evaluate rollouts
# Uses the EXISTING Isaac env. Requires the 'isaacsim' + 'mimic' extras already installed there. This script will
# NOT install them unless ALLOW_ISAAC_ENV_CHANGES=1 (which runs `isaaclab.sh -i mimic` per the docs).
#   STAGE=all|annotate|generate|train|play   N_GEN=1000   GEN_ENVS=100   EPOCHS=1000
#   REUSE_DATA=1 (default): STAGE=all skips annotate+generate when generated_dataset.hdf5 already opens and holds
#                >= N_GEN demos (a rerun after a train/play failure then starts at training). REUSE_DATA=0 regenerates.
#   MIMIC_EXP_NAME: robomimic experiment name (default bc_rnn_low_dim_franka_stack-<timestamp>, unique per run, so
#                robomimic never asks "model directory already exists! overwrite? (y/n)", which headless runs cannot
#                answer). If that folder exists anyway, it is moved aside to <dir>.bak-<timestamp>, never deleted.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
source "$(dirname "$0")/../lib/isaaclab_env.sh"
isaac_skip_if_requested; isaaclab_wsl_note
# Headless: ISAACLAB_HEADLESS_FLAGS (default "--viz none --livestream 0") go to annotate/generate/play
# (they use Isaac Lab launcher args). robomimic/train.py is plain PyTorch and takes no launcher flags.
: "${STAGE:=all}"; : "${N_GEN:=1000}"; : "${GEN_ENVS:=100}"; : "${EPOCHS:=1000}"
MIMIC_TASK=IsaacContrib-Stack-Cube-Franka-IK-Rel   # robomimic task name (train + play)
case "$STAGE" in all|annotate|generate|train|play) ;; *) die "STAGE must be all|annotate|generate|train|play (got '$STAGE')" ;; esac
log "Mimic stage: $STAGE"
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

if [ "$STAGE" = play ]; then :   # play needs only the checkpoint, not the datasets
elif [ ! -s "$D/dataset.hdf5" ]; then log "Downloading human demo dataset"; wget -c -O "$D/dataset.hdf5" "$URL"; fi
REUSED=0
s() { if [ "$REUSED" = 1 ]; then [ "$1" = train ] || [ "$1" = play ]; else [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; fi; }
# demo count of an Isaac Lab HDF5 dataset (data/demo_N groups); empty if missing, unreadable or partial
demo_count() { [ -s "$1" ] || return 0; "$RUN" python -c 'import sys,h5py; f=h5py.File(sys.argv[1],"r"); print(len(f["data"].keys()))' "$1" 2>/dev/null | tail -n1 | grep -E '^[0-9]+$' || true; }
GEN_FILE="$D/generated_dataset.hdf5"
if [ "$STAGE" = all ] && [ "${REUSE_DATA:-1}" = 1 ]; then
  n="$(demo_count "$GEN_FILE")"
  if [ -n "$n" ] && [ "$n" -ge "$N_GEN" ]; then
    ok "Reusing $GEN_FILE ($n demos >= N_GEN=$N_GEN): skipping annotate + generate (REUSE_DATA=0 regenerates)"
    REUSED=1   # s() is now true only for train + play
  fi
fi
if s annotate; then
  "$RUN" python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
    --auto --input_file "$D/dataset.hdf5" --output_file "$D/annotated_dataset.hdf5" $ISAACLAB_HEADLESS_FLAGS
fi
if s generate; then
  gpu_monitor_start "$D/gpu_generate.csv"
  "$RUN" python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --num_envs 20 --generation_num_trials 10 \
    --input_file "$D/annotated_dataset.hdf5" --output_file "$D/generated_dataset_small.hdf5" $ISAACLAB_HEADLESS_FLAGS
  "$RUN" python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --num_envs "$GEN_ENVS" --generation_num_trials "$N_GEN" \
    --input_file "$D/annotated_dataset.hdf5" --output_file "$D/generated_dataset.hdf5" $ISAACLAB_HEADLESS_FLAGS
  echo "Peak GPU memory (generate): $(gpu_monitor_stop) MiB"
fi
if s train; then
  # train.py: output = $ISAACLAB_PATH/logs/robomimic/<task>/<--name>/<timestamp>/models/*.pth. robomimic 0.4.0's
  # get_exp_dir() calls input() when <--name> exists (EOFError headless). A unique name avoids it; an existing folder
  # is moved aside, never removed. stdin is /dev/null so any other prompt fails fast instead of hanging.
  EXP_ROOT="$ISAACLAB_PATH/logs/robomimic/$MIMIC_TASK"; ts="$(date +%Y%m%d-%H%M%S)"
  if [ -n "${MIMIC_EXP_NAME:-}" ]; then EXP="$MIMIC_EXP_NAME"
  else EXP="bc_rnn_low_dim_franka_stack-$ts"; i=1; while [ -e "$EXP_ROOT/$EXP" ]; do EXP="bc_rnn_low_dim_franka_stack-$ts-$i"; i=$((i+1)); done; fi
  EXP_DIR="$EXP_ROOT/$EXP"
  if [ -e "$EXP_DIR" ]; then
    bak="$EXP_DIR.bak-$ts"; i=1; while [ -e "$bak" ]; do bak="$EXP_DIR.bak-$ts-$i"; i=$((i+1)); done
    mv -T "$EXP_DIR" "$bak"; warn "robomimic folder existed: moved it to $bak (not deleted)"
  fi
  TLOG="$(mktemp)"
  "$RUN" python scripts/imitation_learning/robomimic/train.py --task "$MIMIC_TASK" --algo bc --name "$EXP" \
    --epochs "$EPOCHS" --dataset "$GEN_FILE" < /dev/null 2>&1 | tee "$TLOG"
  # train.py catches training errors, prints "run failed with error" and still exits 0: treat that as a failure.
  if grep -q 'run failed with error' "$TLOG"; then rm -f "$TLOG"; die "robomimic training failed (see 'run failed with error' above)"; fi
  rm -f "$TLOG"
  NEW_CK="$(find_robomimic_ckpt "$ISAACLAB_PATH/logs/robomimic/$MIMIC_TASK/$EXP")"
  [ -n "$NEW_CK" ] || die "robomimic training wrote no model .pth under $EXP_DIR"
  ok "robomimic checkpoint (this run): $NEW_CK"
  : "${CHECKPOINT:=$NEW_CK}"   # play below uses this run's checkpoint, never an older one
fi
if s play; then
  # robomimic saves to logs/robomimic/<task>/<experiment>/<timestamp>/models/model_epoch_N.pth (depth varies by
  # version), so find_robomimic_ckpt searches at any depth, newest first, preferring this task's folder.
  CK="${CHECKPOINT:-$(find_robomimic_ckpt "$ISAACLAB_PATH/logs/robomimic" "$MIMIC_TASK")}"
  [ -n "$CK" ] || die "No robomimic checkpoint found under $ISAACLAB_PATH/logs/robomimic; set CHECKPOINT=/path/model.pth"
  [ -f "$CK" ] || die "No robomimic checkpoint found: CHECKPOINT=$CK does not exist"
  ok "robomimic checkpoint: $CK"
  # play.py prints 'Successful trials: K, out of N trials' and 'Success rate: <0..1>' (the runner's metric).
  "$RUN" python scripts/imitation_learning/robomimic/play.py --task "$MIMIC_TASK" \
    --num_rollouts "${N_ROLLOUTS:-50}" --checkpoint "$CK" $ISAACLAB_HEADLESS_FLAGS
fi
ok "Mimic pipeline stage '$STAGE' done. Data in $D"
