#!/usr/bin/env bash
# Isaac Lab Mimic: checkpoint finder (robomimic's 3-level layout), failure hint for the user's real "no checkpoint"
# log (must not be the OpenGL/EGL hint), play.py success-rate metric, and MIMIC_STAGE=play passthrough, including
# the real scripts/train/isaaclab_mimic_stack.sh run against a fake Isaac Lab checkout.
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; K="$(cd "$T/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export PATH="$T/fake-bin:$PATH" RSP_HOME="$TMP/rsp"
source "$K/scripts/lib/common.sh"; source "$K/scripts/lib/results.sh"
set +e
nfail=0
check() { if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: expected '$3', got '$2'"; nfail=$((nfail+1)); fi; }
TASK=IsaacContrib-Stack-Cube-Franka-IK-Rel

# --- checkpoint finder ---
R="$TMP/IsaacLab/logs/robomimic"
check "finder: no logs dir -> empty"       "$(find_robomimic_ckpt "$R" "$TASK")" ""
old="$R/$TASK/bc_rnn_low_dim_franka_stack/20261031120000/models/model_epoch_10.pth"
new="$R/$TASK/bc_rnn_low_dim_franka_stack/20261101003144/models/model_epoch_20.pth"
other="$R/OtherTask/exp/20261102000000/models/model_epoch_99.pth"
mkdir -p "$(dirname "$old")" "$(dirname "$new")" "$(dirname "$other")"
touch -d '2026-10-31 12:00' "$old"; touch -d '2026-11-01 00:31' "$new"; touch -d '2026-11-02 00:00' "$other"
touch -d '2026-11-03 00:00' "$R/$TASK/bc_rnn_low_dim_franka_stack/20261101003144/config.json"   # not a model
check "finder: 3-level layout, newest in task dir" "$(find_robomimic_ckpt "$R" "$TASK")" "$new"
check "finder: old glob would have missed it" "$(ls "$R"/*/*/models/*.pth 2>/dev/null | wc -l)" 0
check "finder: no task -> newest anywhere"  "$(find_robomimic_ckpt "$R")" "$other"
check "finder: unknown task -> falls back"  "$(find_robomimic_ckpt "$R" NoSuchTask)" "$other"
sp="$TMP/with space/logs/robomimic/$TASK/e/1/models/model_epoch_1.pth"; mkdir -p "$(dirname "$sp")"; touch "$sp"
check "finder: path with a space"           "$(find_robomimic_ckpt "$TMP/with space/logs/robomimic" "$TASK")" "$sp"

# --- hint for the user's real failing tail (dual 3090, 2026-11-01) ---
F="$T/fixtures/isaac_mimic.rc1.nockpt.log"
h="$(RSP_FORCE_WSL=0 failure_hint isaac_mimic "$F" 1)"
check "nockpt hint names the checkpoint"    "$(cut -c1-59 <<<"$h")" "Mimic trained but the play stage found no checkpoint (.pth)"
check "nockpt hint gives MIMIC_STAGE=play"  "$(grep -c 'MIMIC_STAGE=play' <<<"$h")" 1
check "nockpt hint is not OpenGL/EGL"       "$(grep -ci 'OpenGL\|osmesa' <<<"$h")" 0
# Without the [FAIL] line, the benign Kit display warning / egl-probe lines alone must not give the EGL hint.
grep -v 'No robomimic checkpoint' "$F" > "$TMP/benign.log"
check "Kit 'default display' warning is not an EGL error" "$(RSP_FORCE_WSL=0 failure_hint isaac_mimic "$TMP/benign.log" 1 | grep -ci 'OpenGL\|osmesa')" 0
check "other kit [FAIL] line is quoted"     "$(printf 'x\n[FAIL] something else broke\n' > "$TMP/f.log"; RSP_FORCE_WSL=0 failure_hint isaac_mimic "$TMP/f.log" 1 | cut -c1-40)" "Isaac error: something else broke; see t"
# The user's real culprit: Kit's "Failed to startup plugin carb.windowing-glfw.plugin (... IGLContext ...)" warning,
# early in the log and within its last 80 lines. The old matcher (GLContext) turned it into the OpenGL/EGL hint.
check "fixture has the real glfw line early + late" "$(grep -c 'Failed to startup plugin carb.windowing-glfw.plugin' "$F") $(head -n 10 "$F" | grep -qc 'IGLContext' && echo early) $(tail -n 3 "$F" | grep -qc 'IGLContext' && echo late)" "6 early late"
check "glfw warning alone: no EGL hint"     "$(grep -E 'windowing-glfw' "$F" > "$TMP/glfw.log"; RSP_FORCE_WSL=0 failure_hint isaac_mimic "$TMP/glfw.log" 1 | grep -ci 'OpenGL\|osmesa')" 0
if [ -n "${OLD_RESULTS:-}" ] && [ -f "$OLD_RESULTS" ]; then   # optional: OLD_RESULTS=<old results.sh> shows the regression
  check "old matcher gave the wrong hint" "$(bash -c 'source "$1"; source "$2"; set +e; RSP_FORCE_WSL=0 failure_hint isaac_mimic "$3" 1' _ "$K/scripts/lib/common.sh" "$OLD_RESULTS" "$F" | cut -c1-19)" "OpenGL/EGL problem:"
fi
# A real EGL error at the end of a log still gets the EGL hint.
printf 'Traceback (most recent call last):\nRuntimeError: eglInitialize() failed\n' > "$TMP/egl.log"
check "real EGL error still hinted"         "$(RSP_FORCE_WSL=0 failure_hint act_aloha "$TMP/egl.log" 1 | cut -c1-19)" "OpenGL/EGL problem:"

# --- metric (play.py release/3.0.0 prints 'Successful trials: K, out of N trials' / 'Success rate: x') ---
sed "s|@RUNTMP@|$TMP|g" "$T/fixtures/isaac_mimic.log" > "$TMP/play.log"
check "metric: play success rate"           "$(extract_metric isaac_mimic "$TMP/play.log")" "play success 40.0% (2/5 rollouts)"
check "metric: trained, no play"            "$(extract_metric isaac_mimic "$F")" "trained (epoch 20), no play result"
# The user's real QUICK play result on the dual 3090 (2026-11-01 12:10 SGT: PASS, 147s, 2668 MiB, 0/5); the
# play.py lines follow release/3.0.0's print format, the Kit lines are the user's.
check "metric: real QUICK play 0/5"         "$(extract_metric isaac_mimic "$T/fixtures/isaac_mimic.play0.log")" "play success 0.0% (0/5 rollouts)"
printf 'Success rate: 1.0\n' > "$TMP/p1.log"
check "metric: rate only"                   "$(extract_metric isaac_mimic "$TMP/p1.log")" "play success 100.0%"

# --- runner passthrough: MIMIC_STAGE / STAGE reach the step command ---
for v in "MIMIC_STAGE=play" "STAGE=play"; do
  rm -rf "$TMP/runs"
  env RSP_DRY_RUN=1 RSP_DRY_RUN_TMP="$TMP" RUNS_DIR="$TMP/runs" RSP_FORCE_WSL=0 QUICK=1 RUN_MIMIC=1 ONLY_STEPS=isaac_mimic "$v" \
    bash "$K/scripts/run_wsl_headless.sh" > "$TMP/run.txt" 2>&1; rc=$?
  L="$(ls "$TMP"/runs/*/06_isaac_mimic.log)"; S="$(ls "$TMP"/runs/*/SUMMARY.md)"
  check "runner $v: exit 0"                 "$rc" 0
  check "runner $v: STAGE=play in command"  "$(grep -c '^\[dry-run\] cmd: env .*N_ROLLOUTS=5 STAGE=play bash .*isaaclab_mimic_stack.sh' "$L")" 1
  check "runner $v: metric in SUMMARY"      "$(grep -E '^\| isaac_mimic \|' "$S" | awk -F'|' '{gsub(/^ +| +$/,"",$6); print $6}')" "play success 40.0% (2/5 rollouts)"
done
rm -rf "$TMP/runs"
env RSP_DRY_RUN=1 RSP_DRY_RUN_TMP="$TMP" RUNS_DIR="$TMP/runs" RSP_FORCE_WSL=0 QUICK=1 RUN_MIMIC=1 ONLY_STEPS=isaac_mimic \
  bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1
check "runner default: STAGE=all"           "$(grep -c 'STAGE=all bash' "$(ls "$TMP"/runs/*/06_isaac_mimic.log)")" 1

# --- the real mimic script, STAGE=play, against a fake Isaac Lab checkout + fake python (records its argv) ---
IL="$TMP/IsaacLab"; echo 3.0.0 > "$IL/VERSION"; touch "$IL/isaaclab.sh"
cat > "$TMP/fakepy" <<'PY'
#!/usr/bin/env bash
case "$*" in *"import isaaclab_mimic"*) exit 0 ;; esac
echo "fakepy $*" >> "$FAKEPY_ARGS"
case "$*" in *robomimic/play.py*) printf 'Successful trials: 3, out of 5 trials\nSuccess rate: 0.6\n' ;; esac
PY
chmod +x "$TMP/fakepy"
run_mimic() { env -u STAGE ISAACLAB_PATH="$IL" ISAACLAB_MODE=python ISAAC_PY="$TMP/fakepy" FAKEPY_ARGS="$TMP/args" RSP_FORCE_WSL=0 "$@" \
  bash "$K/scripts/train/isaaclab_mimic_stack.sh" > "$TMP/mimic.log" 2>&1; }
: > "$TMP/args"; run_mimic STAGE=play N_ROLLOUTS=5; rc=$?
check "mimic STAGE=play: exit 0"            "$rc" 0
check "mimic play: one python call (play)"  "$(wc -l < "$TMP/args")" 1
check "mimic play: newest task checkpoint"  "$(grep -c -- "--checkpoint $new" "$TMP/args")" 1
check "mimic play: headless flags"          "$(grep -c -- '--viz none --livestream 0' "$TMP/args")" 1
check "mimic play: no dataset download"     "$(grep -c 'Downloading' "$TMP/mimic.log")" 0
check "mimic play: metric"                  "$(extract_metric isaac_mimic "$TMP/mimic.log")" "play success 60.0% (3/5 rollouts)"
: > "$TMP/args"; run_mimic STAGE=play CHECKPOINT="$old"
check "mimic play: CHECKPOINT= wins"        "$(grep -c -- "--checkpoint $old" "$TMP/args")" 1
run_mimic STAGE=play CHECKPOINT="$TMP/nope.pth"; rc=$?
check "mimic play: missing CHECKPOINT fails" "$rc" 1
rm -rf "$IL/logs"; run_mimic STAGE=play; rc=$?
check "mimic play: no checkpoint -> rc 1"   "$rc" 1
check "mimic play: no checkpoint -> hint"   "$(failure_hint isaac_mimic "$TMP/mimic.log" 1 | cut -c1-20)" "Mimic trained but th"
run_mimic STAGE=bogus; rc=$?
check "mimic: bad STAGE rejected"           "$rc" 1

# --- the user's real full-run failure (dual 3090, 2026-11-01 13:07 SGT): robomimic's overwrite prompt -> EOFError ---
E="$T/fixtures/isaac_mimic.rc1.eof_overwrite.log"
h="$(RSP_FORCE_WSL=0 failure_hint isaac_mimic "$E" 1)"
check "overwrite hint names robomimic's folder prompt" "$(cut -c1-64 <<<"$h")" "robomimic asked whether to overwrite a model folder left by an e"
check "overwrite hint: not OpenGL/EGL"      "$(grep -ci 'OpenGL\|osmesa' <<<"$h")" 0
check "other EOFError keeps the generic hint" "$(printf 'x = input()\nEOFError: EOF when reading a line\n' > "$TMP/eof.log"; RSP_FORCE_WSL=0 failure_hint isaac_mimic "$TMP/eof.log" 1 | cut -c1-30)" "a tool asked an interactive qu"

# --- train stage against a fake python that behaves like robomimic 0.4.0 get_exp_dir (prompts if <name> exists) ---
cat > "$TMP/fakepy2" <<'PY'
#!/usr/bin/env bash
case "$*" in *"import isaaclab_mimic"*) exit 0 ;; esac
case "$*" in *h5py*) [ -n "${FAKE_DEMOS:-}" ] && echo "$FAKE_DEMOS" || { echo "OSError: unable to open file" >&2; exit 1; }; exit 0 ;; esac
echo "fakepy $*" >> "$FAKEPY_ARGS"
case "$*" in
  *robomimic/train.py*)
    name=bc_rnn_low_dim_franka_stack; a=("$@")
    for i in "${!a[@]}"; do [ "${a[$i]}" = --name ] && name="${a[$((i+1))]}"; done
    task=IsaacContrib-Stack-Cube-Franka-IK-Rel; base="logs/robomimic/$task/$name"
    if [ -e "$base" ]; then
      printf 'WARNING: model directory (%s) already exists! \noverwrite? (y/n)\n' "$PWD/$base"
      read -r ans || { echo "EOFError: EOF when reading a line"; exit 1; }
      [ "$ans" = y ] && rm -rf "$base"
    fi
    [ -n "${FAKE_TRAIN_FAIL:-}" ] && { echo "run failed with error:"; echo "boom"; exit 0; }
    mkdir -p "$base/20261101130000/models"; touch "$base/20261101130000/models/model_epoch_20.pth"
    echo "save checkpoint to $PWD/$base/20261101130000/models/model_epoch_20.pth"; echo "finished run successfully!" ;;
  *robomimic/play.py*) printf 'Successful trials: 1, out of 5 trials\nSuccess rate: 0.2\n' ;;
esac
PY
chmod +x "$TMP/fakepy2"
run_mimic2() { env -u STAGE -u CHECKPOINT ISAACLAB_PATH="$IL" ISAACLAB_MODE=python ISAAC_PY="$TMP/fakepy2" FAKEPY_ARGS="$TMP/args" \
  RSP_FORCE_WSL=0 N_GEN=20 EPOCHS=20 N_ROLLOUTS=5 "$@" \
  bash "$K/scripts/train/isaaclab_mimic_stack.sh" > "$TMP/mimic.log" 2>&1; }
mkdir -p "$RSP_HOME/data/isaaclab_mimic"; echo x > "$RSP_HOME/data/isaaclab_mimic/dataset.hdf5"
TD="$IL/logs/robomimic/$TASK"
mkdir -p "$TD/bc_rnn_low_dim_franka_stack/20261101003144/models"; echo old > "$TD/bc_rnn_low_dim_franka_stack/20261101003144/models/model_epoch_20.pth"
# A future-dated older checkpoint: newest by mtime, but play must still use THIS run's checkpoint.
mkdir -p "$TD/zzz/1/models"; touch -d '2030-01-01' "$TD/zzz/1/models/model_epoch_1.pth"
# Reproduce the bug first: the old default name (no --name) hits the prompt and fails with EOFError.
( cd "$IL" && FAKEPY_ARGS=/dev/null "$TMP/fakepy2" scripts/imitation_learning/robomimic/train.py --task "$TASK" < /dev/null ) > "$TMP/old.txt" 2>&1
check "fake robomimic reproduces the EOFError"  "$(grep -c 'EOFError: EOF when reading a line' "$TMP/old.txt")" 1
# Rerun with the user's leftovers: generated_dataset.hdf5 holding 1000 demos + the old model folder.
echo x > "$RSP_HOME/data/isaaclab_mimic/generated_dataset.hdf5"
: > "$TMP/args"; run_mimic2 FAKE_DEMOS=1000; rc=$?
check "rerun: exit 0 (no prompt)"           "$rc" 0
check "rerun: no EOFError"                  "$(grep -c EOFError "$TMP/mimic.log")" 0
check "rerun: data reused, no annotate/generate" "$(grep -c 'annotate_demos\|generate_dataset' "$TMP/args")" 0
check "rerun: reuse is logged"              "$(grep -c 'Reusing .*generated_dataset.hdf5 (1000 demos >= N_GEN=20)' "$TMP/mimic.log")" 1
check "rerun: unique --name"                "$(grep -cE -- '--name bc_rnn_low_dim_franka_stack-[0-9]{8}-[0-9]{6} ' "$TMP/args")" 1
check "rerun: old model folder kept"        "$(cat "$TD/bc_rnn_low_dim_franka_stack/20261101003144/models/model_epoch_20.pth")" old
check "rerun: play uses this run's ckpt"    "$(grep 'play.py' "$TMP/args" | grep -cE -- '--checkpoint .*/bc_rnn_low_dim_franka_stack-[0-9]{8}-[0-9]{6}/20261101130000/models/model_epoch_20.pth')" 1
check "rerun: metric"                       "$(extract_metric isaac_mimic "$TMP/mimic.log")" "play success 20.0% (1/5 rollouts)"
check "rerun: later STAGE=play finds a new-layout ckpt" "$(find_robomimic_ckpt "$IL/logs/robomimic" "$TASK" | grep -c 'zzz/1/models')" 1
# Too few demos, unreadable file, or REUSE_DATA=0 -> annotate + generate run again.
: > "$TMP/args"; run_mimic2 FAKE_DEMOS=5
check "5 demos < N_GEN: regenerates"        "$(grep -c 'annotate_demos\|generate_dataset' "$TMP/args")" 3
: > "$TMP/args"; run_mimic2
check "unreadable dataset: regenerates"     "$(grep -c 'generate_dataset' "$TMP/args")" 2
: > "$TMP/args"; run_mimic2 FAKE_DEMOS=1000 REUSE_DATA=0
check "REUSE_DATA=0: regenerates"           "$(grep -c 'annotate_demos\|generate_dataset' "$TMP/args")" 3
: > "$TMP/args"; run_mimic2 FAKE_DEMOS=1000 STAGE=train; run_mimic2 FAKE_DEMOS=1000 STAGE=train
check "same-second reruns get distinct names" "$(grep -oE -- '--name [^ ]+' "$TMP/args" | sort -u | wc -l)" 2
check "STAGE=train x2: trains twice, no play" "$(grep -c "robomimic/train.py" "$TMP/args") $(grep -c "play.py" "$TMP/args")" "2 0"
# A fixed MIMIC_EXP_NAME that already exists is moved aside (not deleted), then training succeeds.
mkdir -p "$TD/mine/1/models"; echo keep > "$TD/mine/1/models/model_epoch_9.pth"
: > "$TMP/args"; run_mimic2 FAKE_DEMOS=1000 MIMIC_EXP_NAME=mine; rc=$?
check "MIMIC_EXP_NAME existing: exit 0"     "$rc" 0
check "MIMIC_EXP_NAME existing: moved to .bak" "$(cat "$TD"/mine.bak-*/1/models/model_epoch_9.pth)" keep
mkdir -p "$TD/mine/2"; run_mimic2 FAKE_DEMOS=1000 MIMIC_EXP_NAME=mine
check "MIMIC_EXP_NAME twice in a second: 2 backups" "$(ls -d "$TD"/mine.bak-* | wc -l)" 2
check "MIMIC_EXP_NAME: fresh ckpt"          "$(ls "$TD/mine/20261101130000/models/")" model_epoch_20.pth
# train.py swallows training errors (prints 'run failed with error', exits 0): the step must fail, no play.
: > "$TMP/args"; run_mimic2 FAKE_DEMOS=1000 FAKE_TRAIN_FAIL=1; rc=$?
check "train 'run failed with error' -> rc 1" "$rc" 1
check "train failure: play not run"         "$(grep -c 'play.py' "$TMP/args")" 0

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
