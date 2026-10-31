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
# A real EGL error at the end of a log still gets the EGL hint.
printf 'Traceback (most recent call last):\nRuntimeError: eglInitialize() failed\n' > "$TMP/egl.log"
check "real EGL error still hinted"         "$(RSP_FORCE_WSL=0 failure_hint act_aloha "$TMP/egl.log" 1 | cut -c1-19)" "OpenGL/EGL problem:"

# --- metric (play.py release/3.0.0 prints 'Successful trials: K, out of N trials' / 'Success rate: x') ---
sed "s|@RUNTMP@|$TMP|g" "$T/fixtures/isaac_mimic.log" > "$TMP/play.log"
check "metric: play success rate"           "$(extract_metric isaac_mimic "$TMP/play.log")" "play success 40.0% (2/5 rollouts)"
check "metric: trained, no play"            "$(extract_metric isaac_mimic "$F")" "trained (epoch 20), no play result"
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

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
