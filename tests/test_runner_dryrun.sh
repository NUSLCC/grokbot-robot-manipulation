#!/usr/bin/env bash
# Control-flow test for scripts/run_wsl_headless.sh in RSP_DRY_RUN mode (fixture logs, fake nvidia-smi).
# Checks PASS/FAIL/SKIP handling, dependency skips, metric parsing, hints, SUMMARY.md and the exit code.
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; K="$(cd "$T/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export RSP_HOME="$TMP/rsp" RUNS_DIR="$TMP/runs" RSP_DRY_RUN=1 RSP_DRY_RUN_TMP="$TMP" QUICK=1 RSP_FORCE_WSL=1
export PATH="$T/fake-bin:$PATH" FAKE_GPU_USED=4321
mkdir -p "$TMP/eval_smolvla_libero_x"; echo '{"overall":{"pc_success":85.0}}' > "$TMP/eval_smolvla_libero_x/eval_info.json"
nfail=0
# Guard: the dry run replays tests/fixtures/<step>.log; a missing fixture silently turns every metric into n/a.
for f in act_aloha base_deps diffusion_pusht isaac_physx_smoke isaac_rl_franka isaac_validate lerobot_install \
         maniskill_install maniskill_ppo openpi_install openpi_pi05_libero_eval smolvla_libero_eval system_check; do
  if [ ! -s "$T/fixtures/$f.log" ]; then echo "FAIL - fixture missing: tests/fixtures/$f.log (is it git-ignored?)"; nfail=$((nfail+1)); fi
done
check() { if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: expected '$3', got '$2'"; nfail=$((nfail+1)); fi; }

RSP_DRY_RUN_RC="act_aloha=1 diffusion_pusht=137 openpi_install=2 maniskill_ppo=77" bash "$K/scripts/run_wsl_headless.sh" > "$TMP/out.txt" 2>&1
rc=$?
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
row() { grep -E "^\| $1 \|" "$S" | awk -F'|' '{gsub(/^ +| +$/,"",$3); print $3}'; }
met() { grep -E "^\| $1 \|" "$S" | awk -F'|' '{gsub(/^ +| +$/,"",$6); print $6}'; }
check "exit code = number of FAILs"   "$rc" 3
check "SUMMARY.md exists"             "$([ -s "$S" ] && echo y)" y
check "system_check PASS"             "$(row system_check)" PASS
check "act_aloha FAIL"                "$(row act_aloha)" FAIL
check "act metric parsed despite FAIL" "$(met act_aloha)" "loss 0.231, eval success 20.0%"
check "diffusion FAIL (rc 137)"       "$(row diffusion_pusht)" FAIL
check "smolvla metric from eval_info.json" "$(met smolvla_libero_eval)" "success 85.0%"
check "openpi_install FAIL"           "$(row openpi_install)" FAIL
check "openpi eval SKIP (dependency)" "$(row openpi_pi05_libero_eval)" SKIP
check "maniskill_ppo SKIP via rc 77"  "$(row maniskill_ppo)" SKIP
check "isaac_mimic optional SKIP"     "$(row isaac_mimic)" SKIP
check "isaac_rl metric"               "$(met isaac_rl_franka)" "mean reward 0.57, iter 30"
check "peak GPU from sampler"         "$(grep -E '^\| system_check \|' "$S" | awk -F'|' '{gsub(/ /,"",$5); print $5}')" 4321
check "Blackwell hint"                "$(grep -c 'no Blackwell (sm_120) kernels' "$S")" 1
check "RAM hint for rc 137"           "$(grep -c 'killed for RAM' "$S")" 1
check "console table printed"         "$(grep -c '^STEP  ' "$TMP/out.txt")" 1
check "per-step logs written"         "$(find "$RUNS_DIR" -name '*.log' | grep -c .)" 13

# All-pass run with SKIP_ISAAC=1 and maniskill no-render fixture
rm -rf "$RUNS_DIR"
SKIP_ISAAC=1 RSP_DRY_RUN_RC="" bash "$K/scripts/run_wsl_headless.sh" > "$TMP/out2.txt" 2>&1; rc=$?
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "all-pass exit 0"               "$rc" 0
check "SKIP_ISAAC skips isaac_validate" "$(row isaac_validate)" SKIP
check "SKIP_ISAAC skips isaac_rl"     "$(row isaac_rl_franka)" SKIP
check "maniskill no-render metric"    "$(met maniskill_ppo)" "eval success_once 0.8750, return 31.0625 (no-render fallback)"
check "openpi metric"                 "$(met openpi_pi05_libero_eval)" "success 90.0% of 10 eps"

# ONLY_STEPS
rm -rf "$RUNS_DIR"
ONLY_STEPS="system_check maniskill_ppo" bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "ONLY_STEPS runs selected"      "$(row maniskill_ppo)" PASS
check "ONLY_STEPS skips others"       "$(row act_aloha)" SKIP
# Native Linux (dual 3090, as on the user's workstation): interrupted dpkg in base_deps + failed Isaac smoke.
# Everything that depends on base_deps must be SKIP (transitively: installs SKIP -> their train/eval steps SKIP).
rm -rf "$RUNS_DIR"
RSP_FORCE_WSL=0 FAKE_GPU_NAME="NVIDIA GeForce RTX 3090" FAKE_GPU_CC=8.6 FAKE_GPU_COUNT=2 \
  RSP_DRY_RUN_RC="base_deps=100 isaac_validate=1" bash "$K/scripts/run_wsl_headless.sh" > "$TMP/out4.txt" 2>&1; rc=$?
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
hint() { grep -A3 "^### $1\$" "$S" | grep '^- Hint:' | sed 's/^- Hint: //'; }
check "native: exit code = 2 FAILs"       "$rc" 2
check "native: platform line"             "$(grep -c '^- Platform: native Linux' "$S")" 1
check "native: base_deps FAIL"            "$(row base_deps)" FAIL
check "native: dpkg hint"                 "$(hint base_deps)" "apt is broken: run sudo dpkg --configure -a, then rerun (rc=100)"
check "native: no Vulkan hint for apt log" "$(hint base_deps | grep -ci vulkan)" 0
check "native: lerobot_install SKIP"      "$(met lerobot_install)" "needs base_deps (FAILED)"
for st in act_aloha diffusion_pusht smolvla_libero_eval; do
  check "native: $st SKIP (skipped dep)"  "$(row $st) / $(met $st)" "SKIP / needs lerobot_install (SKIPPED)"
done
check "native: openpi eval SKIP"          "$(met openpi_pi05_libero_eval)" "needs openpi_install (SKIPPED)"
check "native: maniskill_ppo SKIP"        "$(met maniskill_ppo)" "needs maniskill_install (SKIPPED)"
check "native: isaac_rl SKIP"             "$(met isaac_rl_franka)" "needs isaac_validate (FAILED)"
check "native: Newton mismatch hint"     "$(hint isaac_validate)" \
  "Newton version mismatch with Isaac Lab; see the Newton/Warp pins row in the step log for the fix command (rc=1)"
check "native: smoke log tail in SUMMARY" "$(grep -c "^AttributeError: 'ModelBuilder' object has no attribute '_cloth_label'" "$S")" 1
check "native: pins row + fix in SUMMARY" "$(grep -c '^Newton/Warp pins .*WARN.*uv pip install --python' "$S")" 1
check "native: no WSL wording in hints"   "$(grep -E '^- Hint:' "$S" | grep -ciE 'wsl')" 0
check "native: no WSL wording on console" "$(grep -E 'hint:|WSL2' "$TMP/out4.txt" | grep -ciE 'wsl2|wslconfig')" 0

# Same failures on simulated WSL2: the Isaac hint now carries the WSL2 support note.
rm -rf "$RUNS_DIR"
RSP_FORCE_WSL=1 RSP_DRY_RUN_RC="base_deps=100 isaac_validate=1" bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "wsl: dpkg hint same wording"       "$(hint base_deps)" "apt is broken: run sudo dpkg --configure -a, then rerun (rc=100)"
check "wsl: Newton hint, no WSL blame"   "$(hint isaac_validate | cut -c1-24)" "Newton version mismatch "

# User-chosen skips do NOT block dependents: SKIP_INSTALL=1 still runs the train/eval steps.
rm -rf "$RUNS_DIR"
SKIP_INSTALL=1 SKIP_ISAAC=1 bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1; rc=$?
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "SKIP_INSTALL: exit 0"              "$rc" 0
check "SKIP_INSTALL: lerobot_install SKIP" "$(row lerobot_install)" SKIP
check "SKIP_INSTALL: act_aloha still runs" "$(row act_aloha)" PASS

# rc 77 (step skipped itself at runtime) blocks its dependents.
rm -rf "$RUNS_DIR"
SKIP_ISAAC=1 RSP_DRY_RUN_RC="lerobot_install=77" bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "rc77 install: act_aloha SKIP"      "$(met act_aloha)" "needs lerobot_install (SKIPPED)"

# The user's real LeRobot failures (renamed --eval_freq, LIBERO first-run prompt) replayed: hints name the cause.
rm -rf "$RUNS_DIR"
RSP_FORCE_WSL=0 SKIP_ISAAC=1 RSP_DRY_RUN_RC="act_aloha=2 smolvla_libero_eval=1" bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "renamed flag hint"   "$(hint act_aloha | cut -c1-55)" "the tool's version does not accept --eval_freq=2000 (fl"
check "LIBERO prompt hint"  "$(hint smolvla_libero_eval | cut -c1-36)" "LIBERO asked its first-run question "

# The rerun of only the failed LeRobot steps: they run although their install dependency is skipped.
rm -rf "$RUNS_DIR"
ONLY_STEPS="act_aloha diffusion_pusht smolvla_libero_eval" bash "$K/scripts/run_wsl_headless.sh" > "$TMP/only.txt" 2>&1; rc=$?
S="$(ls "$RUNS_DIR"/*/SUMMARY.md | sed -n 1p)"
check "ONLY_STEPS rerun: exit 0"            "$rc" 0
check "ONLY_STEPS rerun: act_aloha runs"    "$(row act_aloha)" PASS
check "ONLY_STEPS rerun: smolvla runs"      "$(row smolvla_libero_eval)" PASS
check "ONLY_STEPS rerun: install skipped"   "$(met lerobot_install)" "not in ONLY_STEPS"
check "ONLY_STEPS rerun: no sudo prompt"    "$(grep -c 'asking once now' "$TMP/only.txt")" 0

# Unit tests for hints / apt / dpkg / declared dependencies (run from here so the existing CI step covers them).
if bash "$T/test_hints_apt.sh" > "$TMP/hints.txt" 2>&1; then echo "ok   - tests/test_hints_apt.sh ($(grep -c '^ok' "$TMP/hints.txt") checks)"
else grep -v '^ok' "$TMP/hints.txt"; echo "FAIL - tests/test_hints_apt.sh"; nfail=$((nfail+1)); fi

if bash "$T/test_lerobot_libero.sh" > "$TMP/lr.txt" 2>&1; then echo "ok   - tests/test_lerobot_libero.sh ($(grep -c '^ok' "$TMP/lr.txt") checks)"
else grep -v '^ok' "$TMP/lr.txt"; echo "FAIL - tests/test_lerobot_libero.sh"; nfail=$((nfail+1)); fi

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
