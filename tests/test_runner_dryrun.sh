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
echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
