#!/usr/bin/env bash
# run_wsl_headless.sh - curated, fully headless validation run. Made for WSL2 Ubuntu 24.04, also works on native
# Ubuntu 22.04/24.04. Reuses the kit's scripts; one failing step never stops the others.
#
#   QUICK=1 bash scripts/run_wsl_headless.sh      # short runs (~1-3 h incl. downloads on a first run)
#   bash scripts/run_wsl_headless.sh              # full-length default training/eval settings
#
# Output: runs/<timestamp>/<NN>_<step>.log, <step>.gpu.csv (nvidia-smi samples) and runs/<timestamp>/SUMMARY.md.
# The final table shows: step, PASS/FAIL/SKIP, wall time, peak GPU MB, key metric, and log path + hint on failure.
#
# Options (env vars):
#   QUICK=1              short settings for every step
#   SKIP_ISAAC=1         skip the Isaac Sim / Isaac Lab steps
#   ISAAC_PHYSX=0        skip the extra Isaac Sim PhysX smoke (Kit; the step most likely to fail on WSL2)
#   RUN_MIMIC=1          also run the Isaac Lab Mimic pipeline (needs the 'mimic' extra in your Isaac env)
#   SKIP_STEPS="a b"     skip steps by name    ONLY_STEPS="a b"   run only these steps
#   SKIP_INSTALL=1       do not run base_deps / *_install steps (assume already installed)
#   STEP_TIMEOUT=14400   per-step timeout in seconds (0 = none, default)
#   RUNS_DIR=path        where runs/<timestamp> goes (default: <repo>/runs)
#   RSP_DRY_RUN=1        do not run real steps: replay tests/fixtures logs (CI / control-flow check).
#                        RSP_DRY_RUN_RC="step=rc ..." forces exit codes, e.g. "act_aloha=1 maniskill_ppo=77".
# Exit code: number of FAILed steps.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib/common.sh
source "$D/lib/common.sh"
# shellcheck source=lib/results.sh
source "$D/lib/results.sh"
set +e   # common.sh enables errexit; the runner handles every failure itself

: "${QUICK:=0}"; : "${SKIP_ISAAC:=0}"; : "${ISAAC_PHYSX:=1}"; : "${RUN_MIMIC:=0}"; : "${SKIP_INSTALL:=0}"
: "${SKIP_STEPS:=}"; : "${ONLY_STEPS:=}"; : "${STEP_TIMEOUT:=0}"; : "${RSP_DRY_RUN:=0}"; : "${RSP_DRY_RUN_RC:=}"
RUN_ID="$(stamp)"; OUT="${RUNS_DIR:-$RSP_KIT_DIR/runs}/$RUN_ID"; mkdir -p "$OUT"
MAIN_LOG="$OUT/runner.log"
exec > >(tee "$MAIN_LOG") 2>&1

# ---- no GUI anywhere ----
unset DISPLAY WAYLAND_DISPLAY
export SDL_VIDEODRIVER=dummy QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg LIVESTREAM=0
export WANDB_MODE="${WANDB_MODE:-offline}" HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_MININTERVAL=30

log "run_wsl_headless.sh  run=$RUN_ID  QUICK=$QUICK  dry_run=$RSP_DRY_RUN  -> $OUT"
log "platform: $(is_wsl && echo WSL2 || echo 'native Linux'), Ubuntu $(os_version_id), profile $PROFILE: $RSP_PROFILE_DESC"
is_wsl && log "WSL2 note: Isaac Sim/Isaac Lab on WSL2 are not officially supported by NVIDIA; those steps run anyway (SKIP_ISAAC=1 to skip)."

if [ "$QUICK" = 1 ]; then
  Q_ISAAC=(SMOKE_TASKS=Isaac-Reach-Franka SMOKE_NUM_ENVS=32 SMOKE_ITERS=2)
  Q_RL=(TASK=Isaac-Reach-Franka NUM_ENVS=512 MAX_ITERS=30 PLAY=0)
  Q_ACT=(STEPS=2000 EVAL_FREQ=2000 N_EVAL=5)
  Q_DP=(STEPS=2000 EVAL_FREQ=2000 N_EVAL=10 BATCH=32)
  Q_SMOL=(SUITES=libero_object N_EPISODES=1)
  Q_PI=(SUITE=libero_spatial TRIALS=1)
  Q_MS=(TOTAL=500_000 NUM_ENVS=1024)
  Q_MIMIC=(N_GEN=20 GEN_ENVS=10 EPOCHS=20 N_ROLLOUTS=5)
else
  Q_ISAAC=(SMOKE_TASKS=Isaac-Reach-Franka); Q_RL=(PLAY=0); Q_ACT=(); Q_DP=(); Q_SMOL=(); Q_PI=(); Q_MS=(); Q_MIMIC=()
fi

# ---- step table: name | kind (metric parser) | depends-on | command ----
STEPS=(); declare -A KIND DEP CMD STATUS SECS PEAK METRIC LOGF HINT
add() { STEPS+=("$1"); KIND[$1]="$2"; DEP[$1]="$3"; CMD[$1]="$4"; }
q() { local a out=""; for a in "$@"; do out+=" $(printf '%q' "$a")"; done; echo "$out"; }
add system_check   syscheck       ""               "bash $(q "$D/00_check_system.sh")"
add base_deps      install        ""               "bash $(q "$D/01_base_deps.sh")"
add isaac_validate isaac_validate ""               "env SMOKE_PHYSICS=default $(q "${Q_ISAAC[@]}") bash $(q "$D/02_isaaclab_validate.sh")"
add isaac_physx_smoke isaac_validate isaac_validate "env SMOKE_PHYSICS=isaacsim_physx $(q "${Q_ISAAC[@]}") bash $(q "$D/02_isaaclab_validate.sh")"
add isaac_rl_franka isaac_rl      isaac_validate   "env $(q "${Q_RL[@]}") bash $(q "$D/train/isaaclab_rl_franka.sh")"
add isaac_mimic    install        isaac_validate   "env $(q "${Q_MIMIC[@]}") bash $(q "$D/train/isaaclab_mimic_stack.sh")"
add lerobot_install install       base_deps        "bash $(q "$D/10_install_lerobot.sh")"
add act_aloha      lerobot_train  lerobot_install  "env $(q "${Q_ACT[@]}") bash $(q "$D/train/lerobot_act_aloha.sh")"
add diffusion_pusht lerobot_train lerobot_install  "env $(q "${Q_DP[@]}") bash $(q "$D/train/lerobot_diffusion_pusht.sh")"
add smolvla_libero_eval lerobot_eval lerobot_install "env $(q "${Q_SMOL[@]}") bash $(q "$D/eval/lerobot_smolvla_libero.sh")"
add openpi_install install        base_deps        "bash $(q "$D/11_install_openpi.sh")"
add openpi_pi05_libero_eval openpi_eval openpi_install "env $(q "${Q_PI[@]}") bash $(q "$D/eval/openpi_pi05_libero.sh")"
add maniskill_install install     base_deps        "bash $(q "$D/14_install_maniskill.sh")"
add maniskill_ppo  maniskill_ppo  maniskill_install "env $(q "${Q_MS[@]}") bash $(q "$D/train/maniskill_ppo.sh")"

in_list() { local x; for x in $2; do [ "$x" = "$1" ] && return 0; done; return 1; }
skip_reason() {
  local s="$1"
  if [ -n "$ONLY_STEPS" ] && ! in_list "$s" "$ONLY_STEPS"; then echo "not in ONLY_STEPS"; return; fi
  in_list "$s" "$SKIP_STEPS" && { echo "SKIP_STEPS"; return; }
  case "$s" in
    isaac_*) [ "$SKIP_ISAAC" = 1 ] && { echo "SKIP_ISAAC=1"; return; } ;;
  esac
  [ "$s" = isaac_physx_smoke ] && [ "$ISAAC_PHYSX" != 1 ] && { echo "ISAAC_PHYSX=0"; return; }
  [ "$s" = isaac_mimic ] && [ "$RUN_MIMIC" != 1 ] && { echo "optional (RUN_MIMIC=1)"; return; }
  if [ "$SKIP_INSTALL" = 1 ]; then case "$s" in base_deps|*_install) echo "SKIP_INSTALL=1"; return ;; esac; fi
  local d="${DEP[$s]}"
  if [ -n "$d" ] && [ "${STATUS[$d]:-}" = FAIL ]; then echo "needs $d (FAILED)"; return; fi
  echo ""
}

# sudo once up front so apt prompts do not hide inside a log file
if [ "$RSP_DRY_RUN" != 1 ] && [ "$SKIP_INSTALL" != 1 ] && [ "${SKIP_APT:-0}" != 1 ] && [ -z "$(skip_reason base_deps)" ] && [ "$(id -u)" != 0 ]; then
  log "base_deps may need sudo for apt; asking once now:"; sudo -v || warn "sudo failed; base_deps will probably FAIL (or set SKIP_APT=1)"
fi

# ---- GPU sampler (per step): peak of memory.used on the busiest GPU, in MiB ----
SAMPLER_PID=""
sampler_start() {
  SAMPLER_PID=""
  have nvidia-smi || return 0
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 1 > "$1" 2>/dev/null &
  SAMPLER_PID=$!
}
sampler_stop() {
  if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; fi
  if [ -s "$1" ]; then awk -F, '{gsub(/ /,"",$2); if($2+0>m)m=$2+0}END{print m+0}' "$1"; else echo "n/a"; fi
}
trap '[ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null' EXIT

dry_rc() { local kv; for kv in $RSP_DRY_RUN_RC; do [ "${kv%%=*}" = "$1" ] && { echo "${kv#*=}"; return; }; done; echo 0; }

i=0
for s in "${STEPS[@]}"; do
  i=$((i+1)); n="$(printf '%02d' "$i")"; lf="$OUT/${n}_${s}.log"; LOGF[$s]="$lf"
  why="$(skip_reason "$s")"
  if [ -n "$why" ]; then
    STATUS[$s]=SKIP; SECS[$s]="-"; PEAK[$s]="-"; METRIC[$s]="$why"; HINT[$s]=""
    log "[$n] $s: SKIP ($why)"; continue
  fi
  log "[$n] $s: running  (log: $lf)"
  sampler_start "$OUT/${s}.gpu.csv"
  t0=$(date +%s)
  if [ "$RSP_DRY_RUN" = 1 ]; then
    cmd="bash $(q "$RSP_KIT_DIR/tests/fake_step.sh") $(q "$s") $(q "$(dry_rc "$s")")"
  else
    cmd="${CMD[$s]}"
  fi
  if [ "$STEP_TIMEOUT" != 0 ]; then cmd="timeout --kill-after=60 $STEP_TIMEOUT $cmd"; fi
  # </dev/null: no step can wait on terminal input; output only goes to the step log
  bash -c "$cmd" </dev/null >"$lf" 2>&1
  rc=$?
  SECS[$s]=$(( $(date +%s) - t0 ))
  PEAK[$s]="$(sampler_stop "$OUT/${s}.gpu.csv")"
  case "$rc" in
    0)  STATUS[$s]=PASS; HINT[$s]="" ;;
    "$RSP_EXIT_SKIP") STATUS[$s]=SKIP; HINT[$s]="$(grep -aoE 'SKIP: .*' "$lf" | tail -n1 | sed 's/^SKIP: //')" ;;
    *)  STATUS[$s]=FAIL; HINT[$s]="$(failure_hint "$s" "$lf" "$rc") (rc=$rc)" ;;
  esac
  METRIC[$s]="$(extract_metric "${KIND[$s]}" "$lf")"
  [ "${STATUS[$s]}" = FAIL ] && [ "${KIND[$s]}" = install ] && METRIC[$s]="n/a"
  [ "${STATUS[$s]}" = SKIP ] && [ -n "${HINT[$s]}" ] && METRIC[$s]="${HINT[$s]}" && HINT[$s]=""
  log "[$n] $s: ${STATUS[$s]} in ${SECS[$s]}s  peak GPU ${PEAK[$s]} MiB  metric: ${METRIC[$s]}"
  if [ "${STATUS[$s]}" = FAIL ]; then warn "      hint: ${HINT[$s]}"; tail -n 15 "$lf" | sed 's/^/      | /'; fi
done

# ---- summary ----
fmt_t() {
  local t="$1"
  if [ "$t" = "-" ]; then echo "-"
  elif [ "$t" -lt 60 ]; then echo "${t}s"
  elif [ "$t" -lt 3600 ]; then printf '%dm%02ds\n' $((t/60)) $((t%60))
  else printf '%dh%02dm\n' $((t/3600)) $((t%3600/60)); fi
}
npass=0; nfail=0; nskip=0
for s in "${STEPS[@]}"; do case "${STATUS[$s]}" in PASS) npass=$((npass+1));; FAIL) nfail=$((nfail+1));; SKIP) nskip=$((nskip+1));; esac; done
SUM="$OUT/SUMMARY.md"
{
  echo "# Headless run $RUN_ID"
  echo
  echo "- Platform: $(is_wsl && echo 'WSL2' || echo 'native Linux'), Ubuntu $(os_version_id), kernel $(uname -r)"
  echo "- Profile: \`$PROFILE\` - $RSP_PROFILE_DESC"
  echo "- GPU: ${RSP_GPU_NAME:-none detected} x${RSP_GPU_COUNT:-0} (compute capability ${RSP_GPU_CC:-?}), torch wheels: $TORCH_CUDA"
  echo "- Mode: QUICK=$QUICK$( [ "$RSP_DRY_RUN" = 1 ] && echo ', DRY RUN (fixture logs, nothing real was run)')"
  echo "- Result: **$npass PASS, $nfail FAIL, $nskip SKIP**"
  echo
  echo "| Step | Result | Wall time | Peak GPU MB | Key metric |"
  echo "|---|---|---|---|---|"
  for s in "${STEPS[@]}"; do
    echo "| $s | ${STATUS[$s]} | $(fmt_t "${SECS[$s]}") | ${PEAK[$s]} | ${METRIC[$s]//|//} |"
  done
  if [ "$nfail" -gt 0 ]; then
    echo; echo "## Failures"; echo
    for s in "${STEPS[@]}"; do
      [ "${STATUS[$s]}" = FAIL ] || continue
      echo "### $s"; echo; echo "- Log: \`${LOGF[$s]}\`"; echo "- Hint: ${HINT[$s]}"; echo; echo '```'; tail -n 20 "${LOGF[$s]}"; echo '```'; echo
    done
  fi
  echo
  echo "Peak GPU MB = highest memory.used on any GPU while the step ran (whole GPU, sampled every second by nvidia-smi)."
  echo "Rerun only what failed: \`ONLY_STEPS=\"<step> ...\" bash scripts/run_wsl_headless.sh\` (add SKIP_INSTALL=1 if the installs passed)."
} > "$SUM"

echo
printf '%-24s %-5s %9s %8s  %s\n' STEP RESULT TIME "GPU_MB" "KEY METRIC"
printf '%-24s %-5s %9s %8s  %s\n' "------------------------" "-----" "---------" "--------" "----------------------------------------"
for s in "${STEPS[@]}"; do
  printf '%-24s %-5s %9s %8s  %s\n' "$s" "${STATUS[$s]}" "$(fmt_t "${SECS[$s]}")" "${PEAK[$s]}" "${METRIC[$s]}"
  [ "${STATUS[$s]}" = FAIL ] && printf '%-24s %s\n%-24s %s\n' "" "log:  ${LOGF[$s]}" "" "hint: ${HINT[$s]}"
done
echo
log "$npass PASS, $nfail FAIL, $nskip SKIP.  Summary: $SUM"
is_wsl && log "From Windows: \\\\wsl\$\\${WSL_DISTRO_NAME:-Ubuntu-24.04}${SUM//\//\\}"
exit "$nfail"
