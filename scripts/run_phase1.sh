#!/usr/bin/env bash
# run_phase1.sh - phase 1 end to end: audit -> base deps -> Isaac validation -> LeRobot -> first eval -> first training.
# Each step is idempotent; rerun after fixing whatever failed. QUICK=1 shortens training for a pipeline check.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
source "$D/lib/common.sh"
start_log "phase1"
RESULTS=()
run_step() {
  local name="$1"; shift
  log "== $name =="
  if "$@"; then RESULTS+=("$name|PASS"); ok "$name"
  else RESULTS+=("$name|FAIL"); warn "$name failed"; fi
}
run_step "00_check_system" bash "$D/00_check_system.sh"
run_step "01_base_deps" bash "$D/01_base_deps.sh"
run_step "02_isaaclab_validate (non-blocking)" bash -c "bash '$D/02_isaaclab_validate.sh' || true"
run_step "10_install_lerobot" bash "$D/10_install_lerobot.sh"
run_step "eval/lerobot_smolvla_libero (quick)" bash "$D/eval/lerobot_smolvla_libero.sh"
if [ "${QUICK:-0}" = 1 ]; then
  run_step "train/lerobot_act_aloha (short)" bash -c "STEPS=2000 EVAL_FREQ=1000 bash '$D/train/lerobot_act_aloha.sh'"
else
  run_step "train/lerobot_act_aloha" bash "$D/train/lerobot_act_aloha.sh"
fi
echo
log "== Phase 1 summary =="
fails=0
for r in "${RESULTS[@]}"; do IFS='|' read -r n s <<<"$r"; printf '%-34s %s\n' "$n" "$s"; [ "$s" = FAIL ] && fails=$((fails+1)); done
if [ "$fails" -eq 0 ]; then ok "Phase 1 PASS"
else warn "Phase 1 has $fails failing step(s). See $REPORT for logs."; fi
log "Next: 11_install_openpi.sh + eval/openpi_pi05_libero.sh, 12_install_openvla_oft.sh + eval/openvla_oft_libero.sh"
