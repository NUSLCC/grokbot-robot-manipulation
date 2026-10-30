#!/usr/bin/env bash
# Used only by RSP_DRY_RUN=1: replays tests/fixtures/<step>.log (real upstream log formats) and exits with RC.
#   fake_step.sh STEP RC
s="$1"; rc="${2:-0}"; f="$(cd "$(dirname "$0")" && pwd)/fixtures/$s.log"
echo "[dry-run] $s (rc=$rc)"
if [ -f "$f" ]; then sed "s|@RUNTMP@|${RSP_DRY_RUN_TMP:-/tmp}|g" "$f"; else echo "[ OK ] $s done (no fixture)"; fi
if [ "$rc" = 77 ]; then echo "[WARN] SKIP: simulated skip for $s"; fi
case "$rc" in
  1)   echo "RuntimeError: CUDA error: no kernel image is available for execution on the device" ;;
  137) echo "Killed" ;;
  0|77) ;;
  *)   echo "error: something generic went wrong" ;;
esac
sleep "${FAKE_STEP_SLEEP:-0}"
exit "$rc"
