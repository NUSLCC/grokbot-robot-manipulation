#!/usr/bin/env bash
# openpi pi0.5 on LIBERO, non-Docker path from examples/libero/README.md:
#   server: uv run scripts/serve_policy.py --env LIBERO   (default ckpt gs://openpi-assets/checkpoints/pi05_libero)
#   client: examples/libero/main.py in the Python 3.8 venv
# Official reference (openpi README): pi0.5 @30k: spatial 98.8 / object 98.2 / goal 98.0 / libero_10 92.4.
# SUITES=libero_spatial,libero_object,libero_goal,libero_10 evaluates several suites with one policy server and
# prints the combined "Total success rate" / "Total episodes" last (per-suite lines above it). SUITE= still works.
# Custom checkpoint: SERVER_ARGS="policy:checkpoint --policy.config=pi05_libero --policy.dir=/path/to/ckpt"
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection; ensure_uv
: "${SUITE:=libero_spatial}"; : "${SUITES:=$SUITE}"; : "${TRIALS:=5}"; : "${PORT:=8000}"; : "${SERVER_ARGS:=}"
SRC="$RSP_SRC/openpi"; [ -d "$SRC/.venv" ] || die "Run scripts/11_install_openpi.sh first"
OUT="$RSP_LOGS/eval_openpi_libero_$(stamp)"; mkdir -p "$OUT"
cd "$SRC"
UVRUN=(uv run); [ -f .rsp_no_sync ] && UVRUN=(uv run --no-sync)   # see 11_install_openpi.sh OPENPI_TORCH_CU128
select_mujoco_gl "$SRC/examples/libero/.venv/bin/python"          # EGL, or OSMesa fallback on WSL2
log "Starting policy server (first run downloads the checkpoint)"
# shellcheck disable=SC2086
"${UVRUN[@]}" scripts/serve_policy.py --env LIBERO --port "$PORT" $SERVER_ARGS > "$OUT/server.log" 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null || true' EXIT
wait_port "$PORT" 1800 || { tail -n 40 "$OUT/server.log"; die "server did not come up"; }
ok "server listening on :$PORT"
gpu_monitor_start "$OUT/gpu.csv"
IFS=, read -ra SUITE_LIST <<<"$SUITES"; tot_s=0; tot_n=0; per=()
for st in "${SUITE_LIST[@]}"; do
  log "LIBERO suite $st: $TRIALS trials per task"
  PYTHONPATH="$SRC/third_party/libero" examples/libero/.venv/bin/python examples/libero/main.py \
    --args.task-suite-name "$st" --args.num-trials-per-task "$TRIALS" --args.port "$PORT" \
    --args.video-out-path "$OUT/videos/$st" 2>&1 | tee "$OUT/client_$st.log"
  r="$(grep -aoE 'Total success rate: *[0-9.eE+-]+' "$OUT/client_$st.log" | tail -n1 | grep -oE '[0-9.eE+-]+$' || true)"
  n="$(grep -aoE 'Total episodes: *[0-9]+' "$OUT/client_$st.log" | tail -n1 | grep -oE '[0-9]+$' || true)"
  [ -n "$r" ] && [ -n "$n" ] || die "suite $st: no 'Total success rate' / 'Total episodes' in $OUT/client_$st.log"
  k="$(awk -v r="$r" -v n="$n" 'BEGIN{printf "%d", r*n+0.5}')"; tot_s=$((tot_s+k)); tot_n=$((tot_n+n))
  per+=("$st $k/$n")
done
if [ "${#SUITE_LIST[@]}" -gt 1 ]; then
  for x in "${per[@]}"; do echo "Suite result: $x"; done
  echo "Total success rate: $(awk -v s="$tot_s" -v n="$tot_n" 'BEGIN{printf "%.4f", s/n}')"
  echo "Total episodes: $tot_n"
fi
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Logs + videos in $OUT"
