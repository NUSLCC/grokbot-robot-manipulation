#!/usr/bin/env bash
# openpi pi0.5 on LIBERO, non-Docker path from examples/libero/README.md:
#   server: uv run scripts/serve_policy.py --env LIBERO   (default ckpt gs://openpi-assets/checkpoints/pi05_libero)
#   client: examples/libero/main.py in the Python 3.8 venv
# Official reference (openpi README): pi0.5 @30k: spatial 98.8 / object 98.2 / goal 98.0 / libero_10 92.4.
# Custom checkpoint: SERVER_ARGS="policy:checkpoint --policy.config=pi05_libero --policy.dir=/path/to/ckpt"
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection; ensure_uv
: "${SUITE:=libero_spatial}"; : "${TRIALS:=5}"; : "${PORT:=8000}"; : "${SERVER_ARGS:=}"
SRC="$RSP_SRC/openpi"; [ -d "$SRC/.venv" ] || die "Run scripts/11_install_openpi.sh first"
OUT="$RSP_LOGS/eval_openpi_libero_$(stamp)"; mkdir -p "$OUT"
cd "$SRC"
log "Starting policy server (first run downloads the checkpoint)"
# shellcheck disable=SC2086
uv run scripts/serve_policy.py --env LIBERO --port "$PORT" $SERVER_ARGS > "$OUT/server.log" 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null || true' EXIT
wait_port "$PORT" 1800 || { tail -n 40 "$OUT/server.log"; die "server did not come up"; }
ok "server listening on :$PORT"
gpu_monitor_start "$OUT/gpu.csv"
PYTHONPATH="$SRC/third_party/libero" examples/libero/.venv/bin/python examples/libero/main.py \
  --args.task-suite-name "$SUITE" --args.num-trials-per-task "$TRIALS" --args.port "$PORT" \
  --args.video-out-path "$OUT/videos" 2>&1 | tee "$OUT/client.log"
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Logs + videos in $OUT"
