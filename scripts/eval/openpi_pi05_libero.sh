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
# Own LIBERO config, pinned to openpi's third_party/libero. Before, the client read the shared ~/.libero/config.yaml,
# which another LIBERO copy (e.g. LeRobot's hf-libero) can point at its own bddl/init-state/asset files.
h="$HOME/.libero/config.yaml"
[ -f "$h" ] && log "shared $h: $(grep -m1 '^benchmark_root' "$h" || echo 'no benchmark_root')"
libero_config "$SRC/examples/libero/.venv/bin/python" openpi "$SRC/third_party/libero"
if [ -f "$h" ] && [ -n "${LIBERO_CONFIG_PATH:-}" ] && [ "$(grep -m1 '^benchmark_root' "$h")" != "$(grep -m1 '^benchmark_root' "$LIBERO_CONFIG_PATH/config.yaml")" ]; then
  warn "$h points at another LIBERO copy; this eval now uses $LIBERO_CONFIG_PATH/config.yaml instead"
fi
# The client must talk to OUR server: a port already in use (e.g. a server orphaned by an earlier run) made
# wait_port succeed at once while our server failed to bind, and the client then evaluated whatever was listening.
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
p0="$PORT"; while port_busy "$PORT"; do PORT=$((PORT+1)); [ "$PORT" -gt $((p0+50)) ] && die "ports $p0-$PORT are all in use"; done
[ "$PORT" != "$p0" ] && warn "port $p0 is already in use (another server? see: ss -ltnp | grep :$p0); using port $PORT"
log "Starting policy server on :$PORT (first run downloads the checkpoint)"
# Own session/process group, so the EXIT trap stops uv AND its python child (no orphaned server holding the port).
SETSID=(); have setsid && SETSID=(setsid)
# shellcheck disable=SC2086
"${SETSID[@]}" "${UVRUN[@]}" scripts/serve_policy.py --env LIBERO --port "$PORT" $SERVER_ARGS > "$OUT/server.log" 2>&1 &
SPID=$!
stop_server() { kill -- "-$SPID" 2>/dev/null || kill "$SPID" 2>/dev/null || true; }
trap stop_server EXIT
server_alive() { kill -0 "$SPID" 2>/dev/null || { tail -n 40 "$OUT/server.log"; die "policy server exited ($1); see $OUT/server.log"; }; }
t0=$SECONDS   # wait for OUR server: stop at once if it exits (wait_port alone would wait 30 min for a dead server)
until port_busy "$PORT"; do server_alive "while starting"; [ $((SECONDS-t0)) -ge 1800 ] && { tail -n 40 "$OUT/server.log"; die "server did not come up"; }; sleep 2; done
server_alive "after start"
ok "server listening on :$PORT (pid $SPID, ready after $((SECONDS-t0)) s)"
gpu_monitor_start "$OUT/gpu.csv"
IFS=, read -ra SUITE_LIST <<<"$SUITES"; tot_s=0; tot_n=0; per=()
for st in "${SUITE_LIST[@]}"; do
  server_alive "before suite $st"
  log "LIBERO suite $st: $TRIALS trials per task"
  PYTHONPATH="$SRC/third_party/libero" examples/libero/.venv/bin/python examples/libero/main.py \
    --args.task-suite-name "$st" --args.num-trials-per-task "$TRIALS" --args.port "$PORT" \
    --args.video-out-path "$OUT/videos/$st" 2>&1 | tee "$OUT/client_$st.log"
  r="$(grep -aoE 'Total success rate: *[0-9.eE+-]+' "$OUT/client_$st.log" | tail -n1 | grep -oE '[0-9.eE+-]+$' || true)"
  n="$(grep -aoE 'Total episodes: *[0-9]+' "$OUT/client_$st.log" | tail -n1 | grep -oE '[0-9]+$' || true)"
  [ -n "$r" ] && [ -n "$n" ] || die "suite $st: no 'Total success rate' / 'Total episodes' in $OUT/client_$st.log"
  k="$(awk -v r="$r" -v n="$n" 'BEGIN{printf "%d", r*n+0.5}')"; tot_s=$((tot_s+k)); tot_n=$((tot_n+n))
  per+=("$st $k/$n")
  grep -aq "Task suite: $st" "$OUT/client_$st.log" || warn "suite $st: client log has no 'Task suite: $st' line"
done
if [ "${#SUITE_LIST[@]}" -gt 1 ]; then
  for x in "${per[@]}"; do echo "Suite result: $x"; done
  echo "Total success rate: $(awk -v s="$tot_s" -v n="$tot_n" 'BEGIN{printf "%.4f", s/n}')"
  echo "Total episodes: $tot_n"
fi
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"
if [ "$tot_s" = 0 ] && [ "${ALLOW_ZERO_SUCCESS:-0}" != 1 ]; then
  # 0 successes from a pretrained pi0.5 (reference ~98%) means a broken setup, not a score. Collect clues.
  v="$(find "$OUT/videos" -name '*.mp4' 2>/dev/null | sort | sed -n 1p)"
  if [ -n "$v" ]; then
    b="$(examples/libero/.venv/bin/python -c 'import sys,imageio,numpy as np; r=imageio.get_reader(sys.argv[1]); f=[np.asarray(x,dtype=float) for i,x in zip(range(30),r)]; print("%.1f %.1f"%(np.mean(f),np.std(f)))' "$v" 2>/dev/null || true)"
    echo "Video check: $v mean/std pixel = ${b:-unreadable} (near 0 / tiny std = black or blank frames: rendering problem)"
  fi
  echo "Server log head:"; sed -n 1,25p "$OUT/server.log"
  die "pi0.5 had 0 successes in $tot_n episodes: setup problem, not a score (ALLOW_ZERO_SUCCESS=1 accepts it). Logs + videos in $OUT"
fi
ok "Logs + videos in $OUT"
