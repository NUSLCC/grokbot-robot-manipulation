#!/usr/bin/env bash
# openpi pi0.5 LIBERO diagnostics: why does the policy return NaN / huge actions? Read-only: changes no env.
#   bash scripts/diag/openpi_policy_check.sh        (CUDA_VISIBLE_DEVICES=1 to test GPU 1; DIAG_CPU=0 skips the CPU run)
# Steps (each prints KEY=0/1 lines; a VERDICT is printed at the end; everything is in $RSP_LOGS/diag_openpi_*/diag.log):
#   1 versions: uv, openpi commit, `uv sync --dry-run` (would uv change the env = drift from uv.lock?)
#   2 client: first real LIBERO observation built like examples/libero/main.py -> OBS_FINITE, saved to obs.npz
#   3 server in-process on the GPU: params NaN scan, policy.infer on openpi's example + the real obs -> GPU_ACTIONS_FINITE
#   4 same on the CPU (JAX_PLATFORMS=cpu, slow)                                                  -> CPU_ACTIONS_FINITE
#   5 round trip: real server + openpi_client, the eval's exact path, then 5 env steps            -> ROUNDTRIP_ACTIONS_FINITE
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection; ensure_uv
set +e
D="$(cd "$(dirname "$0")" && pwd)"; SRC="$RSP_SRC/openpi"
[ -d "$SRC/.venv" ] || die "Run scripts/11_install_openpi.sh first"
OUT="$RSP_LOGS/diag_openpi_$(stamp)"; mkdir -p "$OUT"; exec > >(tee "$OUT/diag.log") 2>&1
CPY="$SRC/examples/libero/.venv/bin/python"; cd "$SRC" || exit 1
k() { grep -aoE "^$1=[0-9]+" "$OUT/diag.log" | tail -n1 | cut -d= -f2; }

log "1/5 versions"
echo "uv=$(uv --version 2>&1)  openpi=$(git log -1 --format='%h %cd' --date=short 2>/dev/null)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv,noheader 2>/dev/null
echo "--- openpi git reflog (a recent 'pull' = openpi_install updated openpi main, and with it uv.lock):"
git reflog -5 --date=iso 2>/dev/null
echo "--- when jax / nvidia / flax wheels in .venv last changed:"
find .venv/lib/python3*/site-packages -maxdepth 1 \( -name 'jax*.dist-info' -o -name 'nvidia_*.dist-info' -o -name 'flax*.dist-info' \) \
  -printf '%TY-%Tm-%Td %TH:%TM  %f\n' 2>/dev/null | sort | tail -n 25
echo "--- uv sync --dry-run (lines other than 'Resolved/Audited/Would make no changes' = env differs from uv.lock):"
GIT_LFS_SKIP_SMUDGE=1 timeout 300 uv sync --dry-run 2>&1 | tail -n 25

log "2/5 client observation (examples/libero/.venv)"
select_mujoco_gl "$CPY"; libero_config "$CPY" openpi "$SRC/third_party/libero"
PYTHONPATH="$SRC/third_party/libero" timeout 600 "$CPY" -u "$D/openpi_client_obs.py" "$OUT/obs.npz"

log "3/5 server in-process on ${CUDA_VISIBLE_DEVICES:+GPU $CUDA_VISIBLE_DEVICES}${CUDA_VISIBLE_DEVICES:-the default GPU}"
timeout 1800 uv run --no-sync python -u "$D/openpi_server_infer.py" "$OUT/obs.npz"

if [ "${DIAG_CPU:-1}" = 1 ]; then
  log "4/5 server in-process on the CPU (slow: several minutes)"
  JAX_PLATFORMS=cpu timeout 3600 uv run --no-sync python -u "$D/openpi_server_infer.py" "$OUT/obs.npz"
else log "4/5 CPU run skipped (DIAG_CPU=0)"; fi

log "5/5 round trip through a real policy server (the eval's path)"
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
PORT="${PORT:-8000}"; while port_busy "$PORT"; do PORT=$((PORT+1)); done
SETSID=(); have setsid && SETSID=(setsid)
"${SETSID[@]}" uv run --no-sync scripts/serve_policy.py --env LIBERO --port "$PORT" > "$OUT/server.log" 2>&1 &
SPID=$!; trap 'kill -- "-$SPID" 2>/dev/null || kill "$SPID" 2>/dev/null' EXIT
t0=$SECONDS; until port_busy "$PORT"; do kill -0 "$SPID" 2>/dev/null || { tail -n 30 "$OUT/server.log"; break; }; [ $((SECONDS-t0)) -ge 1800 ] && break; sleep 2; done
if port_busy "$PORT"; then
  PYTHONPATH="$SRC/third_party/libero" timeout 600 "$CPY" -u "$D/openpi_client_obs.py" "$OUT/obs2.npz" --port "$PORT"
else warn "server did not come up; see $OUT/server.log"; fi

sleep 1; echo; echo "===== SUMMARY ====="
for x in OBS_FINITE PARAMS_NONFINITE GPU_ACTIONS_FINITE CPU_ACTIONS_FINITE ROUNDTRIP_ACTIONS_FINITE ENV_CTRL_FINITE; do echo "$x=$(k "$x")"; done
o="$(k OBS_FINITE)"; p="$(k PARAMS_NONFINITE)"; g="$(k GPU_ACTIONS_FINITE)"; c="$(k CPU_ACTIONS_FINITE)"; r="$(k ROUNDTRIP_ACTIONS_FINITE)"
if [ "$o" = 0 ]; then V="client observation is NaN/wrong shape: LIBERO/robosuite in examples/libero/.venv"
elif [ -n "$p" ] && [ "$p" != 0 ]; then V="checkpoint params contain NaN/Inf: corrupted cache; move \$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero aside and rerun (it re-downloads)"
elif [ "$g" = 0 ] && [ "$c" = 1 ]; then V="GPU-only NaN: JAX/XLA/CUDA numerics on this GPU (wheel drift or XLA autotuning); compare 'uv sync --dry-run' above, try the other GPU"
elif [ "$g" = 0 ] && [ "$c" = 0 ]; then V="NaN on GPU and CPU: model/inputs (norm stats or checkpoint), not the GPU"
elif [ "$g" = 0 ]; then V="GPU NaN (CPU not run: DIAG_CPU=1 tells GPU vs model)"
elif [ "$g" = 1 ] && [ "$r" = 0 ]; then V="model is fine in-process but NaN through the server: client/server transport (openpi_client versions) or server config"
elif [ "$g" = 1 ] && [ "$r" = 1 ]; then V="policy outputs are finite end to end on this obs: look at later steps (rerun the eval) or the env side (ENV_CTRL_FINITE)"
else V="incomplete: see the step output above"; fi
echo "VERDICT: $V"; echo "Full log: $OUT/diag.log"
