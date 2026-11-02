#!/usr/bin/env bash
# openpi pi0.5 LIBERO diagnostics: why does the policy return NaN / huge actions? Read-only: changes no env, no files
# outside $RSP_LOGS/diag_openpi_<time>/ (everything is also in diag.log there).
#   bash scripts/diag/openpi_policy_check.sh
#     DIAG_GPUS="0 1"   GPUs to test one at a time (default: every GPU nvidia-smi lists)
#     DIAG_CPU=1        also run the policy on the CPU (slow; tells model/input NaN from GPU NaN)
#     DIAG_ROUNDTRIP=0  skip the real server + openpi_client round trip
# Steps:
#   1 versions + GPU health (temperature, clocks, throttle reasons, ECC where supported, Xid errors in the kernel log)
#   2 client: first real LIBERO observation built like examples/libero/main.py           -> OBS_FINITE
#   3 checkpoint params read from disk on the CPU, twice                                 -> PARAMS_NONFINITE, PARAMS_READS_MATCH
#   4 per GPU: VRAM pattern test, matmul vs CPU, params restored onto the GPU vs the CPU read, policy inference
#                                     -> GPU<n>_MEM_ERRORS, GPU<n>_MATMUL_OK, GPU<n>_RESTORE_MATCH, GPU<n>_ACTIONS_FINITE
#   5 optional CPU inference                                                             -> CPU_ACTIONS_FINITE
#   6 round trip through a real policy server on a GPU whose in-process actions were finite -> ROUNDTRIP_ACTIONS_FINITE
set -uo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; ensure_uv
set +e
D="$(cd "$(dirname "$0")" && pwd)"; SRC="$RSP_SRC/openpi"
[ -d "$SRC/.venv" ] || die "Run scripts/11_install_openpi.sh first"
OUT="$RSP_LOGS/diag_openpi_$(stamp)"; mkdir -p "$OUT"; exec > >(tee "$OUT/diag.log") 2>&1
CPY="$SRC/examples/libero/.venv/bin/python"; cd "$SRC" || exit 1
k() { grep -aoE "^$1=[0-9]+" "$OUT/diag.log" | tail -n1 | cut -d= -f2; }
SPY=(uv run --no-sync python -u "$D/openpi_server_infer.py")
GPUS_ALL="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')"
DIAG_GPUS="${DIAG_GPUS:-${GPUS_ALL:-0}}"; unset CUDA_VISIBLE_DEVICES

log "1/6 versions + GPU health"
echo "uv=$(uv --version 2>&1)  openpi=$(git log -1 --format='%h %cd' --date=short 2>/dev/null)  GPUs to test: $DIAG_GPUS"
nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active,ecc.errors.uncorrected.volatile.total,memory.used --format=csv 2>/dev/null
x="$( (journalctl -k -b --no-pager 2>/dev/null; dmesg 2>/dev/null) | grep -i 'NVRM: Xid' | tail -n 5)"
echo "Xid errors (kernel log): ${x:-none visible (root-only logs: sudo dmesg | grep -i xid)}"
echo "--- when jax / nvidia / flax wheels in .venv last changed:"
find .venv/lib/python3*/site-packages -maxdepth 1 \( -name 'jax*.dist-info' -o -name 'nvidia_*.dist-info' -o -name 'flax*.dist-info' \) \
  -printf '%TY-%Tm-%Td %TH:%TM  %f\n' 2>/dev/null | sort | tail -n 8
echo "--- uv sync --dry-run (anything but 'Would make no changes'/'Resolved'/'Audited' = env differs from uv.lock):"
GIT_LFS_SKIP_SMUDGE=1 timeout 300 uv sync --dry-run 2>&1 | tail -n 10

log "2/6 client observation (examples/libero/.venv, CPU)"
select_mujoco_gl "$CPY"; libero_config "$CPY" openpi "$SRC/third_party/libero"
PYTHONPATH="$SRC/third_party/libero" CUDA_VISIBLE_DEVICES="${DIAG_GPUS%% *}" timeout 600 "$CPY" -u "$D/openpi_client_obs.py" "$OUT/obs.npz"

log "3/6 checkpoint params on disk, read twice on the CPU (no GPU involved)"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" timeout 1800 "${SPY[@]}" scan "$OUT/params_crc.json"

for g in $DIAG_GPUS; do
  log "4/6 GPU $g: VRAM test, matmul, params restored onto the GPU, policy inference"
  CUDA_VISIBLE_DEVICES="$g" XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 2400 "${SPY[@]}" gpu "$OUT/params_crc.json" "$OUT/obs.npz"
done

if [ "${DIAG_CPU:-0}" = 1 ]; then
  log "5/6 policy on the CPU (slow)"
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" timeout 3600 "${SPY[@]}" cpu "$OUT/obs.npz"
else log "5/6 CPU inference skipped (DIAG_CPU=1 runs it)"; fi

RT=""; for g in $DIAG_GPUS; do [ "$(k "GPU${g}_ACTIONS_FINITE")" = 1 ] && { RT="$g"; break; }; done
if [ "${DIAG_ROUNDTRIP:-1}" = 1 ] && [ -n "$RT" ]; then
  log "6/6 round trip through a real policy server on GPU $RT (the eval's path)"
  port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
  PORT="${PORT:-8000}"; while port_busy "$PORT"; do PORT=$((PORT+1)); done
  SETSID=(); have setsid && SETSID=(setsid)
  CUDA_VISIBLE_DEVICES="$RT" "${SETSID[@]}" uv run --no-sync scripts/serve_policy.py --env LIBERO --port "$PORT" > "$OUT/server.log" 2>&1 &
  SPID=$!; trap 'kill -- "-$SPID" 2>/dev/null || kill "$SPID" 2>/dev/null' EXIT
  t0=$SECONDS; until port_busy "$PORT"; do kill -0 "$SPID" 2>/dev/null || { tail -n 30 "$OUT/server.log"; break; }; [ $((SECONDS-t0)) -ge 1800 ] && break; sleep 2; done
  if port_busy "$PORT"; then
    PYTHONPATH="$SRC/third_party/libero" CUDA_VISIBLE_DEVICES="$RT" timeout 600 "$CPY" -u "$D/openpi_client_obs.py" "$OUT/obs2.npz" --port "$PORT"
  else warn "server did not come up; see $OUT/server.log"; fi
else log "6/6 round trip skipped (${RT:+DIAG_ROUNDTRIP=0}${RT:-no GPU gave finite actions in-process})"; fi

sleep 1; echo; echo "===== SUMMARY ====="
keys="OBS_FINITE PARAMS_NONFINITE PARAMS_NONFINITE_LEAVES PARAMS_READS_MATCH"
for g in $DIAG_GPUS; do keys="$keys GPU${g}_MEM_ERRORS GPU${g}_MATMUL_OK GPU${g}_RESTORE_MATCH GPU${g}_ACTIONS_FINITE"; done
for x in $keys CPU_ACTIONS_FINITE ROUNDTRIP_ACTIONS_FINITE ENV_CTRL_FINITE; do echo "$x=$(k "$x")"; done
bad=""; good=""; badhw=""
for g in $DIAG_GPUS; do
  hw=""; m="$(k "GPU${g}_MEM_ERRORS")"; mm="$(k "GPU${g}_MATMUL_OK")"; r="$(k "GPU${g}_RESTORE_MATCH")"
  if [ -z "$m" ] || [ -z "$mm" ]; then hw="$hw check-crashed"; fi
  [ -n "$m" ] && [ "$m" != 0 ] && hw="$hw VRAM-errors"; [ -n "$mm" ] && [ "$mm" != 1 ] && hw="$hw matmul"
  [ -n "$r" ] && [ "$r" != 1 ] && hw="$hw restore-mismatch"
  if [ "$(k "GPU${g}_ACTIONS_FINITE")" = 1 ] && [ -z "$hw" ]; then good="$good $g"; else bad="$bad $g"; badhw="$badhw GPU$g:[${hw:- ok} ; actions $( [ "$(k "GPU${g}_ACTIONS_FINITE")" = 1 ] && echo finite || echo NaN)]"; fi
done
bad="${bad# }"; good="${good# }"; badhw="${badhw# }"
P="$(k PARAMS_NONFINITE)"
NOTE="(on-disk params: ${P:-?} non-finite values in $(k PARAMS_NONFINITE_LEAVES) leaves, read on the CPU; a released checkpoint can legitimately hold a few, so this alone is not a corruption verdict)"
if [ "$(k OBS_FINITE)" = 0 ]; then V="client observation is NaN/wrong shape: LIBERO/robosuite in examples/libero/.venv"
elif [ -n "$bad" ] && [ -n "$good" ]; then V="GPU-specific: problem on GPU ${bad} ($badhw), GPU ${good} is fine -> run pi0.5 on GPU ${good%% *} (CUDA_VISIBLE_DEVICES=${good%% *}); check the bad card (temperatures, Xid, reseat/power cable, a longer VRAM test)"
elif [ -n "$bad" ] && [ -z "$good" ] && [ -n "$(tr -d ' ' <<<"$badhw" | grep -E 'VRAM|matmul|restore')" ]; then V="hardware checks failed on every GPU tested ($badhw): suspect the driver/host rather than one card"
elif [ -n "$bad" ] && [ -z "$good" ]; then
  case "$(k CPU_ACTIONS_FINITE)" in
    1) V="NaN on every GPU but finite on the CPU, hardware checks clean: JAX/XLA GPU numerics (driver vs jax CUDA wheels)";;
    0) V="NaN on GPUs and on the CPU: the model inputs/weights (norm stats, checkpoint) or openpi code, not the GPU";;
    *) V="NaN on every GPU with clean hardware checks: rerun with DIAG_CPU=1 to tell GPU numerics from model/input";;
  esac
elif [ "$(k ROUNDTRIP_ACTIONS_FINITE)" = 0 ]; then V="policy is finite in-process but NaN through the server: client/server transport (openpi_client) or server config"
elif [ -n "$good" ]; then V="policy outputs are finite on GPU ${good} in-process and end to end: rerun the eval on that GPU"
else V="incomplete: see the step output above"; fi
echo "VERDICT: $V"
[ "$(k PARAMS_READS_MATCH)" = 0 ] && echo "WARNING: two CPU reads of the same checkpoint files differ: host RAM or disk read problem too (run memtest86+, check the disk with smartctl)"
[ -n "$bad" ] && echo "GPU health for GPU ${bad}: nvidia-smi -q -i ${bad%% *} -d ECC,TEMPERATURE,PERFORMANCE,CLOCK ; sudo dmesg | grep -i xid ; DIAG_GPUS=${bad%% *} rerun of this script repeats the VRAM + matmul test"
echo "NOTE: $NOTE"; echo "Full log: $OUT/diag.log"
