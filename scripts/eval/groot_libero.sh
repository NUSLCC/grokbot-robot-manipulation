#!/usr/bin/env bash
# GR00T N1.7 LIBERO eval (server-client), from examples/LIBERO/README.md (tag n1.7-release).
#   GROOT_SUITE=libero_10|libero_spatial|libero_object|libero_goal   ENV_NAME=libero_sim/<task>  (see that README's task list)
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection; ensure_uv
: "${GROOT_SUITE:=libero_10}"; : "${N_EPISODES:=5}"; : "${N_ENVS:=5}"; : "${PORT:=5555}"
: "${ENV_NAME:=libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it}"
SRC="$RSP_SRC/Isaac-GR00T"; [ -d "$SRC/.venv" ] || die "Run scripts/13_install_groot.sh first"
CLIENT_PY="$SRC/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
[ -x "$CLIENT_PY" ] || die "LIBERO client venv missing; run GROOT_SIMS=LIBERO scripts/13_install_groot.sh"
OUT="$RSP_LOGS/eval_groot_libero_$(stamp)"; mkdir -p "$OUT"; cd "$SRC"
CK="checkpoints/GR00T-N1.7-LIBERO/$GROOT_SUITE"
if [ ! -f "$CK/config.json" ]; then
  uv run hf download nvidia/GR00T-N1.7-LIBERO --include "$GROOT_SUITE/config.json" "$GROOT_SUITE/embodiment_id.json" \
    "$GROOT_SUITE/model-*.safetensors" "$GROOT_SUITE/model.safetensors.index.json" "$GROOT_SUITE/processor_config.json" \
    "$GROOT_SUITE/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
fi
uv run python gr00t/eval/run_gr00t_server.py --model-path "$CK" --embodiment-tag LIBERO_PANDA \
  --use-sim-policy-wrapper --port "$PORT" > "$OUT/server.log" 2>&1 &
SPID=$!; trap 'kill $SPID 2>/dev/null || true' EXIT
wait_port "$PORT" 1800 || { tail -n 40 "$OUT/server.log"; die "GR00T server did not start (gated Cosmos-Reason2-2B? run: uv run hf auth login)"; }
gpu_monitor_start "$OUT/gpu.csv"
"$CLIENT_PY" gr00t/eval/rollout_policy.py --n-episodes "$N_EPISODES" --policy-client-host 127.0.0.1 \
  --policy-client-port "$PORT" --max-episode-steps 720 --env-name "$ENV_NAME" --n-action-steps 8 --n-envs "$N_ENVS" \
  2>&1 | tee "$OUT/client.log"
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"; ok "Logs in $OUT"
