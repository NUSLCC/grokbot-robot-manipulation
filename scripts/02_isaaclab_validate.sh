#!/usr/bin/env bash
# 02_isaaclab_validate.sh - detect and validate the EXISTING Isaac Lab 3.0.0 + Isaac Sim 6.2.0 install.
# Read-only with respect to the Isaac environment: uses `uv run --no-sync` / isaaclab.sh, never installs.
#
# Env vars:
#   ISAACLAB_PATH, ISAACLAB_MODE, ISAAC_PY, ISAACLAB_CONDA_ENV   see scripts/lib/isaaclab_env.sh
#   SMOKE_TASKS="Isaac-Reach-Franka Isaac-Lift-Franka"           tasks for the RL smoke test
#   SMOKE_NUM_ENVS=64  SMOKE_ITERS=3  SMOKE_TIMEOUT=1800
#   SMOKE_PHYSICS="default isaacsim_physx"   'default' = Newton (Isaac Lab 3.0 default); isaacsim_physx exercises Isaac Sim 6.2
#   SKIP_SMOKE=1                              only run detection + version checks
#   SKIP_ISAAC=1                              skip everything (exit code 77 = SKIP for run_wsl_headless.sh)
#   ISAACLAB_HEADLESS_FLAGS                   default "--viz none --livestream 0" (Isaac Lab 3.0 has no --headless)
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/isaaclab_env.sh"
isaac_skip_if_requested

: "${SMOKE_TASKS:=Isaac-Reach-Franka Isaac-Lift-Franka}"   # IDs verified in release/3.0.0 source (core/reach, core/lift)
: "${SMOKE_NUM_ENVS:=64}"
: "${SMOKE_ITERS:=3}"
: "${SMOKE_TIMEOUT:=1800}"
: "${SMOKE_PHYSICS:=default isaacsim_physx}"

OUT="$RSP_LOGS/isaaclab_validate_$(stamp)"; mkdir -p "$OUT"
exec > >(tee "$OUT/report.txt") 2>&1
RESULTS=(); FAILED_LOGS=()
rec() { RESULTS+=("$1|$2|$3"); }

log "== Detecting Isaac Lab / Isaac Sim =="
isaaclab_wsl_note
isaaclab_detect
isaaclab_summary
[ "$ISAACLAB_MODE" != none ] || die "Isaac Lab not found. Export ISAACLAB_PATH=/path/to/IsaacLab (and ISAACLAB_CONDA_ENV or ISAAC_PY if it lives in conda/a venv)."

V="$(cat "$ISAACLAB_PATH/VERSION" 2>/dev/null || echo unknown)"
if [ "$V" = "$EXPECTED_ISAACLAB_VERSION" ]; then rec "Isaac Lab VERSION" PASS "$V"; else rec "Isaac Lab VERSION" WARN "found '$V', expected $EXPECTED_ISAACLAB_VERSION"; fi
GITREF="$(git -C "$ISAACLAB_PATH" describe --tags --always 2>/dev/null || echo n/a) ($(git -C "$ISAACLAB_PATH" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a))"
echo "  git: $GITREF"
if [ "$ISAACLAB_MODE" = sh ] && [ -z "${ISAACLAB_CONDA_ENV:-}" ] && [ ! -x "$ISAACLAB_PATH/env_isaaclab/bin/python" ] && [ ! -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then
  warn "isaaclab.sh would fall back to system python3. If Isaac Lab lives in a conda env, set ISAACLAB_CONDA_ENV=<name>."
fi

log "== Python stack inside the Isaac env (read-only) =="
if [ -n "${ISAAC_PY:-}" ]; then
  cat > "$OUT/probe.py" <<'PY'
import importlib.metadata as m, json, sys
def v(p):
    try: return m.version(p)
    except Exception: return None
info = {"python": sys.version.split()[0], "executable": sys.executable}
for p in ["isaaclab","isaaclab-tasks","isaaclab-rl","isaaclab-mimic","isaacsim","isaacsim-kernel","torch","torchvision",
          "warp-lang","transformers","numpy","rsl-rl-lib","skrl","stable-baselines3","rl-games","robomimic","newton","rlinf"]:
    info[p] = v(p)
try:
    import torch
    info["torch.cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name; info["vram_gib"] = round(p.total_memory/2**30, 1)
        info["capability"] = f"{p.major}.{p.minor}"
        x = torch.randn(2048, 2048, device="cuda"); info["matmul_ok"] = bool((x @ x).isfinite().all().item())
except Exception as e:
    info["torch_error"] = repr(e)
print(json.dumps(info, indent=1))
PY
  if [ "$ISAACLAB_MODE" = uv ]; then PROBE=(bash -c "cd '$ISAACLAB_PATH' && uv run --no-sync python '$OUT/probe.py'")
  else PROBE=("$ISAAC_PY" "$OUT/probe.py"); fi
  if timeout 600 "${PROBE[@]}" > "$OUT/probe.json" 2> "$OUT/probe.err"; then
    cat "$OUT/probe.json"
    j() { python3 -c "import json,sys;d=json.load(open('$OUT/probe.json'));print(d.get('$1'))"; }
    PYV="$(j python)"; TV="$(j torch)"; TC="$(j torch.cuda)"; CA="$(j cuda_available)"; SIMV="$(j isaacsim)"
    case "$PYV" in 3.12.*) rec "Python" PASS "$PYV (Isaac Lab 3.0 requires 3.12)";; *) rec "Python" FAIL "$PYV (Isaac Lab 3.0 requires >=3.12,<3.13)";; esac
    [ "$TV" = "None" ] && rec "PyTorch" FAIL "not importable" || {
      case "$TV" in 2.11.0*) rec "PyTorch" PASS "$TV cuda=$TC (release/3.0.0 pins torch==2.11.0 cu130)";;
                   *) rec "PyTorch" WARN "$TV cuda=$TC (release/3.0.0 pins torch==2.11.0 cu130; something replaced it?)";; esac; }
    [ "$CA" = "True" ] && rec "CUDA from Isaac env" PASS "$(j gpu), $(j vram_gib) GiB, matmul_ok=$(j matmul_ok)" || rec "CUDA from Isaac env" FAIL "torch.cuda.is_available()=$CA $(j torch_error)"
    if [ "$SIMV" != "None" ]; then
      case "$SIMV" in 6.2.*) rec "Isaac Sim (pip wheel)" PASS "isaacsim $SIMV";; *) rec "Isaac Sim (pip wheel)" WARN "isaacsim $SIMV (expected 6.2.x)";; esac
    elif [ -n "${ISAACSIM_ROOT:-}" ]; then
      SV="$(cat "$ISAACSIM_ROOT/VERSION" 2>/dev/null | sed -n 1p || echo '?')"
      rec "Isaac Sim (binary/_isaac_sim)" PASS "$ISAACSIM_ROOT VERSION=$SV (verify it reads 6.2.x)"
    else rec "Isaac Sim" WARN "no isaacsim wheel in env and no _isaac_sim / ~/isaacsim found; Newton-only tasks still work"; fi
    [ "$(j isaaclab-mimic)" != "None" ] && rec "isaaclab_mimic" PASS "$(j isaaclab-mimic)" || rec "isaaclab_mimic" INFO "not installed (only needed for Mimic workflow)"
    [ "$(j robomimic)" != "None" ] && rec "robomimic" PASS "$(j robomimic)" || rec "robomimic" INFO "not installed (Mimic BC training needs the 'mimic' extra)"
    [ "$(j rsl-rl-lib)" != "None" ] && rec "rsl_rl" PASS "$(j rsl-rl-lib)" || rec "rsl_rl" WARN "rsl-rl-lib not found; smoke test uses rsl_rl"
  else
    rec "Python probe" FAIL "see $OUT/probe.err"; cat "$OUT/probe.err" | tail -n 20
  fi
else
  rec "Python probe" WARN "no interpreter resolved; set ISAAC_PY or ISAACLAB_CONDA_ENV"
fi

log "== Driver vs Isaac requirements =="
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n 1p | tr -d ' ' || true)"; DRV="${DRV:-0}"
ver_ge "$DRV" "$DRIVER_TESTED_ISAAC" && rec "Driver" PASS "$DRV >= $DRIVER_TESTED_ISAAC (Isaac Sim 6.2 tested)" || rec "Driver" WARN "$DRV < $DRIVER_TESTED_ISAAC"

if [ "${SKIP_SMOKE:-0}" = 1 ]; then warn "SKIP_SMOKE=1"; else
  export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
  export ISAACLAB_PATH ISAACLAB_MODE ISAAC_PY
  RUN="$(dirname "$0")/lib/isaaclab_run.sh"
  log "== list_envs (manipulation tasks) =="
  if timeout 900 "$RUN" cli list_envs > "$OUT/list_envs.txt" 2>&1; then
    grep -oE 'Isaac(Contrib)?-[A-Za-z0-9-]*(Franka|Lift|Reach|Drawer|Stack|Allegro|Cabinet|Mimic)[A-Za-z0-9-]*' "$OUT/list_envs.txt" | sort -u > "$OUT/manipulation_tasks.txt" || true
    rec "list_envs" PASS "$(wc -l < "$OUT/manipulation_tasks.txt") manipulation task IDs -> $OUT/manipulation_tasks.txt"
    head -n 40 "$OUT/manipulation_tasks.txt" | sed 's/^/  /'
  else rec "list_envs" FAIL "see $OUT/list_envs.txt"; FAILED_LOGS+=("$OUT/list_envs.txt"); fi

  for task in $SMOKE_TASKS; do
    for phys in $SMOKE_PHYSICS; do
      tag="${task}__${phys}"; extra=()
      [ "$phys" != default ] && extra=("physics=$phys")
      log "== Headless RL smoke: $task  physics=$phys  num_envs=$SMOKE_NUM_ENVS  iters=$SMOKE_ITERS =="
      gpu_monitor_start "$OUT/gpu_$tag.csv"
      t0=$(date +%s); rc=0
      timeout "$SMOKE_TIMEOUT" "$RUN" cli train --rl_library rsl_rl --task "$task" \
        --num_envs "$SMOKE_NUM_ENVS" --max_iterations "$SMOKE_ITERS" --seed 1 ${extra[@]+"${extra[@]}"} $ISAACLAB_HEADLESS_FLAGS \
        > "$OUT/smoke_$tag.log" 2>&1 || rc=$?
      peak="$(gpu_monitor_stop)"; dt=$(( $(date +%s) - t0 ))
      if [ "$rc" = 0 ]; then rec "smoke $task [$phys]" PASS "${dt}s, peak GPU mem ${peak} MiB (whole GPU)"
      else rec "smoke $task [$phys]" FAIL "rc=$rc after ${dt}s; log $OUT/smoke_$tag.log"; FAILED_LOGS+=("$OUT/smoke_$tag.log"); fi
    done
  done
fi

echo; log "== Isaac Lab validation summary =="
printf '%-34s %-5s %s\n' CHECK RESULT DETAIL
nfail=0
for r in "${RESULTS[@]}"; do IFS='|' read -r a b c <<<"$r"; printf '%-34s %-5s %s\n' "$a" "$b" "$c"; [ "$b" = FAIL ] && nfail=$((nfail+1)); done
echo "Logs: $OUT"
cat <<'TIP'
Notes:
 * 'default' physics = Newton (Warp); it does NOT exercise Isaac Sim. The isaacsim_physx run does.
 * If a physics preset is rejected for a task, run list_envs and check that task's presets.
 * No changes were made to the Isaac environment.
TIP
# Failed logs go LAST so the runner's console tail and SUMMARY.md show the real error, not just this table.
for f in ${FAILED_LOGS[@]+"${FAILED_LOGS[@]}"}; do
  echo; log "== Last 30 lines of $f =="
  tail -n 30 "$f" 2>/dev/null || echo "  (log missing)"
  e="$(grep -aE '(Error|Exception|error:|FATAL|Segmentation fault|core dumped)' "$f" 2>/dev/null | grep -avE '^\s*(File "|warnings\.warn)' | tail -n1 | sed 's/^[[:space:]]*//' | cut -c1-160 || true)"
  [ -n "$e" ] && echo "key error: $e"
done
exit "$nfail"
