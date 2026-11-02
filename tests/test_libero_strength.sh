#!/usr/bin/env bash
# Stronger LIBERO evals: LIBERO_SUITES / LIBERO_EPISODES reach both eval commands (runner dry run), and the real
# openpi_pi05_libero.sh runs several suites with one server and prints the combined total (fake uv + fake client).
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; K="$(cd "$T/.." && pwd)"
TMP="$(mktemp -d)"; trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$TMP"' EXIT
export PATH="$T/fake-bin:$PATH" RSP_HOME="$TMP/rsp" RSP_FORCE_WSL=0
source "$K/scripts/lib/common.sh"; source "$K/scripts/lib/results.sh"
set +e
nfail=0
check() { if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: expected '$3', got '$2'"; nfail=$((nfail+1)); fi; }
ALL=libero_spatial,libero_object,libero_goal,libero_10

# --- runner passthrough ---
dry() { rm -rf "$TMP/runs"; env RSP_DRY_RUN=1 RSP_DRY_RUN_TMP="$TMP" RUNS_DIR="$TMP/runs" SKIP_INSTALL=1 \
  ONLY_STEPS="smolvla_libero_eval openpi_pi05_libero_eval" "$@" bash "$K/scripts/run_wsl_headless.sh" > /dev/null 2>&1; }
cmd() { grep -m1 '^\[dry-run\] cmd:' "$(ls "$TMP"/runs/*/*_"$1".log)" | tr -d '\\'; }   # printf %q escapes commas
dry QUICK=0 LIBERO_SUITES=all LIBERO_EPISODES=10
check "all -> smolvla SUITES + N_EPISODES" "$(cmd smolvla_libero_eval | grep -c "SUITES=$ALL N_EPISODES=10 bash")" 1
check "all -> openpi SUITES + TRIALS"      "$(cmd openpi_pi05_libero_eval | grep -c "SUITES=$ALL TRIALS=10 bash")" 1
dry QUICK=1 LIBERO_SUITES=libero_goal LIBERO_EPISODES=3
check "QUICK + override: last value wins (smolvla)" "$(cmd smolvla_libero_eval | grep -oE '(SUITES|N_EPISODES)=[^ ]+' | tail -n2 | tr '\n' ' ')" "SUITES=libero_goal N_EPISODES=3 "
check "QUICK + override: openpi"           "$(cmd openpi_pi05_libero_eval | grep -c 'SUITES=libero_goal TRIALS=3 bash')" 1
dry QUICK=0
check "unset -> script defaults (smolvla)" "$(cmd smolvla_libero_eval | grep -c 'SUITES=\|N_EPISODES=')" 0
check "unset -> script defaults (openpi)"  "$(cmd openpi_pi05_libero_eval | grep -c 'SUITES=\|TRIALS=')" 0

# --- real openpi script, multi-suite, against fakes ---
SRC="$RSP_HOME/src/openpi"; mkdir -p "$SRC/.venv" "$SRC/examples/libero/.venv/bin" "$TMP/bin"
PORT=$((20000 + RANDOM % 20000))
cat > "$TMP/bin/uv" <<'SH'
#!/usr/bin/env bash
p=8000; a=("$@"); for i in "${!a[@]}"; do [ "${a[$i]}" = --port ] && p="${a[$((i+1))]}"; done
exec python3 -m http.server "$p" --bind 127.0.0.1
SH
cat > "$SRC/examples/libero/.venv/bin/python" <<'SH'
#!/usr/bin/env bash
[ "$1" = -c ] && { echo "0.0 0.0"; exit 0; }   # video brightness check
case "$1" in *libero_config.py) exec python3 "$@" ;; esac
case "$*" in *openpi_client_obs.py*) echo "OBS_FINITE=1"; case "$*" in *--port*) echo "ROUNDTRIP_ACTIONS_FINITE=${FAKE_RT:-0}"; echo "ENV_CTRL_FINITE=0 ctrl_absmax=nan" ;; esac; exit 0 ;; esac   # real config writer (finds the fake third_party/libero)
[ -f "${LIBERO_CONFIG_PATH:-/nonexistent}/config.yaml" ] && echo "cfg $LIBERO_CONFIG_PATH" >> "$FAKE_ARGS.cfg"
a=("$@"); for i in "${!a[@]}"; do case "${a[$i]}" in --args.task-suite-name) s="${a[$((i+1))]}" ;; --args.num-trials-per-task) n="${a[$((i+1))]}" ;; --args.port) p="${a[$((i+1))]}" ;; --args.video-out-path) vd="${a[$((i+1))]}" ;; esac; done
echo "fakeclient $s $n" >> "$FAKE_ARGS"; echo "$p" >> "$FAKE_ARGS.port"; sleep "${FAKE_CLIENT_SLEEP:-0}"
mkdir -p "$vd"; : > "$vd/rollout_x_failure.mp4"; echo "INFO:root:Task suite: $s" >&2
[ -n "${FAKE_NAN:-}" ] && echo "WARNING: Nan, Inf or huge value in CTRL at ACTUATOR 0. The simulation is unstable. Time = 0.5000."
e=$((n*10)); case "$s" in libero_10) k=$((e-3)) ;; libero_goal) k=$((e-1)) ;; *) k=$e ;; esac; [ -n "${FAKE_ZERO:-}" ] && k=0
echo "INFO:root:Total success rate: $(awk -v k=$k -v e=$e 'BEGIN{print k/e}')" >&2; echo "INFO:root:Total episodes: $e" >&2
SH
chmod +x "$TMP/bin/uv" "$SRC/examples/libero/.venv/bin/python"
mkdir -p "$SRC/third_party/libero/libero/libero"; touch "$SRC/third_party/libero/libero/libero/__init__.py"
mkdir -p "$TMP/home/.libero"; printf 'benchmark_root: "/elsewhere/hf_libero/libero/libero"\n' > "$TMP/home/.libero/config.yaml"
runpi() { env -u LIBERO_CONFIG_PATH HOME="$TMP/home" PATH="$TMP/bin:$PATH" RSP_SKIP_GL_PROBE=1 PORT="$PORT" FAKE_ARGS="$TMP/args" "$@" \
  bash "$K/scripts/eval/openpi_pi05_libero.sh" > "$TMP/pi.log" 2>&1; }
: > "$TMP/args"; runpi SUITES="$ALL" TRIALS=10; rc=$?
check "openpi 4 suites: exit 0"             "$rc" 0
check "openpi 4 suites: one client per suite" "$(cut -d' ' -f2 "$TMP/args" | paste -sd,)" "$ALL"
check "openpi 4 suites: trials passed"      "$(grep -c ' 10$' "$TMP/args")" 4
check "openpi 4 suites: per-suite lines"    "$(grep -c '^Suite result: ' "$TMP/pi.log")" 4
check "openpi 4 suites: libero_10 line"     "$(grep '^Suite result: libero_10' "$TMP/pi.log")" "Suite result: libero_10 97/100"
check "openpi: own LIBERO config, openpi's copy" "$(grep -c "\"$SRC/third_party/libero/libero/libero\"" "$RSP_HOME/config/libero/openpi/config.yaml")" 1
check "openpi: client sees that config"     "$(sort -u "$TMP/args.cfg")" "cfg $RSP_HOME/config/libero/openpi"
check "openpi: warns about the shared ~/.libero" "$(grep -c 'config.yaml points at another LIBERO copy' "$TMP/pi.log")" 1
check "openpi: shared ~/.libero untouched"  "$(cat "$TMP/home/.libero/config.yaml")" 'benchmark_root: "/elsewhere/hf_libero/libero/libero"'
check "openpi 4 suites: combined metric"    "$(extract_metric openpi_eval "$TMP/pi.log")" "success 99.0% of 400 eps"
: > "$TMP/args"; runpi SUITE=libero_object TRIALS=2
check "openpi SUITE= (single) still works"  "$(cat "$TMP/args")" "fakeclient libero_object 2"
check "openpi single: metric"               "$(extract_metric openpi_eval "$TMP/pi.log")" "success 100.0% of 20 eps"
check "openpi single: no combined block"    "$(grep -c '^Suite result:' "$TMP/pi.log")" 0


# --- 0% guard (run 20261102-084220 on GPU 1: every episode 'Success: False', 0/400, yet the step PASSED) ---
: > "$TMP/args"; runpi SUITES="$ALL" TRIALS=10 FAKE_ZERO=1; rc=$?
check "openpi 0 successes -> rc 1"          "$rc" 1
check "openpi 0: says setup problem"        "$(grep -c 'pi0.5 had 0 successes in 400 episodes' "$TMP/pi.log")" 1
check "openpi 0: video check printed"       "$(grep -c '^Video check: .*mean/std pixel = 0.0 0.0' "$TMP/pi.log")" 1
: > "$TMP/args"; runpi SUITE=libero_spatial TRIALS=1 FAKE_ZERO=1 ALLOW_ZERO_SUCCESS=1; rc=$?
check "ALLOW_ZERO_SUCCESS=1 accepts 0%"     "$rc" 0
# runner: a PASSing pretrained eval with 0% becomes FAIL (fixture rebuilt from the user's grep of that run)
dry QUICK=0 LIBERO_SUITES=all LIBERO_EPISODES=10 FAKE_STEP_VARIANT=zero
S="$(ls "$TMP"/runs/*/SUMMARY.md)"
rowv() { grep -E "^\| $1 \|" "$S" | awk -F'|' -v c="$2" '{gsub(/^ +| +$/,"",$c); print $c}'; }
check "runner: pi0.5 0% -> FAIL"            "$(rowv openpi_pi05_libero_eval 3)" FAIL
check "runner: pi0.5 0% metric kept"        "$(rowv openpi_pi05_libero_eval 6)" "success 0.0% of 400 eps"
check "runner: 0% hint"                     "$(grep -c 'pretrained policy scored 0%' "$S")" 1
check "runner: smolvla (no zero fixture) still PASS" "$(rowv smolvla_libero_eval 3)" PASS
dry QUICK=0 FAKE_STEP_VARIANT=zero ALLOW_ZERO_SUCCESS=1
S="$(ls "$TMP"/runs/*/SUMMARY.md)"
check "runner: ALLOW_ZERO_SUCCESS=1 -> PASS" "$(rowv openpi_pi05_libero_eval 3)" PASS
# SmolVLA metric: 1 decimal (the real run printed 'success 63.24999999999999%')
mkdir -p "$TMP/ev"; echo '{"overall":{"pc_success":63.24999999999999}}' > "$TMP/ev/eval_info.json"
check "smolvla metric 1 decimal"            "$(printf 'Results in %s\n' "$TMP/ev" > "$TMP/sv.log"; extract_metric lerobot_eval "$TMP/sv.log")" "success 63.2%"
echo '{"overall":{"pc_success":0.0}}' > "$TMP/ev/eval_info.json"
check "smolvla 0.0 metric (FAILs in runner)" "$(extract_metric lerobot_eval "$TMP/sv.log")" "success 0.0%"

# --- our own server: busy port -> next free port; server stopped on exit; dead server -> fail ---
python3 -m http.server "$PORT" --bind 127.0.0.1 > /dev/null 2>&1 & BUSY=$!
for i in $(seq 1 50); do (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && break; sleep 0.1; done
: > "$TMP/args"; : > "$TMP/args.port"; runpi SUITE=libero_spatial TRIALS=1; rc=$?
check "busy port: exit 0"                   "$rc" 0
check "busy port: warns"                    "$(grep -c "port $PORT is already in use" "$TMP/pi.log")" 1
check "busy port: client uses our port"     "$(cat "$TMP/args.port")" "$((PORT+1))"
kill "$BUSY" 2>/dev/null; wait "$BUSY" 2>/dev/null
sleep 0.5
check "server stopped after the eval"       "$( (exec 3<>"/dev/tcp/127.0.0.1/$((PORT+1))") 2>/dev/null && echo still-listening || echo stopped)" stopped
cat > "$TMP/bin/uv" <<'SH'
#!/usr/bin/env bash
p=8000; a=("$@"); for i in "${!a[@]}"; do [ "${a[$i]}" = --port ] && p="${a[$((i+1))]}"; done
python3 -m http.server "$p" --bind 127.0.0.1 & h=$!; sleep 3; kill $h; echo "server crashed"; exit 1
SH
: > "$TMP/args"; runpi SUITES="$ALL" TRIALS=1 FAKE_CLIENT_SLEEP=2; rc=$?
check "server dies mid-run -> rc 1"         "$rc" 1
check "server dies mid-run: clear message"  "$(grep -c 'policy server exited (before suite' "$TMP/pi.log")" 1
check "server dies mid-run: stops before all 4 suites" "$([ "$(wc -l < "$TMP/args")" -lt 4 ] && echo stopped)" stopped
printf '#!/usr/bin/env bash\necho "OSError: [Errno 98] Address already in use"; exit 1\n' > "$TMP/bin/uv"
: > "$TMP/args"; SECONDS=0; runpi SUITE=libero_spatial TRIALS=1; rc=$?
check "server crashes at start -> rc 1"     "$rc" 1
check "server crashes at start: fast"       "$([ "$SECONDS" -lt 30 ] && echo fast || echo slow)" fast
check "server crashes at start: message + log" "$(grep -c 'policy server exited (while starting)\|Address already in use' "$TMP/pi.log")" 2
check "server crashes at start: no client"  "$(wc -l < "$TMP/args")" 0

# --- NaN actions (quick check on c576e43, 2026-11-02 13:23 SGT: 0/20, 'Nan, Inf or huge value in CTRL', video OK) ---
N="$T/fixtures/openpi_pi05_libero_eval.rc1.nan.log"
h="$(failure_hint openpi_pi05_libero_eval "$N" 1)"
check "NaN hint wins over the EGL hint"     "$(cut -c1-34 <<<"$h")" "the policy sent NaN/huge actions ("
check "NaN hint: no OpenGL/osmesa advice"   "$(grep -ci 'osmesa' <<<"$h")" 0
check "NaN hint names the diag script"      "$(grep -c 'scripts/diag/openpi_policy_check.sh' <<<"$h")" 1
check "EGL-only tail still gets the EGL hint" "$(grep -v 'Nan, Inf' "$N" > "$TMP/egl2.log"; failure_hint openpi_pi05_libero_eval "$TMP/egl2.log" 1 | cut -c1-19)" "OpenGL/EGL problem:"
printf '#!/usr/bin/env bash\np=8000; a=("$@"); for i in "${!a[@]}"; do [ "${a[$i]}" = --port ] && p="${a[$((i+1))]}"; done\nexec python3 -m http.server "$p" --bind 127.0.0.1\n' > "$TMP/bin/uv"
: > "$TMP/args"; runpi SUITE=libero_spatial TRIALS=2 FAKE_ZERO=1 FAKE_NAN=1; rc=$?
check "eval: NaN warning -> rc 1"           "$rc" 1
check "eval: says NaN actions + diag"       "$(grep -cE '^NaN actions: .* [0-9]+ times: .*openpi_policy_check.sh' "$TMP/pi.log")" 1

# --- scripts/diag/openpi_policy_check.sh against fakes: GPU NaN, CPU finite -> GPU-only verdict ---
cat > "$TMP/bin/uv" <<'SH'
#!/usr/bin/env bash
case "$1" in --version) echo "uv 0.9.0"; exit 0 ;; sync) echo "Would make no changes"; exit 0 ;; esac
case "$*" in
  *openpi_server_infer.py*) if [ "${JAX_PLATFORMS:-}" = cpu ]; then echo "CPU_ACTIONS_FINITE=${FAKE_CPU:-1}"; else echo "PARAMS_NONFINITE=${FAKE_PARAMS:-0}"; echo "GPU_ACTIONS_FINITE=${FAKE_GPU:-0}"; fi ;;
  *serve_policy.py*) p=8000; a=("$@"); for i in "${!a[@]}"; do [ "${a[$i]}" = --port ] && p="${a[$((i+1))]}"; done; exec python3 -m http.server "$p" --bind 127.0.0.1 ;;
esac
SH
chmod +x "$TMP/bin/uv"
rundiag() { env -u LIBERO_CONFIG_PATH HOME="$TMP/home" PATH="$TMP/bin:$PATH" RSP_SKIP_GL_PROBE=1 PORT="$PORT" FAKE_ARGS="$TMP/args" "$@" \
  bash "$K/scripts/diag/openpi_policy_check.sh" > "$TMP/diag.txt" 2>&1; }
rundiag; rc=$?
check "diag: exit 0"                        "$rc" 0
check "diag: summary keys"                  "$(grep -cE '^(OBS_FINITE|PARAMS_NONFINITE|GPU_ACTIONS_FINITE|CPU_ACTIONS_FINITE|ROUNDTRIP_ACTIONS_FINITE)=[0-9]' "$TMP/diag.txt")" 11
check "diag: GPU NaN + CPU ok -> GPU-only"  "$(grep -c '^VERDICT: GPU-only NaN' "$TMP/diag.txt")" 1
rundiag FAKE_CPU=0
check "diag: NaN on both -> model/inputs"   "$(grep -c '^VERDICT: NaN on GPU and CPU' "$TMP/diag.txt")" 1
rundiag FAKE_PARAMS=12
check "diag: NaN params -> corrupted cache" "$(grep -c '^VERDICT: checkpoint params contain NaN' "$TMP/diag.txt")" 1
rundiag FAKE_GPU=1 FAKE_RT=0 DIAG_CPU=0
check "diag: in-process ok, round trip NaN" "$(grep -c '^VERDICT: model is fine in-process but NaN through the server' "$TMP/diag.txt")" 1
check "diag: DIAG_CPU=0 skips the CPU run"  "$(grep -c '^CPU_ACTIONS_FINITE=[0-9]' "$TMP/diag.txt")" 0
check "diag: server stopped afterwards"     "$(sleep 0.5; (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null && echo listening || echo stopped)" stopped
check "diag: log file written"              "$(ls "$RSP_HOME"/logs/diag_openpi_*/diag.log | wc -l | tr -d ' ')" 4
echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
