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
a=("$@"); for i in "${!a[@]}"; do case "${a[$i]}" in --args.task-suite-name) s="${a[$((i+1))]}" ;; --args.num-trials-per-task) n="${a[$((i+1))]}" ;; esac; done
echo "fakeclient $s $n" >> "$FAKE_ARGS"
e=$((n*10)); case "$s" in libero_10) k=$((e-3)) ;; libero_goal) k=$((e-1)) ;; *) k=$e ;; esac
echo "INFO:root:Total success rate: $(awk -v k=$k -v e=$e 'BEGIN{print k/e}')" >&2; echo "INFO:root:Total episodes: $e" >&2
SH
chmod +x "$TMP/bin/uv" "$SRC/examples/libero/.venv/bin/python"
runpi() { env PATH="$TMP/bin:$PATH" RSP_SKIP_GL_PROBE=1 PORT="$PORT" FAKE_ARGS="$TMP/args" "$@" \
  bash "$K/scripts/eval/openpi_pi05_libero.sh" > "$TMP/pi.log" 2>&1; }
: > "$TMP/args"; runpi SUITES="$ALL" TRIALS=10; rc=$?
check "openpi 4 suites: exit 0"             "$rc" 0
check "openpi 4 suites: one client per suite" "$(cut -d' ' -f2 "$TMP/args" | paste -sd,)" "$ALL"
check "openpi 4 suites: trials passed"      "$(grep -c ' 10$' "$TMP/args")" 4
check "openpi 4 suites: per-suite lines"    "$(grep -c '^Suite result: ' "$TMP/pi.log")" 4
check "openpi 4 suites: libero_10 line"     "$(grep '^Suite result: libero_10' "$TMP/pi.log")" "Suite result: libero_10 97/100"
check "openpi 4 suites: combined metric"    "$(extract_metric openpi_eval "$TMP/pi.log")" "success 99.0% of 400 eps"
: > "$TMP/args"; runpi SUITE=libero_object TRIALS=2
check "openpi SUITE= (single) still works"  "$(cat "$TMP/args")" "fakeclient libero_object 2"
check "openpi single: metric"               "$(extract_metric openpi_eval "$TMP/pi.log")" "success 100.0% of 20 eps"
check "openpi single: no combined block"    "$(grep -c '^Suite result:' "$TMP/pi.log")" 0

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
