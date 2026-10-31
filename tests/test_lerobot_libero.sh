#!/usr/bin/env bash
# Unit tests: LeRobot CLI flag detection (real `lerobot-train --help` output of v0.6.1 and main, plus the older
# --eval_freq shape), the LeRobot train scripts against a fake lerobot-train that rejects unknown flags like the
# real one, the non-interactive LIBERO config writer, and the hints for these failures.
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; K="$(cd "$T/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export PATH="$T/fake-bin:$PATH" RSP_HOME="$TMP/rsp" MUJOCO_GL=osmesa
# shellcheck source=../scripts/lib/common.sh
source "$K/scripts/lib/common.sh"
# shellcheck source=../scripts/lib/results.sh
source "$K/scripts/lib/results.sh"
set +e
nfail=0
check() { if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: expected '$3', got '$2'"; nfail=$((nfail+1)); fi; }
H="$T/fixtures/lerobot"

# --- 1. flag detection from --help ---
compat() { LEROBOT_TRAIN_HELP_FILE="$1" lerobot_train_compat /nonexistent >/dev/null 2>&1; echo "$LR_EVAL_FREQ_FLAG ${LR_TRACKER_OFF[*]:-none}"; }
check "v0.6.1 help: --env_eval_freq + wandb" "$(compat "$H/train_help.v0.6.1.txt")" "--env_eval_freq --wandb.enable=false"
check "main help: --env_eval_freq + trackio" "$(compat "$H/train_help.main-3f797de5.txt")" "--env_eval_freq --trackio.enable=false"
check "<=v0.5 help: --eval_freq + wandb"     "$(compat "$H/train_help.v0.5.txt")" "--eval_freq --wandb.enable=false"
echo "usage: lerobot-train [-h]" > "$TMP/empty_help.txt"
check "unknown help: default, no tracker"    "$(compat "$TMP/empty_help.txt")" "--env_eval_freq none"

# --- 2. train scripts vs a fake lerobot-train that rejects flags missing from its --help (like draccus) ---
ENVD="$RSP_HOME/envs/lerobot/bin"; mkdir -p "$ENVD"
cat > "$ENVD/lerobot-train" <<'FAKE'
#!/usr/bin/env bash
[ "${1:-}" = --help ] && { cat "$FAKE_HELP"; exit 0; }
bad=""
for a in "$@"; do
  f="${a%%=*}"; f="${f#--}"
  case "$f" in policy.*|env.*) continue ;; esac   # choice-type sub-configs are only listed once a type is chosen
  grep -qE -- "--$f([^a-zA-Z0-9_.]|\$)" "$FAKE_HELP" || bad="$bad $a"
done
[ -n "$bad" ] && { echo "lerobot-train: error: unrecognized arguments:$bad"; exit 2; }
echo "fake lerobot-train ok: $*"
FAKE
chmod +x "$ENVD/lerobot-train"; ln -sf "$(command -v python3)" "$ENVD/python"
for h in v0.6.1 main-3f797de5 v0.5; do
  for s in lerobot_act_aloha lerobot_diffusion_pusht lerobot_smolvla_libero lerobot_pi05_libero_expert_only; do
    out="$(FAKE_HELP="$H/train_help.$h.txt" STEPS=10 EVAL_FREQ=5 N_EVAL=1 bash "$K/scripts/train/$s.sh" 2>&1)"; rc=$?
    check "$s accepted by lerobot-train $h" "$rc" 0
    [ "$rc" = 0 ] || echo "$out" | tail -n 3 | sed 's/^/      | /'
  done
done
out="$(FAKE_HELP="$H/train_help.v0.6.1.txt" STEPS=10 EVAL_FREQ=7 N_EVAL=1 bash "$K/scripts/train/lerobot_act_aloha.sh" 2>&1)"
out="$(grep '^fake lerobot-train ok' <<<"$out")"
check "act passes --env_eval_freq=7 on v0.6.1" "$(grep -c -- '--env_eval_freq=7 ' <<<"$out")" 1
check "act passes --wandb.enable=false on v0.6.1" "$(grep -c -- '--wandb.enable=false' <<<"$out")" 1
check "no script hard-codes --eval_freq= / --wandb.enable=" \
  "$(grep -lE -- '--eval_freq=|--wandb\.enable=' "$K"/scripts/train/lerobot_*.sh "$K"/scripts/eval/lerobot_*.sh | wc -l)" 0

# --- 3. LIBERO config writer (fake libero package with LIBERO's prompt logic; importing it without a config
#        hits input() exactly like the real one) ---
SITE="$TMP/site"; mkdir -p "$SITE/libero/libero"; : > "$SITE/libero/__init__.py"
cat > "$SITE/libero/libero/__init__.py" <<'PYF'
import os
open(os.environ["LIBERO_IMPORT_MARKER"], "w").close()
libero_config_path = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
config_file = os.path.join(libero_config_path, "config.yaml")
if not os.path.exists(config_file):
    answer = input("Do you want to specify a custom path for the dataset folder? (Y/N): ")
PYF
export LIBERO_IMPORT_MARKER="$TMP/imported"
import_libero() { PYTHONPATH="$SITE" python3 -c "import libero.libero" </dev/null 2>&1 | tail -n1; }
check "no config: LIBERO import hits EOFError" "$(LIBERO_CONFIG_PATH="$TMP/none" import_libero)" "EOFError: EOF when reading a line"
rm -f "$LIBERO_IMPORT_MARKER"
( unset LIBERO_CONFIG_PATH; libero_config python3 lerobot "$SITE" >/dev/null 2>&1; echo "$LIBERO_CONFIG_PATH" > "$TMP/lcp" )
C="$RSP_HOME/config/libero/lerobot"
check "libero_config exports per-env dir"     "$(cat "$TMP/lcp")" "$C"
check "writer did not import libero.libero"   "$([ -e "$LIBERO_IMPORT_MARKER" ] && echo imported || echo no)" no
check "config.yaml written"                   "$([ -s "$C/config.yaml" ] && echo y)" y
check "config has LIBERO's 5 keys"            "$(grep -cE '^(benchmark_root|bddl_files|init_states|datasets|assets): "' "$C/config.yaml")" 5
check "benchmark_root = package dir"          "$(sed -n 's/^benchmark_root: "\(.*\)"$/\1/p' "$C/config.yaml")" "$SITE/libero/libero"
check "bddl_files = LIBERO default string"    "$(sed -n 's/^bddl_files: "\(.*\)"$/\1/p' "$C/config.yaml")" "$SITE/libero/libero/./bddl_files"
if python3 -c 'import yaml' 2>/dev/null; then
  check "valid YAML (PyYAML)" "$(python3 -c 'import sys,yaml;d=yaml.safe_load(open(sys.argv[1]));print(sorted(d)==sorted(["assets","bddl_files","benchmark_root","datasets","init_states"]))' "$C/config.yaml")" True
else
  check "valid YAML subset (JSON-quoted scalars)" "$(python3 -c 'import json,sys
ok=all(json.loads(l.split(": ",1)[1]) for l in open(sys.argv[1]) if l.strip())
print(ok)' "$C/config.yaml")" True
fi
check "with config: LIBERO import has no prompt" "$(LIBERO_CONFIG_PATH="$C" import_libero)" ""
check "rerun keeps the config" "$(PYTHONPATH="$SITE" python3 "$K/scripts/lib/libero_config.py" "$C" | cut -c1-19)" "libero_config: kept"
sed -i 's|^benchmark_root: .*|benchmark_root: "/old/env/libero/libero"|' "$C/config.yaml"
PYTHONPATH="$SITE" python3 "$K/scripts/lib/libero_config.py" "$C" >/dev/null
check "stale config rewritten"   "$(sed -n 's/^benchmark_root: "\(.*\)"$/\1/p' "$C/config.yaml")" "$SITE/libero/libero"
check "stale config backed up"   "$([ -s "$C/config.yaml.bak" ] && echo y)" y
check "no libero: writer exits 3" "$(PYTHONPATH="$TMP/empty" python3 "$K/scripts/lib/libero_config.py" "$TMP/x" 2>/dev/null; echo $?)" 3
check "every LIBERO eval writes the config" \
  "$(grep -L 'libero_config ' "$K"/scripts/eval/*libero*.sh | wc -l)" 0
check "no install pipes 'N' into LIBERO any more" "$(grep -l "printf 'N" "$K"/scripts/*.sh | wc -l)" 0

# --- 4. hints, from the real logs of the user's run ---
check "hint: renamed flag" "$(failure_hint act_aloha "$T/fixtures/act_aloha.rc2.log" 2)" \
  "the tool's version does not accept --eval_freq=2000 (flag renamed upstream): git pull the kit, or check the pinned ref in scripts/lib/versions.env"
check "hint: LIBERO prompt" "$(failure_hint smolvla_libero_eval "$T/fixtures/smolvla_libero_eval.rc1.log" 1 | cut -c1-41)" \
  "LIBERO asked its first-run question (no c"
printf 'Continue? [y/N]\nEOFError: EOF when reading a line\n' > "$TMP/eof.log"
check "hint: generic interactive prompt" "$(failure_hint maniskill_ppo "$TMP/eof.log" 1 | cut -c1-34)" "a tool asked an interactive questi"

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
