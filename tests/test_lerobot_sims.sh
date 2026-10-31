#!/usr/bin/env bash
# Unit tests: the ALOHA / PushT eval simulators for the LeRobot train scripts.
#  - eval envs run in-process (--eval.use_async_envs=false): LeRobot v0.6.1's forkserver eval workers never import
#    gym_aloha / gym_pusht and fail with "NamespaceNotFound" even when the package is installed (lerobot#4931)
#  - lerobot_sim_ensure checks the sim BEFORE training, auto-installs a missing extra (or fails fast)
#  - lib/lerobot_sim_probe.py exit codes, EVAL_ONLY=1 re-evaluation of the newest checkpoint, failure hints
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

# --- 1. in-process eval flag from --help ---
sync_flag() { LEROBOT_TRAIN_HELP_FILE="$1" lerobot_train_compat /nonexistent >/dev/null 2>&1; echo "${LR_EVAL_SYNC[*]:-none}"; }
for h in v0.6.1 main-3f797de5 v0.5; do
  check "$h help: eval envs in-process" "$(sync_flag "$H/train_help.$h.txt")" "--eval.use_async_envs=false"
done
echo "usage: lerobot-train [-h]" > "$TMP/empty_help.txt"
check "unknown help: no sync flag"     "$(sync_flag "$TMP/empty_help.txt")" none
check "LEROBOT_ASYNC_EVAL=1 opts out"  "$(LEROBOT_ASYNC_EVAL=1 sync_flag "$H/train_help.v0.6.1.txt")" none

# --- 2. probe script exit codes, with fake gymnasium / gym_* packages (no real sims needed) ---
P="$K/scripts/lib/lerobot_sim_probe.py"; SITE="$TMP/site"; mkdir -p "$SITE/gymnasium" "$SITE/gym_pusht" "$SITE/gym_aloha"
cat > "$SITE/gymnasium/__init__.py" <<'PYF'
import os
class _Frame:  shape = (96, 96, 3)
class _Env:
    def reset(self, seed=None): return {}, {}
    def render(self):
        if os.environ.get("FAKE_RENDER_FAIL"): raise RuntimeError("no GL context")
        return _Frame()
    def close(self): pass
def make(env_id, **kw):
    assert kw.get("render_mode") == "rgb_array", kw
    return _Env()
PYF
: > "$SITE/gym_pusht/__init__.py"; : > "$SITE/gym_aloha/__init__.py"
probe() { PYTHONPATH="$SITE" python3 "$P" "$@" 2>&1; echo "rc=$?"; }
check "probe ok: pusht"            "$(probe pusht | tr '\n' ' ')" "sim probe ok: gym_pusht/PushT-v0 render (96, 96, 3) (SDL_VIDEODRIVER=dummy) rc=0 "
check "probe ok: aloha default task" "$(probe aloha | head -n1 | cut -d' ' -f4)" "gym_aloha/AlohaTransferCube-v0"
check "probe: package missing -> 3" "$(probe xarm | tail -n1)" "rc=3"
check "probe: render fails -> 4"    "$(FAKE_RENDER_FAIL=1 probe pusht | tail -n1)" "rc=4"
check "probe: render error reported" "$(FAKE_RENDER_FAIL=1 probe pusht | head -n1 | grep -c 'RuntimeError: no GL context')" 1

# --- 3. lerobot_sim_ensure: auto-install a missing extra, fail fast otherwise ---
FP="$TMP/fake_probe.sh"; cat > "$FP" <<'FAKE'
#!/usr/bin/env bash
# fake probe: "installed" once the fake uv created the marker
shift   # python passes the script path first
if [ -e "$FAKE_SIM_MARKER" ]; then echo "sim probe ok: gym_$1/T render (1, 1, 3)"; exit 0; fi
echo "sim probe: gym_$1 is not installed"; exit "${FAKE_PROBE_RC:-3}"
FAKE
FUV="$TMP/fake_uv.sh"; cat > "$FUV" <<'FAKE'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_UV_LOG"; [ "${FAKE_UV_FAIL:-0}" = 1 ] && exit 1; touch "$FAKE_SIM_MARKER"
FAKE
chmod +x "$FP" "$FUV"
# bash as the "python": runs the fake probe script
export RSP_SIM_PROBE="$FP" RSP_UV="$FUV" FAKE_UV_LOG="$TMP/uv.log" FAKE_SIM_MARKER="$TMP/installed"
mkdir -p "$RSP_SRC/lerobot"
ensure() { ( lerobot_sim_ensure bash "$@" ) 2>&1; echo "rc=$?"; }
out="$(ensure pusht)"
check "missing sim: auto-installs"         "$(tail -n1 <<<"$out")" "rc=0"
check "auto-install logs what it does"     "$(grep -c "adding LeRobot's 'pusht' extra" <<<"$out")" 1
check "auto-install = lerobot[pusht] editable, same torch backend" "$(cat "$TMP/uv.log")" \
  "pip install --python bash --torch-backend $LEROBOT_TORCH_BACKEND -e $RSP_SRC/lerobot[pusht]"
check "already installed: no uv call"      "$(ensure pusht >/dev/null; wc -l < "$TMP/uv.log" | tr -d ' ')" 1
rm -f "$FAKE_SIM_MARKER"
out="$(LEROBOT_AUTO_EXTRAS=0 ensure aloha)"
check "LEROBOT_AUTO_EXTRAS=0: fails fast"  "$(tail -n1 <<<"$out")" "rc=1"
check "fail-fast message names the fix"    "$(grep -c 'run scripts/10_install_lerobot.sh (it installs the aloha/pusht extras)' <<<"$out")" 1
out="$(FAKE_UV_FAIL=1 ensure aloha)"
check "uv failure: fails"                  "$(tail -n1 <<<"$out")" "rc=1"
out="$(FAKE_PROBE_RC=4 ensure aloha)"
check "render failure: fails with GL hint" "$(grep -c 'does not work headless.*MUJOCO_GL=osmesa' <<<"$out")" 1

# --- 4. train scripts: sim check before training, in-process eval, EVAL_ONLY ---
ENVD="$RSP_HOME/envs/lerobot/bin"; mkdir -p "$ENVD"
for b in lerobot-train lerobot-eval; do cat > "$ENVD/$b" <<FAKE
#!/usr/bin/env bash
[ "\${1:-}" = --help ] && { cat "$H/train_help.v0.6.1.txt"; exit 0; }
echo "fake $b: \$*" | tee -a "$TMP/calls.log"
FAKE
chmod +x "$ENVD/$b"; done
ln -sf "$(command -v bash)" "$ENVD/python"
touch "$FAKE_SIM_MARKER"
for s in act_aloha diffusion_pusht; do
  : > "$TMP/calls.log"
  out="$(STEPS=10 EVAL_FREQ=5 N_EVAL=2 bash "$K/scripts/train/lerobot_$s.sh" 2>&1)"; rc=$?
  check "$s: runs"                          "$rc" 0
  check "$s: sim probed before training"    "$(grep -n -e 'sim probe ok' -e 'fake lerobot-train' <<<"$out" | head -n1 | grep -c 'sim probe ok')" 1
  check "$s: eval envs in-process"          "$(grep -c -- '--eval.use_async_envs=false' "$TMP/calls.log")" 1
  rm -f "$FAKE_SIM_MARKER"; : > "$TMP/calls.log"
  out="$(LEROBOT_AUTO_EXTRAS=0 STEPS=10 EVAL_FREQ=5 N_EVAL=2 bash "$K/scripts/train/lerobot_$s.sh" 2>&1)"; rc=$?
  check "$s: missing sim fails before training" "$rc:$(wc -l < "$TMP/calls.log" | tr -d ' ')" "1:0"
  touch "$FAKE_SIM_MARKER"
done
# EVAL_ONLY picks the newest checkpoint, never trains
for d in act_aloha_transfer_cube_20261031-160000 act_aloha_transfer_cube_20261031-164000 diffusion_pusht_20261031-164400; do
  mkdir -p "$RSP_CKPT/$d/checkpoints/last/pretrained_model"; echo '{}' > "$RSP_CKPT/$d/checkpoints/last/pretrained_model/config.json"
done
: > "$TMP/calls.log"; out="$(EVAL_ONLY=1 N_EVAL=3 bash "$K/scripts/train/lerobot_act_aloha.sh" 2>&1)"; rc=$?
check "EVAL_ONLY act: ok"                 "$rc" 0
check "EVAL_ONLY act: no training"        "$(grep -c 'fake lerobot-train' "$TMP/calls.log")" 0
check "EVAL_ONLY act: newest checkpoint"  "$(grep -oE 'policy.path=[^ ]+' "$TMP/calls.log")" \
  "policy.path=$RSP_CKPT/act_aloha_transfer_cube_20261031-164000/checkpoints/last/pretrained_model"
check "EVAL_ONLY act: aloha task, in-process, 3 eps" \
  "$(grep -c -- '--env.type=aloha --env.task=AlohaTransferCube-v0 --eval.n_episodes=3 --eval.batch_size=3 --eval.use_async_envs=false' "$TMP/calls.log")" 1
: > "$TMP/calls.log"; out="$(EVAL_ONLY=1 CKPT="$RSP_CKPT/diffusion_pusht_20261031-164400/checkpoints/last/pretrained_model" bash "$K/scripts/train/lerobot_diffusion_pusht.sh" 2>&1)"; rc=$?
check "EVAL_ONLY diffusion with CKPT: ok" "$rc:$(grep -c 'fake lerobot-eval.*--env.type=pusht' "$TMP/calls.log")" "0:1"
rm -rf "$RSP_CKPT"/diffusion_pusht_*
out="$(EVAL_ONLY=1 bash "$K/scripts/train/lerobot_diffusion_pusht.sh" 2>&1)"; rc=$?
check "EVAL_ONLY without checkpoint fails" "$rc:$(grep -c 'no checkpoint found' <<<"$out")" "1:1"

# --- 5. hints, from the user's real failure (training finished, eval workers could not find the sim) ---
for s in act_aloha:gym_aloha diffusion_pusht:gym_pusht; do
  st="${s%%:*}"; sim="${s#*:}"
  h="$(failure_hint "$st" "$T/fixtures/$st.rc1.namespace.log" 1)"
  check "hint $st: names $sim + async bug" "$(grep -c "^$sim was not found inside LeRobot's eval worker processes (LeRobot async-env bug lerobot#4931" <<<"$h")" 1
  check "hint $st: EVAL_ONLY rerun"        "$(grep -c 'EVAL_ONLY=1 reruns just the eval' <<<"$h")" 1
  check "metric $st: loss kept"            "$(extract_metric lerobot_train "$T/fixtures/$st.rc1.namespace.log" | cut -c1-5)" "loss "
done
printf "gymnasium.error.NamespaceNotFound: Namespace gym_pusht not found. Have you installed the proper package for gym_pusht?\n" > "$TMP/ns.log"
check "hint: NamespaceNotFound in-process -> install" "$(failure_hint diffusion_pusht "$TMP/ns.log" 1)" \
  "the gym_pusht simulator isn't installed; run scripts/10_install_lerobot.sh (it adds the aloha/pusht extras)"
printf "ModuleNotFoundError: Package 'gym_aloha' required for env 'aloha' not found. Please install it or check PYTHONPATH.\n" > "$TMP/mnf.log"
check "hint: lerobot ModuleNotFoundError -> install" "$(failure_hint act_aloha "$TMP/mnf.log" 1 | cut -c1-39)" "the gym_aloha simulator isn't installed"
printf "[FAIL] gym_aloha eval env does not work headless (rc=4): sim probe: x\n" > "$TMP/render.log"
check "hint: headless render problem" "$(failure_hint act_aloha "$TMP/render.log" 1 | cut -c1-45)" "gym_aloha eval env does not work headless: se"

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
