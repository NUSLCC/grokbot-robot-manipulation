#!/usr/bin/env bash
# Unit tests: failure hints (native vs WSL2 wording, matched on log content / rc), apt source notes, interrupted
# dpkg detection, and that every train/eval step in run_wsl_headless.sh declares its install dependency.
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; K="$(cd "$T/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export PATH="$T/fake-bin:$PATH" RSP_HOME="$TMP/rsp"
# shellcheck source=../scripts/lib/common.sh
source "$K/scripts/lib/common.sh"
# shellcheck source=../scripts/lib/results.sh
source "$K/scripts/lib/results.sh"
set +e
nfail=0
check() { if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: expected '$3', got '$2'"; nfail=$((nfail+1)); fi; }
L="$TMP/step.log"
hint_for() { RSP_FORCE_WSL="$1" failure_hint "$2" "$3" "$4"; }   # WSL(0/1) STEP LOG RC

# --- apt / dpkg hints, from the real log on the user's native Ubuntu 22.04 workstation ---
B="$T/fixtures/base_deps.rc100.log"
check "dpkg interrupted (native)"   "$(hint_for 0 base_deps "$B" 100)" "apt is broken: run sudo dpkg --configure -a, then rerun"
check "dpkg interrupted (wsl)"      "$(hint_for 1 base_deps "$B" 100)" "apt is broken: run sudo dpkg --configure -a, then rerun"
printf 'Installing apt packages: vulkan-tools libegl1 libosmesa6\nE: Unable to locate package foo\n' > "$L"
check "rc=100 generic apt, package names are not errors" "$(hint_for 0 base_deps "$L" 100)" \
  "apt-get failed (rc=100): see the 'E:' lines in the log, fix apt, then rerun bash scripts/01_base_deps.sh"
echo "E: Could not get lock /var/lib/dpkg/lock-frontend. It is held by process 1234 (unattended-upgr)" > "$L"
check "apt lock"                    "$(hint_for 0 base_deps "$L" 100 | cut -c1-12)" "apt is busy "

# --- native Linux never gets WSL wording; WSL2 does ---
echo "Killed" > "$L"
check "RAM native: no WSL"          "$(hint_for 0 act_aloha "$L" 137 | grep -ci wsl)" 0
check "RAM wsl: .wslconfig"         "$(hint_for 1 act_aloha "$L" 137 | grep -c '.wslconfig')" 1
echo "torch.OutOfMemoryError: CUDA out of memory." > "$L"
check "OOM native: no Windows"      "$(hint_for 0 act_aloha "$L" 1 | grep -ciE 'wsl|windows')" 0
echo "RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed" > "$L"
check "Vulkan native: no WSL"       "$(hint_for 0 maniskill_ppo "$L" 1 | grep -ci wsl)" 0
check "Vulkan native: names vulkan" "$(hint_for 0 maniskill_ppo "$L" 1 | cut -c1-30)" "Vulkan rendering unavailable: "
check "Vulkan wsl: WSL2 note"       "$(hint_for 1 maniskill_ppo "$L" 1 | grep -c 'normal on WSL2')" 1
echo "smoke Isaac-Reach-Franka [default] FAIL  rc=1 after 15s" > "$L"
check "Isaac native: no WSL"        "$(hint_for 0 isaac_validate "$L" 1 | grep -ci wsl)" 0
check "Isaac wsl: WSL2 note"        "$(hint_for 1 isaac_validate "$L" 1 | grep -c 'WSL2 is not officially supported')" 1
echo "RuntimeError: CUDA error: no kernel image is available for execution on the device" > "$L"
check "Blackwell hint"              "$(hint_for 0 act_aloha "$L" 1 | cut -c1-48)" "torch in this env has no Blackwell (sm_120) kern"
for st in system_check base_deps isaac_validate act_aloha maniskill_ppo; do
  for rc in 1 2 100 124 137; do
    check "no WSL wording native: $st rc=$rc (generic log)" "$(echo 'error' > "$L"; hint_for 0 "$st" "$L" "$rc" | grep -ciE 'wsl|windows')" 0
  done
done

# --- Isaac Lab 3.0 Newton/Warp pins (detect-only check row) + the matching runner hint ---
echo "AttributeError: 'ModelBuilder' object has no attribute '_cloth_label'" > "$L"
for w in 0 1; do
  check "Newton hint (wsl=$w)"      "$(hint_for "$w" isaac_validate "$L" 1)" "Newton version mismatch with Isaac Lab; see the Newton/Warp pins row in the step log for the fix command"
done
# shellcheck source=../scripts/lib/isaaclab_env.sh
source "$K/scripts/lib/isaaclab_env.sh"; set +e
PY=/home/u/IsaacLab/.venv/bin/python
FIX="uv pip install --python $PY \"newton[sim]==1.6.1\" \"mujoco~=3.12.0\" \"mujoco-warp~=3.12.0\" \"warp-lang==1.17.0\" \"newton-usd-schemas>=0.5.0\""
r="$(newton_pins_check 3.0.0 "$PY" 1.8.0.dev0 1.17.0 3.12.1 3.12.0 0.5.2)"
check "pins: user's newton 1.8.0.dev0 -> WARN" "${r%%|*}" WARN
check "pins: names newton"          "$(grep -cF 'newton 1.8.0.dev0 (needs ==1.6.1)' <<<"$r")" 1
check "pins: exact fix command"     "$(grep -cF "$FIX" <<<"$r")" 1
check "pins: matching -> PASS"      "$(newton_pins_check 3.0.0 "$PY" 1.6.1 1.17.0 3.12.0 3.12.2 0.5.0 | cut -d'|' -f1)" PASS
check "pins: mujoco 3.13 -> WARN"   "$(newton_pins_check 3.0.0 "$PY" 1.6.1 1.17.0 3.13.0 3.12.0 0.5.0 | cut -d'|' -f1)" WARN
check "pins: missing pkg -> WARN"   "$(newton_pins_check 3.0.0 "$PY" 1.6.1 None 3.12.0 3.12.0 0.5.0 | grep -c 'warp-lang None')" 1
check "pins: other Isaac Lab -> INFO" "$(newton_pins_check 2.3.0 "$PY" 1.8.0 1.0 3.0 3.0 0.1 | cut -d'|' -f1)" INFO

# --- apt source notes (one line per failed source; NO_PUBKEY from the real log) ---
check "apt note: grok-bot repo"     "$(apt_repo_notes "$B")" "https://downloads.cursor.com/aptrepo grok-bot (NO_PUBKEY 42A1772E62E492D6)"
printf 'Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease\nReading package lists...\n' > "$L"
check "apt note: clean update"      "$(apt_repo_notes "$L")" ""
# apt_update_tolerant never fails, and prints the note (fake sudo -> fake apt-get update failing with that log)
mkdir -p "$TMP/bin"
printf '#!/usr/bin/env bash\nexec env "$@"\n' > "$TMP/bin/sudo"   # like sudo: accepts VAR=value before the command
printf '#!/usr/bin/env bash\ncat %q; exit 100\n' "$B" > "$TMP/bin/apt-get"
chmod +x "$TMP/bin/sudo" "$TMP/bin/apt-get"
out="$(PATH="$TMP/bin:$PATH" apt_update_tolerant 2>&1)"; rc=$?
check "apt_update_tolerant rc=0"    "$rc" 0
check "apt_update_tolerant note"    "$(grep -c 'note: apt source could not be refreshed and was ignored (not needed by this kit): https://downloads.cursor.com/aptrepo grok-bot (NO_PUBKEY 42A1772E62E492D6)' <<<"$out")" 1

# --- interrupted dpkg detection + auto repair ---
mkdir -p "$TMP/upd"
export RSP_DPKG_AUDIT_CMD=true
check "dpkg ok (empty updates dir)" "$(RSP_DPKG_UPDATES_DIR="$TMP/upd" dpkg_interrupted && echo y || echo n)" n
touch "$TMP/upd/0001"
check "dpkg interrupted (updates)"  "$(RSP_DPKG_UPDATES_DIR="$TMP/upd" dpkg_interrupted && echo y || echo n)" y
check "dpkg interrupted (audit)"    "$(RSP_DPKG_UPDATES_DIR="$TMP/none" RSP_DPKG_AUDIT_CMD="echo half-configured" dpkg_interrupted && echo y || echo n)" y
# fake dpkg: --configure -a clears the updates dir
printf '#!/usr/bin/env bash\n[ "$1 $2" = "--configure -a" ] && rm -f %q/* && echo "configured"\n' "$TMP/upd" > "$TMP/bin/dpkg"; chmod +x "$TMP/bin/dpkg"
out="$(PATH="$TMP/bin:$PATH" RSP_DPKG_UPDATES_DIR="$TMP/upd" dpkg_repair_if_interrupted 2>&1)"; rc=$?
check "dpkg repair rc=0"            "$rc" 0
check "dpkg repair logged"          "$(grep -c 'Running: sudo dpkg --configure -a' <<<"$out")" 1
check "dpkg repaired"               "$(RSP_DPKG_UPDATES_DIR="$TMP/upd" dpkg_interrupted && echo y || echo n)" n
touch "$TMP/upd/0001"
out="$(PATH="$TMP/bin:$PATH" RSP_DPKG_UPDATES_DIR="$TMP/upd" RSP_AUTO_DPKG_CONFIGURE=0 dpkg_repair_if_interrupted 2>&1)"; rc=$?
check "dpkg report-only fails"      "$rc" 1
check "dpkg report-only message"    "$(grep -c 'apt is broken' <<<"$out")" 1

# --- every train/eval step declares its install dependency ---
while read -r name kind dep; do
  case "$kind" in
    lerobot_train|lerobot_eval) want=lerobot_install ;;
    openpi_eval) want=openpi_install ;;
    maniskill_ppo) want=maniskill_install ;;
    isaac_rl) want=isaac_validate ;;
    *) continue ;;
  esac
  check "dependency declared: $name" "$dep" "$want"
done < <(grep -E '^add [a-z0-9_]+ ' "$K/scripts/run_wsl_headless.sh" | awk '{print $2, $3, $4}')
check "steps parsed"                "$(grep -cE '^add [a-z0-9_]+ +(lerobot_|openpi_eval|maniskill_ppo|isaac_rl)' "$K/scripts/run_wsl_headless.sh")" 6

echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
