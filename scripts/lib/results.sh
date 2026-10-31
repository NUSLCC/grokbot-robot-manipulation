#!/usr/bin/env bash
# Result helpers for run_wsl_headless.sh: key metric extraction + failure hints from step logs.
# Source this file. Pure bash + grep/sed/awk (no Python packages needed).

# _last LOG ERE  -> last match of ERE in LOG (empty if none)
_last() { grep -aoE "$2" "$1" 2>/dev/null | tail -n1 || true; }
_num()  { grep -aoE '[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' <<<"$1" | tail -n1 || true; }

# extract_metric KIND LOG -> one short string ("n/a" if nothing found)
#   KIND: syscheck | isaac_validate | isaac_rl | lerobot_train | lerobot_eval | openpi_eval | maniskill_ppo | install
extract_metric() {
  local kind="$1" log="$2" m="" a b
  [ -s "$log" ] || { echo "n/a"; return 0; }
  case "$kind" in
    syscheck)
      a="$(_last "$log" 'FAIL: [0-9]+ +WARN: [0-9]+')"; [ -n "$a" ] && m="$(sed 's/  */ /g' <<<"$a")" ;;
    isaac_validate)
      a="$(grep -acE '^(smoke|list_envs|Python|PyTorch|CUDA from|Isaac|isaaclab|robomimic|rsl_rl|Driver)[^|]* PASS ' "$log" || true)"
      b="$(grep -acE '^(smoke|list_envs|Python|PyTorch|CUDA from|Isaac|isaaclab|robomimic|rsl_rl|Driver)[^|]* FAIL ' "$log" || true)"
      if [ "${a:-0}" != 0 ] || [ "${b:-0}" != 0 ]; then m="checks PASS=${a:-0} FAIL=${b:-0}"; fi ;;
    isaac_rl)
      a="$(_num "$(_last "$log" 'Mean reward:? *[-+]?[0-9.eE+-]+')")"
      [ -n "$a" ] && m="mean reward $a"
      b="$(_num "$(_last "$log" 'Learning iteration [0-9]+')")"; [ -n "$b" ] && m="${m:+$m, }iter $b" ;;
    lerobot_train)
      a="$(_num "$(_last "$log" '(^| )loss:[0-9.eE+-]+')")"; [ -n "$a" ] && m="loss $a"
      b="$(_num "$(_last "$log" "(pc_success['\"]?: *|success:)[0-9.]+")")"; [ -n "$b" ] && m="${m:+$m, }eval success ${b}%" ;;
    lerobot_eval)
      a="$(_last "$log" 'Results in [^ ]+')"; a="${a#Results in }"
      if [ -n "$a" ] && [ -f "$a/eval_info.json" ]; then
        b="$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));o=d.get("overall",d);print(o.get("pc_success",""))' "$a/eval_info.json" 2>/dev/null || true)"
      fi
      [ -z "${b:-}" ] && b="$(_num "$(_last "$log" "pc_success['\"]?: *[0-9.]+")")"
      [ -n "${b:-}" ] && m="success ${b}%" ;;
    openpi_eval)
      a="$(_num "$(_last "$log" 'Total success rate: *[0-9.]+')")"
      if [ -n "$a" ]; then m="success $(awk -v x="$a" 'BEGIN{printf "%.1f", x*100}')%"
        b="$(_num "$(_last "$log" 'Total episodes: *[0-9]+')")"; [ -n "$b" ] && m="$m of $b eps"; fi ;;
    maniskill_ppo)
      a="$(_num "$(_last "$log" 'eval_success_once_mean=(tensor\()?[0-9.]+')")"; [ -n "$a" ] && m="eval success_once $a"
      b="$(_num "$(_last "$log" 'eval_return_mean=(tensor\()?[-0-9.]+')")"; [ -n "$b" ] && m="${m:+$m, }return $b"
      grep -aq "rendering disabled" "$log" && m="${m:-ran} (no-render fallback)" ;;
    install)
      grep -aqE '\[ OK \].*(ready|env:|installed)' "$log" && m="installed" ;;
  esac
  echo "${m:-n/a}"
}

# _hint_body LOG -> the log minus lines that only LIST package / file names (apt package lists, apt fetch lines),
# so a package called "vulkan-tools" or "libegl1" in an install line never looks like a Vulkan / EGL error.
_hint_body() {
  grep -avE 'Installing apt packages|apt-get install|^(Get|Hit|Ign):[0-9]+ |^(Reading|Building|Selecting|Preparing|Unpacking|Setting up) |^The following (NEW )?packages' "$1" 2>/dev/null || true
}
# _hint_wsl -> 0 on WSL2. Kept separate so every WSL-only wording sits behind one check.
_hint_wsl() { declare -F is_wsl >/dev/null && is_wsl; }
# _hint_has ERE -> true if the filtered log body (\$body in failure_hint) matches ERE (case-insensitive)
_hint_has() { grep -aqiE "$1" <<<"$body"; }

# failure_hint STEP LOG RC -> one-line hint. Matches on the log content and the exit code; WSL2-specific advice is
# only given when the run is on WSL2 (native Linux never sees a WSL hint).
failure_hint() {
  local step="$1" log="$2" rc="$3" body wsl=0 err
  _hint_wsl && wsl=1
  if [ "$step" = system_check ]; then
    local first; first="$(grep -aE '\[FAIL\]' "$log" 2>/dev/null | sed -n 1p | sed 's/.*\[FAIL\][^ ]* *//' | cut -c1-110)"
    echo "fix the [FAIL] lines in the log${first:+ (first: $first)}"
    return 0
  fi
  body="$(_hint_body "$log")"
  if _hint_has 'dpkg was interrupted'; then
    echo "apt is broken: run sudo dpkg --configure -a, then rerun"
  elif _hint_has 'Sub-process /usr/bin/dpkg returned an error code'; then
    echo "a package failed to install/configure: run sudo dpkg --configure -a, read its error, then rerun"
  elif _hint_has 'Could not get lock|Unable to acquire the dpkg frontend lock|dpkg frontend lock'; then
    echo "apt is busy (another apt/dpkg or unattended-upgrades is running): wait for it to finish, then rerun"
  elif _hint_has 'Unmet dependencies|held broken packages|apt --fix-broken install'; then
    echo "apt has broken dependencies: run sudo apt --fix-broken install, then rerun"
  elif [ "$rc" = 100 ] && [ "$step" = base_deps ]; then
    echo "apt-get failed (rc=100): see the 'E:' lines in the log, fix apt, then rerun bash scripts/01_base_deps.sh"
  elif _hint_has 'no kernel image is available|sm_120 is not compatible|CUDA capability sm_120'; then
    echo "torch in this env has no Blackwell (sm_120) kernels: needs a cu128 build (docs/WSL2.md section 5 'Blackwell' applies to any OS)"
  elif _hint_has 'CUDA out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED'; then
    if [ "$wsl" = 1 ]; then echo "GPU out of memory: lower BATCH / NUM_ENVS, close other GPU apps (on Windows too)"
    else echo "GPU out of memory: lower BATCH / NUM_ENVS, or free the GPU (nvidia-smi shows other processes); GPUS=1 picks the other card"; fi
  elif [ "$rc" = 137 ] || _hint_has '(^|[^a-z])Killed$|MemoryError|Cannot allocate memory'; then
    if [ "$wsl" = 1 ]; then echo "killed for RAM: raise memory/swap in %UserProfile%\\.wslconfig (docs/WSL2.md), then 'wsl --shutdown'"
    else echo "killed for RAM (OOM killer): close other programs or add swap (check: free -h; sudo dmesg | grep -i oom)"; fi
  elif _hint_has 'No space left on device'; then
    if [ "$wsl" = 1 ]; then echo "disk full: free space or set RSP_HOME to a bigger Linux-side disk"
    else echo "disk full: free space (df -h) or set RSP_HOME to a bigger disk"; fi
  elif _hint_has '401 Client Error|403 Client Error|gated repo|GatedRepoError|Access to model .* is restricted'; then
    echo "gated Hugging Face model: accept its license on the Hub, then: ~/robot-sim/envs/lerobot/bin/hf auth login"
  elif _hint_has 'vk::|vkCreateInstance|ErrorExtensionNotPresent|ErrorIncompatibleDriver|ErrorInitializationFailed|failed to find a rendering device|Vulkan (error|is not|not (available|supported|found))|(no|cannot find a) (compatible )?Vulkan'; then
    if [ "$wsl" = 1 ]; then echo "Vulkan rendering unavailable (normal on WSL2): see TROUBLESHOOTING.md 'WSL2: Vulkan'"
    else echo "Vulkan rendering unavailable: check 'vulkaninfo --summary' and that /usr/share/vulkan/icd.d/nvidia_icd.json exists (TROUBLESHOOTING.md 'SAPIEN/Vulkan errors')"; fi
  elif _hint_has 'eglInitialize|eglGetDisplay|EGL_NOT_INITIALIZED|EGLError|Cannot initialize a EGL|GLContext|OSMesa(CreateContext| not| error)|libOSMesa\.so.*(cannot|No such)'; then
    echo "OpenGL/EGL problem: try MUJOCO_GL=osmesa (sudo apt install libosmesa6) - TROUBLESHOOTING.md"
  elif _hint_has 'Isaac Lab not (found|detected)'; then
    echo "Isaac Lab not found: export ISAACLAB_PATH=/path/to/IsaacLab (or SKIP_ISAAC=1)"
  elif _hint_has 'Run scripts/1[0-9]_install'; then
    echo "framework not installed: run its scripts/1x_install_*.sh step first"
  elif [ "$rc" = 124 ]; then
    echo "timed out (STEP_TIMEOUT); first runs download models/datasets and compile shaders - rerun or raise the timeout"
  else
    case "$step" in
      base_deps) echo "apt/sudo or network problem; rerun: bash scripts/01_base_deps.sh" ;;
      isaac_*)
        # 02_isaaclab_validate.sh prints 'key error: ...' for a failed smoke run (taken from the smoke log)
        err="$(grep -aE '^ *key error: ' "$log" 2>/dev/null | tail -n1 | sed 's/^ *key error: //' | cut -c1-140)"
        if [ "$wsl" = 1 ]; then echo "${err:+Isaac error: $err; }see the smoke log tail in the step log and docs/ISAAC_LAB.md (WSL2 is not officially supported by NVIDIA)"
        else echo "${err:+Isaac error: $err; }see the smoke log tail in the step log and docs/ISAAC_LAB.md"; fi ;;
      *) echo "see the last 40 lines of the log" ;;
    esac
  fi
}
