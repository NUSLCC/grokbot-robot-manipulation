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

# failure_hint STEP LOG RC -> one-line hint
failure_hint() {
  local step="$1" log="$2" rc="$3"
  if [ "$step" = system_check ]; then
    local first; first="$(grep -aE '\[FAIL\]' "$log" 2>/dev/null | sed -n 1p | sed 's/.*\[FAIL\][^ ]* *//' | cut -c1-110)"
    echo "fix the [FAIL] lines in the log${first:+ (first: $first)}"
    return 0
  fi
  if grep -aqiE 'no kernel image is available|sm_120 is not compatible|CUDA capability sm_120' "$log" 2>/dev/null; then
    echo "torch in this env has no Blackwell (sm_120) kernels: needs a cu128 build (see docs/WSL2.md 'Blackwell')"
  elif grep -aqiE 'CUDA out of memory|OutOfMemoryError|RESOURCE_EXHAUSTED' "$log" 2>/dev/null; then
    echo "GPU out of memory: lower BATCH / NUM_ENVS, close other GPU apps (on Windows too)"
  elif [ "$rc" = 137 ] || grep -aqE '(^|[^a-z])Killed$|MemoryError|Cannot allocate memory' "$log" 2>/dev/null; then
    echo "killed for RAM: raise memory/swap in %UserProfile%\\.wslconfig (docs/WSL2.md), then 'wsl --shutdown'"
  elif grep -aqiE 'No space left on device' "$log" 2>/dev/null; then
    echo "disk full: free space or set RSP_HOME to a bigger Linux-side disk"
  elif grep -aqiE '401 Client Error|403 Client Error|gated repo|GatedRepoError|Access to model .* is restricted' "$log" 2>/dev/null; then
    echo "gated Hugging Face model: accept its license on the Hub, then: ~/robot-sim/envs/lerobot/bin/hf auth login"
  elif grep -aqiE 'vk::|Vulkan|ErrorIncompatibleDriver|ErrorInitializationFailed|failed to find a rendering device' "$log" 2>/dev/null; then
    echo "Vulkan rendering unavailable (normal on WSL2): see TROUBLESHOOTING.md 'WSL2: Vulkan'"
  elif grep -aqiE 'EGL|eglInitialize|GLContext|OSMesa' "$log" 2>/dev/null; then
    echo "OpenGL/EGL problem: try MUJOCO_GL=osmesa (sudo apt install libosmesa6) - TROUBLESHOOTING.md"
  elif grep -aqiE 'Isaac Lab not (found|detected)' "$log" 2>/dev/null; then
    echo "Isaac Lab not found: export ISAACLAB_PATH=/path/to/IsaacLab (or SKIP_ISAAC=1)"
  elif grep -aqiE 'Run scripts/1[0-9]_install' "$log" 2>/dev/null; then
    echo "framework not installed: run its scripts/1x_install_*.sh step first"
  elif [ "$rc" = 124 ]; then
    echo "timed out (STEP_TIMEOUT); first runs download models/datasets and compile shaders - rerun or raise the timeout"
  else
    case "$step" in
      system_check) echo "fix the [FAIL] lines at the top of the log" ;;
      base_deps)    echo "apt/sudo or network problem; rerun: bash scripts/01_base_deps.sh" ;;
      isaac_*)      echo "see docs/ISAAC_LAB.md; WSL2 is not officially supported by NVIDIA" ;;
      *)            echo "see the last 40 lines of the log" ;;
    esac
  fi
}
