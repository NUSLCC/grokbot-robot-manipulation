#!/usr/bin/env bash
# Shared helpers. Source this file; do not execute it.
# shellcheck disable=SC2034
set -euo pipefail

_RSP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSP_KIT_DIR="$(cd "${_RSP_LIB_DIR}/../.." && pwd)"

_c() { if [ -t 1 ]; then printf '\033[%sm' "$1"; fi; }
log()  { echo "$(_c 36)[rsp $(date +%H:%M:%S)]$(_c 0) $*"; }
ok()   { echo "$(_c 32)[ OK ]$(_c 0) $*"; }
warn() { echo "$(_c 33)[WARN]$(_c 0) $*" >&2; }
fail() { echo "$(_c 31)[FAIL]$(_c 0) $*" >&2; }
die()  { fail "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
# Exit code a script uses to say "skipped on purpose" (run_wsl_headless.sh reports it as SKIP).
RSP_EXIT_SKIP=77
skip_exit() { warn "SKIP: $*"; exit "$RSP_EXIT_SKIP"; }

# ver_ge A B  -> true if version A >= B
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | sed -n 1p)" = "$2" ]; }

# ---------------------------------------------------------------------------------------------
# Platform detection (WSL2, Ubuntu release). Test overrides:
#   RSP_FORCE_WSL=1|0           force WSL on/off (CI and dry runs)
#   RSP_PROC_VERSION_FILE=path  read this instead of /proc/version
#   RSP_OS_RELEASE_FILE=path    read this instead of /etc/os-release
# ---------------------------------------------------------------------------------------------
is_wsl() {
  case "${RSP_FORCE_WSL:-}" in 1|true|yes) return 0 ;; 0|false|no) return 1 ;; esac
  [ -n "${WSL_DISTRO_NAME:-}" ] && return 0
  grep -qi microsoft "${RSP_PROC_VERSION_FILE:-/proc/version}" 2>/dev/null
}
# os_field KEY -> value from os-release (e.g. os_field VERSION_ID -> 24.04)
os_field() {
  local f="${RSP_OS_RELEASE_FILE:-/etc/os-release}"
  [ -r "$f" ] || { echo ""; return 0; }
  sed -n "s/^$1=//p" "$f" | sed -n 1p | tr -d '"'
}
os_version_id() { os_field VERSION_ID; }

# ---------------------------------------------------------------------------------------------
# GPU detection + hardware profile (scripts/lib/profiles/*.env). Detected once, then exported.
#   PROFILE=dual3090|rtx4090|rtx5090_laptop_wsl|generic   override auto-selection
#   (a desktop RTX 5090 or any other GPU gets 'generic'; Blackwell is still detected from compute capability)
# ---------------------------------------------------------------------------------------------
detect_gpus() {
  [ "${RSP_GPU_DETECTED:-0}" = 1 ] && return 0
  RSP_GPU_COUNT=0; RSP_GPU_NAME=""; RSP_GPU_CC=""; RSP_GPU_VRAM_MIB=0
  if have nvidia-smi; then
    local lines
    lines="$(timeout 30 nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)"
    lines="$(printf '%s\n' "$lines" | grep -E '^[^,]+,[^,]+,' || true)"
    if [ -n "$lines" ]; then
      RSP_GPU_COUNT="$(printf '%s\n' "$lines" | grep -c .)"
      RSP_GPU_NAME="$(printf '%s\n' "$lines" | sed -n 1p | cut -d, -f1 | sed 's/^ *//;s/ *$//')"
      RSP_GPU_VRAM_MIB="$(printf '%s\n' "$lines" | awk -F, '{gsub(/[^0-9]/,"",$2); v=$2+0; if(NR==1||v<m)m=v}END{print m+0}')"
      RSP_GPU_CC="$(printf '%s\n' "$lines" | awk -F, '{gsub(/ /,"",$3); if($3 ~ /^[0-9]+\.[0-9]+$/) print $3}' | sort -V | tail -n1)"
    fi
  fi
  RSP_GPU_DETECTED=1
  export RSP_GPU_DETECTED RSP_GPU_COUNT RSP_GPU_NAME RSP_GPU_CC RSP_GPU_VRAM_MIB
}
# Major compute capability of the newest GPU (0 if unknown). 8=Ampere(3090: 8.6) / Ada(4090: 8.9), 12=Blackwell(5090 Laptop: 12.0)
gpu_cc_major() { detect_gpus; local m="${RSP_GPU_CC%%.*}"; case "$m" in ''|*[!0-9]*) echo 0 ;; *) echo "$m" ;; esac; }
# Blackwell (sm_100/sm_120) needs PyTorch built for CUDA >= 12.8 (torch >= 2.7, cu128 wheels).
is_blackwell() {
  [ "${RSP_GPU_ARCH:-}" = blackwell ] && return 0
  [ "$(gpu_cc_major)" -ge 10 ]
}

rsp_select_profile() {
  if [ -z "${PROFILE:-}" ]; then
    detect_gpus
    case "$RSP_GPU_NAME" in
      *5090*Laptop*|*5090*laptop*) PROFILE=rtx5090_laptop_wsl ;;
      *4090*) PROFILE=rtx4090 ;;
      *3090*) PROFILE=dual3090 ;;
      *)      PROFILE=generic ;;
    esac
    RSP_PROFILE_AUTO=1
  fi
  export PROFILE RSP_PROFILE_AUTO="${RSP_PROFILE_AUTO:-0}"
  if [ "$PROFILE" != generic ]; then
    local f="$_RSP_LIB_DIR/profiles/$PROFILE.env"
    if [ ! -f "$f" ]; then
      die "Unknown PROFILE=$PROFILE. Available: generic $(cd "$_RSP_LIB_DIR/profiles" && ls ./*.env | sed 's|^./||;s|\.env$||' | tr '\n' ' ')"
    fi
    # shellcheck disable=SC1090
    source "$f"
  fi
  # Generic fallbacks (any value already exported by you or the profile wins)
  if [ -z "${RSP_GPU_ARCH:-}" ]; then
    case "$(gpu_cc_major)" in 0) RSP_GPU_ARCH=unknown ;; 7) RSP_GPU_ARCH=turing_or_older ;; 8) RSP_GPU_ARCH=ampere_or_ada ;; 9) RSP_GPU_ARCH=hopper ;; *) RSP_GPU_ARCH=blackwell ;; esac
  fi
  : "${RSP_PROFILE_DESC:=generic (no profile matched GPU '${RSP_GPU_NAME:-none}')}"
  : "${RSP_VRAM_CLASS_GB:=24}"
  : "${RSP_EXPECT_GPUS:=1}"
  : "${TORCH_CUDA:=cu128}"
  : "${RSP_WSL_RENDER_FALLBACK:=1}"
  : "${RSP_EXPECT_WSL:=}"
  : "${RSP_PI0_LORA_NOTE:=see README VRAM table}"
  export RSP_GPU_ARCH RSP_PROFILE_DESC RSP_VRAM_CLASS_GB RSP_EXPECT_GPUS TORCH_CUDA RSP_WSL_RENDER_FALLBACK RSP_EXPECT_WSL RSP_PI0_LORA_NOTE
}
rsp_select_profile

# shellcheck source=versions.env
source "${_RSP_LIB_DIR}/versions.env"

# Everything this kit creates lives under RSP_HOME (sources, envs, data, logs).
: "${RSP_HOME:=$HOME/robot-sim}"
RSP_SRC="$RSP_HOME/src"
RSP_ENVS="$RSP_HOME/envs"
RSP_DATA="$RSP_HOME/data"
RSP_LOGS="$RSP_HOME/logs"
RSP_CKPT="$RSP_HOME/checkpoints"
mkdir -p "$RSP_SRC" "$RSP_ENVS" "$RSP_DATA" "$RSP_LOGS" "$RSP_CKPT"

export HF_HOME="${HF_HOME:-$RSP_HOME/hf_cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$RSP_HOME/openpi_cache}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$RSP_HOME/maniskill_data}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT="${MS_SKIP_ASSET_DOWNLOAD_PROMPT:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"   # avoid hardlinked uv cache edits (openpi transformers patch warning)

# Headless rendering defaults. MuJoCo (LIBERO/robosuite/ALOHA): EGL, probed per env by select_mujoco_gl
# (falls back to OSMesa when EGL does not work, e.g. on WSL2). SAPIEN (ManiSkill/SimplerEnv/RoboTwin): Vulkan.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"


# GPU selection: set CUDA_VISIBLE_DEVICES from GPUS (e.g., "0,1") or NUM_GPUS (first N GPUs)
apply_gpu_selection() {
  if [ -n "${GPUS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    # Infer NUM_GPUS if not provided
    if [ -z "${NUM_GPUS:-}" ]; then NUM_GPUS="$(awk -F, '{print NF}' <<<"$GPUS")"; export NUM_GPUS; fi
    log "Using GPUs (CUDA_VISIBLE_DEVICES): $CUDA_VISIBLE_DEVICES"
  elif [ -n "${NUM_GPUS:-}" ]; then
    # Select first NUM_GPUS devices
    if [ "$NUM_GPUS" -gt 0 ] 2>/dev/null; then
      local sel; sel=$(seq -s, 0 $((NUM_GPUS - 1)))
      export CUDA_VISIBLE_DEVICES="$sel"
      log "Using first $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"
    fi
  fi
}

start_log() {
  # start_log NAME -> tee all stdout/stderr to $RSP_LOGS/NAME_$(stamp).txt
  local name="${1:-run}"; mkdir -p "$RSP_LOGS"
  REPORT="$RSP_LOGS/${name}_$(stamp).txt"
  exec > >(tee "$REPORT") 2>&1
  log "Logging to $REPORT"
}

# check_disk_space MIN_GB [PATH]
check_disk_space() {
  local min_gb="$1" path="${2:-$RSP_HOME}"
  local free_gb; free_gb=$(df -BG --output=avail "$path" | tail -n1 | tr -dc 0-9)
  if [ "$free_gb" -lt "$min_gb" ]; then
    die "Low free disk at $path: ${free_gb} GB < required ${min_gb} GB. Free space or set RSP_HOME to a larger disk."
  fi
}

# hf_check_access TYPE(id) -> verifies existence and (when logged in) gated access
# TYPE: model|dataset
hf_check_access() {
  local type="$1" id="$2"
  if ! have python3; then warn "python3 missing; cannot check Hugging Face access for $id"; return 0; fi
  python3 - "$type" "$id" <<'PY' || { warn "Hugging Face check failed for $2"; return 0; }
import os, sys, json
from huggingface_hub import model_info, dataset_info, HfApi, HfHubHTTPError
typ, _id = sys.argv[1], sys.argv[2]
api = HfApi()
try:
    if typ == "model":
        info = model_info(_id, token=True)
        gated = getattr(info, "private", False) or getattr(info, "gated", False)
    else:
        info = dataset_info(_id, token=True)
        gated = getattr(info, "private", False) or getattr(info, "gated", False)
    print(json.dumps({"exists": True, "gated": bool(gated)}))
except HfHubHTTPError as e:
    if e.response is not None and e.response.status_code in (401, 403, 404):
        # Distinguish 404-not-found from gated/no-token
        not_found = e.response.status_code == 404
        print(json.dumps({"exists": not not_found, "gated": not not_found}))
        sys.exit(0)
    raise
PY
}

# Strip variables that leak one Python stack into another. Isaac Sim's setup_python_env.sh /
# setup_conda_env.sh export PYTHONPATH and LD_LIBRARY_PATH entries that break other envs.
clean_python_env() {
  if [ -n "${PYTHONPATH:-}" ]; then warn "Unsetting PYTHONPATH for isolation (was: ${PYTHONPATH})"; fi
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV || true
  if [ -n "${LD_LIBRARY_PATH:-}" ] && echo "$LD_LIBRARY_PATH" | grep -qiE 'isaac|omni|kit'; then
    warn "LD_LIBRARY_PATH contains Isaac/Omniverse paths; removing them for this process."
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -viE 'isaac|omni|kit' | paste -sd: - || true)"
    export LD_LIBRARY_PATH
  fi
}

ensure_uv() {
  if ! have uv; then
    [ -x "$HOME/.local/bin/uv" ] && export PATH="$HOME/.local/bin:$PATH"
  fi
  have uv || die "uv not found. Run scripts/01_base_deps.sh first."
}

# Locate a conda/mamba installation (Miniforge preferred).
conda_base() {
  local c
  for c in "${CONDA_BASE:-}" "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    [ -n "$c" ] && [ -x "$c/bin/conda" ] && { echo "$c"; return 0; }
  done
  if have conda; then conda info --base 2>/dev/null && return 0; fi
  return 1
}
ensure_conda() { conda_base >/dev/null || die "conda not found. Run scripts/01_base_deps.sh (installs Miniforge)."; }

# conda_env_create PREFIX PYVER  (idempotent; envs live under $RSP_ENVS/conda/)
conda_env_create() {
  local prefix="$1" py="$2" base; base="$(conda_base)"
  if [ -x "$prefix/bin/python" ]; then ok "conda env exists: $prefix"; return 0; fi
  log "Creating conda env $prefix (python=$py)"
  "$base/bin/conda" create -y -p "$prefix" "python=$py"
}
# conda_activate PREFIX  (for upstream scripts that call bare `pip`/`python`)
conda_activate() {
  local base; base="$(conda_base)"
  set +u
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  conda activate "$1"
  set -u
}

# git_sync URL DIR [REF]  clone (with submodules) or update to REF. Never discards local changes.
git_sync() {
  local url="$1" dir="$2" ref="${3:-}"
  if [ -d "$dir/.git" ]; then
    log "Repo exists: $dir (fetching)"
    git -C "$dir" fetch --tags --quiet origin || warn "fetch failed for $dir (offline?)"
    if [ -n "$ref" ]; then
      if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then
        warn "$dir has local changes; not switching to $ref"
      else
        git -C "$dir" checkout --quiet "$ref"
        if git -C "$dir" symbolic-ref -q HEAD >/dev/null; then git -C "$dir" pull --ff-only --quiet || true; fi
      fi
    fi
  else
    log "Cloning $url -> $dir"
    git clone --recurse-submodules "$url" "$dir"
    [ -n "$ref" ] && git -C "$dir" checkout --quiet "$ref"
  fi
  git -C "$dir" submodule update --init --recursive
  echo "$(date -Iseconds) $url $(git -C "$dir" rev-parse HEAD) ref=${ref:-default}" >> "$RSP_LOGS/commits.txt"
  ok "$(basename "$dir") @ $(git -C "$dir" rev-parse --short HEAD)"
}

# Wait for TCP port on localhost. wait_port PORT TIMEOUT_S
wait_port() {
  local port="$1" t="${2:-600}" i=0
  while ! (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; do
    sleep 2; i=$((i+2)); [ "$i" -ge "$t" ] && return 1
  done
  exec 3>&- 3<&- 2>/dev/null || true
  return 0
}

# Background GPU memory sampler. gpu_monitor_start FILE ; gpu_monitor_stop -> prints peak MiB
gpu_monitor_start() {
  _RSP_GPU_LOG="$1"; _RSP_GPU_PID=""
  have nvidia-smi || return 0
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "$_RSP_GPU_LOG" 2>/dev/null &
  _RSP_GPU_PID=$!
}
gpu_monitor_stop() {
  if [ -n "${_RSP_GPU_PID:-}" ]; then
    kill "$_RSP_GPU_PID" 2>/dev/null || true
    wait "$_RSP_GPU_PID" 2>/dev/null || true
  fi
  if [ -s "${_RSP_GPU_LOG:-/nonexistent}" ]; then
    awk -F, 'BEGIN{m=0}{gsub(/ /,"",$1); if($1+0>m)m=$1+0}END{print m}' "$_RSP_GPU_LOG"
  else echo "n/a"; fi
}

stamp() { date +%Y%m%d-%H%M%S; }

# ---------------------------------------------------------------------------------------------
# Headless rendering helpers
# ---------------------------------------------------------------------------------------------
_set_gl_platform() { export MUJOCO_GL="$1"; export PYOPENGL_PLATFORM="$1"; }

# select_mujoco_gl PYTHON
# Probes MuJoCo EGL in the given interpreter. If EGL does not work (common on WSL2, where there is no
# NVIDIA EGL), falls back to OSMesa (CPU rendering: works everywhere, noticeably slower).
# Respects an explicit MUJOCO_GL=osmesa|glx|glfw. RSP_SKIP_GL_PROBE=1 keeps the current value.
# RSP_WSL_RENDER_FALLBACK=0 disables the fallback (then an EGL failure surfaces in the step itself).
select_mujoco_gl() {
  local py="$1" rc=0
  case "${MUJOCO_GL:-egl}" in
    egl) ;;
    *) _set_gl_platform "$MUJOCO_GL"; log "MUJOCO_GL=$MUJOCO_GL (set by you; not probing)"; return 0 ;;
  esac
  if [ "${RSP_SKIP_GL_PROBE:-0}" = 1 ]; then _set_gl_platform egl; return 0; fi
  if [ ! -x "$py" ]; then warn "select_mujoco_gl: $py not found; keeping MUJOCO_GL=egl"; _set_gl_platform egl; return 0; fi
  local out
  out="$(MUJOCO_GL=egl PYOPENGL_PLATFORM=egl timeout 120 "$py" - 2>/dev/null <<'PY'
import sys
try:
    import mujoco
except Exception:
    sys.exit(3)          # no mujoco in this env -> cannot probe
ctx = mujoco.GLContext(64, 64)
ctx.make_current()
try:
    from OpenGL import GL
    print("GL_RENDERER=" + GL.glGetString(GL.GL_RENDERER).decode(errors="replace"))
except Exception:
    print("GL_RENDERER=unknown")
ctx.free()
PY
)" || rc=$?
  if [ "$rc" = 0 ]; then
    _set_gl_platform egl
    local rend; rend="$(sed -n 's/^GL_RENDERER=//p' <<<"$out" | tail -n1)"
    ok "MuJoCo EGL rendering works (MUJOCO_GL=egl, renderer: ${rend:-unknown})"
    case "$rend" in *llvmpipe*|*softpipe*|*swrast*) warn "EGL renderer is software ($rend): rendering runs on the CPU (slower, but works)";; esac
    return 0
  fi
  if [ "$rc" = 3 ]; then _set_gl_platform egl; warn "mujoco not importable in $py; keeping MUJOCO_GL=egl"; return 0; fi
  if [ "${RSP_WSL_RENDER_FALLBACK:-1}" = 0 ]; then
    _set_gl_platform egl; warn "MuJoCo EGL probe failed (rc=$rc) and RSP_WSL_RENDER_FALLBACK=0: keeping egl"; return 0
  fi
  if ! ldconfig -p 2>/dev/null | grep -q 'libOSMesa.so'; then
    warn "MuJoCo EGL probe failed and libOSMesa is missing: run scripts/01_base_deps.sh (installs libosmesa6). Keeping egl."
    _set_gl_platform egl; return 0
  fi
  _set_gl_platform osmesa
  warn "MuJoCo EGL not available$(is_wsl && echo ' (expected on WSL2: no NVIDIA EGL there)'). Using MUJOCO_GL=osmesa (CPU rendering; evals run several times slower)."
}

# vulkan_status -> nvidia | dzn | cpu-only | none | unknown   (what `vulkaninfo --summary` reports)
vulkan_status() {
  have vulkaninfo || { echo unknown; return 0; }
  local s; s="$(timeout 60 vulkaninfo --summary 2>/dev/null || true)"
  if [ -z "$s" ] || ! grep -q 'deviceName' <<<"$s"; then echo none
  elif grep -qiE 'deviceName *= *.*NVIDIA' <<<"$s" && ! grep -qiE 'Microsoft|D3D12|dozen' <<<"$s"; then echo nvidia
  elif grep -qiE 'dozen|D3D12|Microsoft Direct3D' <<<"$s"; then echo dzn
  elif grep -qiE 'llvmpipe|lavapipe|PHYSICAL_DEVICE_TYPE_CPU' <<<"$s"; then echo cpu-only
  else echo unknown; fi
}

# ---------------------------------------------------------------------------------------------
# Blackwell (RTX 5090, sm_120) helpers
# ---------------------------------------------------------------------------------------------
# warn_torch_arch FRAMEWORK TORCH_VERSION  - loud warning when a framework pins a torch with no sm_120 kernels.
warn_torch_arch() {
  is_blackwell || return 0
  if ver_ge "$2" 2.7; then
    ok "$1 pins torch $2 -> needs the cu128 (or newer) wheel for Blackwell; using TORCH_CUDA=$TORCH_CUDA"
  else
    warn "============================================================================================"
    warn "$1 pins torch $2, which has NO Blackwell (sm_120) kernels. On an RTX 5090 its GPU code is"
    warn "expected to fail with 'no kernel image is available for execution on the device'."
    warn "The install continues (CPU-side tools still work). See docs/WSL2.md 'Blackwell (RTX 5090)'."
    warn "============================================================================================"
  fi
}

# find_cuda_home MIN_VERSION  -> prints a CUDA toolkit root whose nvcc >= MIN_VERSION (prefers CUDA_HOME, nvcc on PATH)
find_cuda_home() {
  local min="$1" c v
  for c in "${CUDA_HOME:-}" "$( { command -v nvcc >/dev/null 2>&1 && dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")"; } || true)" \
           /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
    [ -n "$c" ] && [ -x "$c/bin/nvcc" ] || continue
    v="$("$c/bin/nvcc" --version 2>/dev/null | grep -oE 'release [0-9.]+' | awk '{print $2}')"
    case "$v" in 12.*) ver_ge "$v" "$min" && { echo "$c"; return 0; } ;; esac
  done
  return 1
}

# ---------------------------------------------------------------------------------------------
# apt / dpkg health. Test overrides:
#   RSP_DPKG_UPDATES_DIR=path   read this instead of /var/lib/dpkg/updates
#   RSP_DPKG_AUDIT_CMD="cmd"    run this instead of 'dpkg --audit'
# ---------------------------------------------------------------------------------------------
# dpkg_interrupted -> true if a previous dpkg/apt run was interrupted (apt then refuses every install with
# "E: dpkg was interrupted, you must manually run 'sudo dpkg --configure -a'").
dpkg_interrupted() {
  local d="${RSP_DPKG_UPDATES_DIR:-/var/lib/dpkg/updates}" a
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then return 0; fi
  a="$(${RSP_DPKG_AUDIT_CMD:-dpkg --audit} 2>/dev/null || true)"
  [ -n "$a" ]
}
# apt_repo_notes FILE -> one line per apt source that failed to refresh, e.g.
#   https://downloads.cursor.com/aptrepo grok-bot (NO_PUBKEY 42A1772E62E492D6)
apt_repo_notes() {
  [ -s "$1" ] || return 0
  awk '
    /^(W|E): / {
      repo=""; why=""
      if (match($0, /GPG error: [^ ]+ [^ ]+/)) { split(substr($0, RSTART+11, RLENGTH-11), a, " "); repo=a[1] " " a[2] }
      else if (match($0, /Failed to fetch [^ ]+/)) {
        u=substr($0, RSTART+16, RLENGTH-16)
        if (u ~ /\/dists\//) { b=u; sub(/\/dists\/.*/, "", b); s=u; sub(/.*\/dists\//, "", s); sub(/\/.*/, "", s); repo=b " " s }
        else repo=u
      }
      else if (match($0, /[Tt]he repository .[^ ]+ [^ ]+/)) { split(substr($0, RSTART+15, RLENGTH-15), a, " "); repo=a[1] " " a[2] }
      if (repo == "") next
      if (match($0, /NO_PUBKEY [0-9A-Fa-f]+/)) why=substr($0, RSTART, RLENGTH)
      else if ($0 ~ /is not signed/) why="not signed"
      else if ($0 ~ /does not have a Release file/) why="no Release file"
      else if ($0 ~ /EXPKEYSIG/) why="expired key"
      else if ($0 ~ /Could not resolve|Temporary failure resolving/) why="network/DNS"
      else if (match($0, /[0-9][0-9][0-9] +[A-Za-z][A-Za-z ]*/)) why=substr($0, RSTART, RLENGTH)
      if (!(repo in seen)) { seen[repo]=why; order[++n]=repo } else if (seen[repo]=="" && why!="") seen[repo]=why
    }
    END { for (i=1;i<=n;i++) { r=order[i]; print r (seen[r]!="" ? " (" seen[r] ")" : "") } }' "$1"
}
# apt_update_tolerant -> 'sudo apt-get update' that never fails the caller. Sources that cannot be refreshed
# (third-party repos with a missing key, offline mirrors) are reported as one note each and otherwise ignored:
# apt keeps their old index files and the kit's packages all come from Ubuntu's own archive.
apt_update_tolerant() {
  local f rc=0 line
  f="$(mktemp)"
  sudo apt-get update 2>&1 | tee "$f" || rc=$?
  while IFS= read -r line; do
    [ -n "$line" ] && warn "note: apt source could not be refreshed and was ignored (not needed by this kit): $line"
  done < <(apt_repo_notes "$f")
  [ "$rc" = 0 ] || warn "apt-get update exited $rc; continuing with the package lists apt already has"
  rm -f "$f"
  return 0
}
# dpkg_repair_if_interrupted -> finish an interrupted dpkg run (sudo dpkg --configure -a) before installing.
# RSP_AUTO_DPKG_CONFIGURE=0 only reports it and fails instead.
dpkg_repair_if_interrupted() {
  dpkg_interrupted || return 0
  if [ "${RSP_AUTO_DPKG_CONFIGURE:-1}" != 1 ]; then
    fail "apt is broken: an earlier dpkg/apt run was interrupted. Run: sudo dpkg --configure -a   then rerun this script."
    return 1
  fi
  warn "An earlier dpkg/apt run was interrupted (apt would refuse to install anything)."
  log "Running: sudo dpkg --configure -a   (finishes the interrupted install; RSP_AUTO_DPKG_CONFIGURE=0 to only report)"
  if sudo DEBIAN_FRONTEND=noninteractive dpkg --configure -a; then ok "dpkg --configure -a finished; apt is usable again"
  else
    fail "apt is broken: 'sudo dpkg --configure -a' failed (see its error above). Fix that package, then rerun this script."
    return 1
  fi
}

# ---------------------------------------------------------------------------------------------
# LIBERO first-import prompt. LIBERO's libero/libero/__init__.py calls input() ("custom dataset path? (Y/N)") when
# $LIBERO_CONFIG_PATH/config.yaml (default ~/.libero/config.yaml) is missing; headless, that is
# "EOFError: EOF when reading a line". libero_config writes the config LIBERO would write for "N" (its defaults)
# without importing libero.libero, and exports LIBERO_CONFIG_PATH. Each env gets its own config dir because
# each env has its own LIBERO copy (hf-libero wheel in LeRobot, openpi's third_party/libero, OFT's clone, ...);
# a shared ~/.libero would point one env at another env's files.
#   libero_config PY NAME [EXTRA_PYTHONPATH]     config dir: $LIBERO_CONFIG_PATH if set, else $RSP_HOME/config/libero/NAME
# Never fatal: on failure it warns (the step then fails with the EOFError hint).
# ---------------------------------------------------------------------------------------------
libero_config() {
  local py="$1" name="$2" pp="${3:-}" dir
  dir="${LIBERO_CONFIG_PATH:-$RSP_HOME/config/libero/$name}"
  if PYTHONPATH="$pp" "$py" "$_RSP_LIB_DIR/libero_config.py" "$dir"; then
    export LIBERO_CONFIG_PATH="$dir"
  else
    warn "could not write the LIBERO config for $name ($py); LIBERO may stop at its first-import question"
  fi
  return 0
}

# ---------------------------------------------------------------------------------------------
# LeRobot CLI compatibility. lerobot-train flag names changed across releases:
#   <= v0.5.x  --eval_freq          v0.6.0+ --env_eval_freq     (renamed in TrainPipelineConfig)
#   <= v0.6.x  --wandb.enable       main after v0.6.1: --trackio.enable  (wandb replaced by Trackio)
# lerobot_train_compat BIN reads `BIN --help` once and sets:
#   LR_EVAL_FREQ_FLAG   the eval-frequency flag (without '=')
#   LR_TRACKER_OFF      array: the flag that turns experiment tracking off (empty if the CLI has neither)
# Test hook: LEROBOT_TRAIN_HELP_FILE=path uses that file instead of running BIN --help.
# ---------------------------------------------------------------------------------------------
_lr_help_has() { grep -qE -- "--$1([^a-zA-Z0-9_.]|\$)" <<<"$2"; }
lerobot_train_compat() {
  local bin="$1" help=""
  if [ -n "${LEROBOT_TRAIN_HELP_FILE:-}" ]; then help="$(cat "$LEROBOT_TRAIN_HELP_FILE")"
  else help="$(timeout 300 "$bin" --help 2>&1 || true)"; fi
  if _lr_help_has env_eval_freq "$help"; then LR_EVAL_FREQ_FLAG=--env_eval_freq
  elif _lr_help_has eval_freq "$help"; then LR_EVAL_FREQ_FLAG=--eval_freq
  else
    LR_EVAL_FREQ_FLAG=--env_eval_freq
    warn "lerobot-train --help lists neither --env_eval_freq nor --eval_freq; using --env_eval_freq (LEROBOT_REF=$LEROBOT_REF)"
  fi
  if _lr_help_has wandb.enable "$help"; then LR_TRACKER_OFF=(--wandb.enable=false)
  elif _lr_help_has trackio.enable "$help"; then LR_TRACKER_OFF=(--trackio.enable=false)
  else LR_TRACKER_OFF=(); fi
  log "lerobot-train flags: eval every N steps = $LR_EVAL_FREQ_FLAG, tracking off = ${LR_TRACKER_OFF[*]:-(none)}"
}
