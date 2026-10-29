#!/usr/bin/env bash
# Shared helpers. Source this file; do not execute it.
# shellcheck disable=SC2034
set -euo pipefail

_RSP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSP_KIT_DIR="$(cd "${_RSP_LIB_DIR}/../.." && pwd)"
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

# Headless rendering defaults. MuJoCo (LIBERO/robosuite/ALOHA/PushT): EGL. SAPIEN (ManiSkill/SimplerEnv/RoboTwin): Vulkan.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

_c() { if [ -t 1 ]; then printf '\033[%sm' "$1"; fi; }
log()  { echo "$(_c 36)[rsp $(date +%H:%M:%S)]$(_c 0) $*"; }
ok()   { echo "$(_c 32)[ OK ]$(_c 0) $*"; }
warn() { echo "$(_c 33)[WARN]$(_c 0) $*" >&2; }
fail() { echo "$(_c 31)[FAIL]$(_c 0) $*" >&2; }
die()  { fail "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ver_ge A B  -> true if version A >= B
ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

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
  _RSP_GPU_LOG="$1"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "$_RSP_GPU_LOG" 2>/dev/null &
  _RSP_GPU_PID=$!
}
gpu_monitor_stop() {
  [ -n "${_RSP_GPU_PID:-}" ] && kill "$_RSP_GPU_PID" 2>/dev/null || true
  wait "${_RSP_GPU_PID:-}" 2>/dev/null || true
  if [ -s "${_RSP_GPU_LOG:-/nonexistent}" ]; then
    awk -F, 'BEGIN{m=0}{gsub(/ /,"",$1); if($1+0>m)m=$1+0}END{print m}' "$_RSP_GPU_LOG"
  else echo "n/a"; fi
}

stamp() { date +%Y%m%d-%H%M%S; }
