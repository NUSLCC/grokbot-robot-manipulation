#!/usr/bin/env bash
# 01_base_deps.sh - system packages + uv + Miniforge. Idempotent.
# Does NOT touch the NVIDIA driver, Isaac Sim, or Isaac Lab.
#
# Options (env vars):
#   SKIP_APT=1               skip apt packages (no sudo available)
#   INSTALL_MINIFORGE=0      do not install Miniforge even if no conda is found
#   UV_VERSION=x.y.z         pin uv (default: latest; 0.13.6 was latest on 2026-10-29)
#   INSTALL_CUDA_TOOLKIT=12-8  also install NVIDIA's cuda-toolkit-<ver> (toolkit only, never the driver).
#                            Repo is picked automatically: wsl-ubuntu on WSL2, ubuntu2204 / ubuntu2404 on native.
#                            Blackwell (RTX 5090) builds need 12-8 or newer.
# Ubuntu 22.04 and 24.04 are both handled: package names are chosen per release and anything apt does not know
# is dropped with a warning instead of failing the install. System Python is never pip-installed into (PEP 668);
# every framework gets its own Python through uv or conda.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
UBU="$(os_version_id)"

APT_PKGS=(
  build-essential cmake ninja-build pkg-config git git-lfs curl wget unzip ca-certificates
  python3-dev tmux htop
  # video (LeRobot/torchcodec, GR00T): Ubuntu 22.04 ffmpeg is 4.4 (torchcodec supports FFmpeg 4-7)
  ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
  # headless GL for MuJoCo/robosuite (EGL primary, OSMesa fallback) + GR00T sim prerequisites
  libegl1 libegl1-mesa-dev libgl1-mesa-dev libglu1-mesa libgles2 libglew-dev libglfw3 libosmesa6 libosmesa6-dev patchelf
  # Vulkan loader + tools for SAPIEN (ManiSkill3, SimplerEnv, RoboTwin 2.0). NVIDIA ICD comes with the driver.
  libvulkan1 vulkan-tools mesa-vulkan-drivers
  # misc build deps seen in upstream install guides
  libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev xvfb
)
# Release-specific names. 24.04 dropped libgl1-mesa-glx (now libgl1 + libglx-mesa0) and ships Python 3.12.
case "$UBU" in
  22.04) APT_PKGS+=(libgl1-mesa-glx) ;;
  24.04|*) APT_PKGS+=(libgl1 libglx-mesa0 libegl-dev) ;;
esac
is_wsl && APT_PKGS+=(pciutils)   # lspci for WSL diagnostics; NEVER add nvidia-* driver packages inside WSL

# apt_known PKG -> true if apt has an installable candidate (handles renames between releases)
apt_known() { apt-cache policy "$1" 2>/dev/null | awk '/Candidate:/{print $2}' | grep -qv '^(none)$'; }

if [ "${SKIP_APT:-0}" != 1 ]; then
  missing=()
  for p in "${APT_PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if [ "${#missing[@]}" -gt 0 ]; then
    sudo apt-get update
    known=(); unknown=()
    for p in "${missing[@]}"; do if apt_known "$p"; then known+=("$p"); else unknown+=("$p"); fi; done
    [ "${#unknown[@]}" -gt 0 ] && warn "apt has no candidate for: ${unknown[*]} (Ubuntu ${UBU:-?}); skipping them"
    missing=(${known[@]+"${known[@]}"})
  fi
  if [ "${#missing[@]}" -gt 0 ]; then
    log "Installing apt packages (Ubuntu ${UBU:-?}): ${missing[*]}"
    # --no-install-recommends keeps apt from pulling unrelated GPU/driver packages
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
  else ok "apt packages already installed"; fi
else warn "SKIP_APT=1: skipping apt"; fi

git lfs install --skip-repo >/dev/null && ok "git-lfs initialised"

# ---- uv ----
if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH"; fi
if ! have uv; then
  if [ -n "${UV_VERSION:-}" ]; then curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  else curl -LsSf https://astral.sh/uv/install.sh | sh; fi
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version)"

# ---- Miniforge (conda) for repos whose official guides use conda ----
if conda_base >/dev/null 2>&1; then ok "conda found at $(conda_base)"
elif [ "${INSTALL_MINIFORGE:-1}" = 1 ]; then
  log "Installing Miniforge to $HOME/miniforge3 (batch mode, does not edit ~/.bashrc)"
  tmp="$(mktemp -d)"
  wget -q -O "$tmp/mf.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash "$tmp/mf.sh" -b -p "$HOME/miniforge3"
  rm -rf "$tmp"
  "$HOME/miniforge3/bin/conda" config --set auto_activate_base false
  ok "Miniforge installed. (Optional) run: ~/miniforge3/bin/conda init bash"
else warn "No conda and INSTALL_MINIFORGE=0: OFT/SimplerEnv/RoboTwin/SimpleVLA-RL scripts will fail"; fi

# ---- Optional side-by-side CUDA toolkit (for building flash-attn / CuRobo). Toolkit only. ----
if [ -n "${INSTALL_CUDA_TOOLKIT:-}" ]; then
  if [ -d "/usr/local/cuda-${INSTALL_CUDA_TOOLKIT/-/.}" ]; then ok "CUDA toolkit ${INSTALL_CUDA_TOOLKIT} already present"
  else
    log "Installing cuda-toolkit-${INSTALL_CUDA_TOOLKIT} from NVIDIA's apt repo (no driver packages)"
    if is_wsl; then CUDA_REPO=wsl-ubuntu      # WSL-Ubuntu repo: toolkit packages without any driver components
    else case "$UBU" in 24.04) CUDA_REPO=ubuntu2404 ;; *) CUDA_REPO=ubuntu2204 ;; esac; fi
    log "NVIDIA CUDA repo: $CUDA_REPO"
    if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
      tmp="$(mktemp -d)"
      wget -q -O "$tmp/cuda-keyring.deb" "https://developer.download.nvidia.com/compute/cuda/repos/$CUDA_REPO/x86_64/cuda-keyring_1.1-1_all.deb"
      sudo dpkg -i "$tmp/cuda-keyring.deb"; rm -rf "$tmp"
    fi
    sudo apt-get update
    # 'cuda-toolkit-X-Y' does not depend on the driver; never install the 'cuda' or 'cuda-drivers' metapackages.
    sudo apt-get install -y --no-install-recommends "cuda-toolkit-${INSTALL_CUDA_TOOLKIT}"
  fi
  warn "Do NOT put a CUDA 12 toolkit first on the global PATH; kit scripts set CUDA_HOME per env."
elif is_blackwell && ! find_cuda_home "${RSP_MIN_NVCC_BLACKWELL:-12.8}" >/dev/null; then
  log "Blackwell GPU and no CUDA >= ${RSP_MIN_NVCC_BLACKWELL:-12.8} toolkit: fine for everything in run_wsl_headless.sh."
  log "  Only needed to BUILD flash-attn / CUDA extensions for sm_120. Optional: INSTALL_CUDA_TOOLKIT=12-8 bash $0"
fi
is_wsl && log "WSL2: the GPU driver comes from Windows. This script never installs NVIDIA driver packages; do not install them yourself."

# ---- Env file users can source ----
cat > "$RSP_HOME/rsp_env.sh" <<ENVEOF
# source ~/robot-sim/rsp_env.sh  (generated by 01_base_deps.sh)
export RSP_HOME="$RSP_HOME"
export HF_HOME="\${HF_HOME:-$HF_HOME}"
export OPENPI_DATA_HOME="\${OPENPI_DATA_HOME:-$OPENPI_DATA_HOME}"
export MS_ASSET_DIR="\${MS_ASSET_DIR:-$MS_ASSET_DIR}"
# MuJoCo backend: kit scripts probe EGL and fall back to osmesa automatically; this default is for your own shells.
export MUJOCO_GL="\${MUJOCO_GL:-$(is_wsl && echo osmesa || echo egl)}"
export PYOPENGL_PLATFORM="\${PYOPENGL_PLATFORM:-\$MUJOCO_GL}"
export PATH="\$HOME/.local/bin:\$PATH"
ENVEOF
ok "Wrote $RSP_HOME/rsp_env.sh"
log "Next: bash scripts/02_isaaclab_validate.sh   then   bash scripts/10_install_lerobot.sh"
