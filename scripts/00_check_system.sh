#!/usr/bin/env bash
# 00_check_system.sh - read-only system audit (driver, CUDA, GPU, RAM, disk, rendering, tooling, Isaac).
# Safe to run any time; changes nothing. Exit code = number of FAIL items (0 = ready).
# Works on native Ubuntu 22.04 / 24.04 and on WSL2 Ubuntu 24.04 (auto-detected; RSP_FORCE_WSL=1|0 to override).
#   RSP_DRY_RUN=1   always exit 0 (used by CI and simulated runs on machines without a GPU)
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
source "$(dirname "$0")/lib/isaaclab_env.sh"

FAILS=0; WARNS=0
f() { fail "$*"; FAILS=$((FAILS+1)); }
w() { warn "$*"; WARNS=$((WARNS+1)); }
REPORT="$RSP_LOGS/system_check_$(stamp).txt"
exec > >(tee "$REPORT") 2>&1
log "System check -> $REPORT"

echo; log "== OS =="
VERSION_ID="$(os_version_id)"
echo "  $(os_field PRETTY_NAME)"
case "$VERSION_ID" in
  22.04) ok "Ubuntu 22.04 (tested by openpi; supported by Isaac Sim 6.2 but marked deprecated there)" ;;
  24.04) ok "Ubuntu 24.04 (Isaac Sim 6.2's preferred release; system Python is 3.12, so every framework gets its own Python via uv/conda)" ;;
  *)     w "Not Ubuntu 22.04/24.04 (found ${VERSION_ID:-?}); apt package names may differ" ;;
esac
GLIBC="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"; GLIBC="${GLIBC:-0}"
ver_ge "$GLIBC" 2.35 && ok "glibc $GLIBC (Isaac Sim wheels need >= 2.35)" || f "glibc $GLIBC < 2.35"
echo "  kernel $(uname -r)"

echo; log "== Hardware profile =="
echo "  PROFILE=$PROFILE ($([ "${RSP_PROFILE_AUTO:-0}" = 1 ] && echo auto-selected || echo set by you)): $RSP_PROFILE_DESC"
echo "  GPU arch: $RSP_GPU_ARCH   VRAM class: ${RSP_VRAM_CLASS_GB} GB   torch CUDA wheels: $TORCH_CUDA   pi0 LoRA: $RSP_PI0_LORA_NOTE"
if is_wsl; then WSL_NOW=1; else WSL_NOW=0; fi
if [ -n "${RSP_EXPECT_WSL:-}" ] && [ "$RSP_EXPECT_WSL" != "$WSL_NOW" ]; then
  w "Profile $PROFILE expects $([ "$RSP_EXPECT_WSL" = 1 ] && echo WSL2 || echo native Linux) but this is $([ "$WSL_NOW" = 1 ] && echo WSL2 || echo native Linux). Set PROFILE=... if auto-selection picked the wrong machine."
fi
[ "${RSP_IS_LAPTOP:-0}" = 1 ] && echo "  Laptop GPU: plug in the charger and set Windows power mode to 'Best performance'; sustained clocks are lower (runs are slower) and it throttles when hot."

if [ "$WSL_NOW" = 1 ]; then
  echo; log "== WSL2 =="
  echo "  distro: ${WSL_DISTRO_NAME:-<WSL_DISTRO_NAME unset>}   kernel: $(uname -r)"
  if [ -d /usr/lib/wsl/lib ]; then
    ok "/usr/lib/wsl/lib exists (GPU user-mode libraries come from the Windows driver)"
    ls /usr/lib/wsl/lib/libcuda.so* >/dev/null 2>&1 && ok "libcuda from the Windows driver found in /usr/lib/wsl/lib" \
      || f "/usr/lib/wsl/lib has no libcuda.so*: update the NVIDIA driver on WINDOWS, then run 'wsl --shutdown' in PowerShell"
  else
    f "/usr/lib/wsl/lib missing: GPU passthrough is not set up. Update WSL ('wsl --update' in PowerShell) and the Windows NVIDIA driver"
  fi
  NVSMI="$(command -v nvidia-smi || true)"
  case "$NVSMI" in
    /usr/lib/wsl/lib/*) ok "nvidia-smi is the Windows-provided one ($NVSMI)" ;;
    "") : ;;  # reported in the GPU section
    *) w "nvidia-smi resolves to $NVSMI, not /usr/lib/wsl/lib/nvidia-smi: a Linux NVIDIA package may be installed inside WSL" ;;
  esac
  LINUX_NV="$(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 'nvidia-driver-*' 'nvidia-dkms-*' 'nvidia-kernel-*' 'libnvidia-gl-*' 'nvidia-utils-*' 'cuda-drivers*' 2>/dev/null | awk '$1 ~ /^ii/ {print $2}' | tr '\n' ' ' || true)"
  if [ -n "$LINUX_NV" ]; then
    f "Linux NVIDIA driver packages are installed INSIDE WSL ($LINUX_NV). Never install a Linux NVIDIA driver in WSL: it overrides the Windows driver's libraries. Remove them (sudo apt-get remove --purge $LINUX_NV) and run 'wsl --shutdown'."
  else
    ok "No Linux NVIDIA driver packages inside WSL (correct: the driver comes from Windows; never install one here)"
  fi
  case "$RSP_KIT_DIR" in
    /mnt/*)
      w "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      w "This repo is on the Windows filesystem ($RSP_KIT_DIR). /mnt/* is 5-20x slower for git, uv and Python"
      w "imports and breaks symlinks/permissions. Clone it into your Linux home instead:  cd ~ && git clone ..."
      w "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" ;;
    *) ok "Repo is on the Linux filesystem ($RSP_KIT_DIR)" ;;
  esac
  case "$RSP_HOME" in
    /mnt/*) f "RSP_HOME=$RSP_HOME is on the Windows filesystem. Envs, caches and checkpoints must live on the Linux side (default ~/robot-sim)." ;;
  esac
  if [ -d /mnt/c ]; then
    CFREE="$(df -BG --output=avail /mnt/c 2>/dev/null | tail -n1 | tr -dc 0-9 || true)"
    [ -n "$CFREE" ] && echo "  Windows C: free: ${CFREE} GB (the WSL virtual disk ext4.vhdx grows into this space; df below shows the virtual disk)"
    if [ -n "$CFREE" ] && [ "$CFREE" -lt 150 ]; then w "Only ${CFREE} GB free on Windows C:. The WSL disk grows on C:, so real free space is the smaller of the two."; fi
  fi
fi

echo; log "== CPU / RAM / disk =="
echo "  CPU: $(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}') ($(nproc) threads)"
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$RAM_GB" -ge 64 ]; then ok "RAM ${RAM_GB} GB (Isaac Sim 'good' = 64 GB)"
elif [ "$RAM_GB" -ge 32 ]; then w "RAM ${RAM_GB} GB: meets Isaac Sim minimum (32 GB); VLA data loaders + sim may swap"
else f "RAM ${RAM_GB} GB < 32 GB (Isaac Sim minimum)"; fi
SWAP_GB=$(awk '/SwapTotal/{printf "%d", $2/1024/1024}' /proc/meminfo); echo "  swap ${SWAP_GB} GB"
if [ "$WSL_NOW" = 1 ]; then
  echo "  (WSL2 sees only what .wslconfig gives it; the default is 50% of Windows RAM)"
  if [ "$RAM_GB" -lt 48 ] || [ "$SWAP_GB" -lt 16 ]; then
    w "WSL2 has ${RAM_GB} GB RAM / ${SWAP_GB} GB swap. Raise it in %UserProfile%\\.wslconfig on Windows, then run 'wsl --shutdown':"
    cat <<'WSLCFG'
      [wsl2]
      memory=48GB      # at most ~75% of the PC's RAM (laptop: e.g. 48GB of 64GB, 24GB of 32GB)
      swap=32GB
      processors=16    # leave a few cores for Windows
WSLCFG
  fi
fi
DISK_FREE_GB=$(df -BG --output=avail "$RSP_HOME" | tail -n1 | tr -dc 0-9)
# Planning estimates (not official): Phase 1 (~120–180 GB), Phase 2 (+80–120 GB), Phase 3 (+80–150 GB)
if [ "$DISK_FREE_GB" -ge 300 ]; then ok "Free disk at $RSP_HOME: ${DISK_FREE_GB} GB"
elif [ "$DISK_FREE_GB" -ge 250 ]; then ok "Free disk ${DISK_FREE_GB} GB (Phase 1 + most of Phase 2 fit; monitor usage)"
elif [ "$DISK_FREE_GB" -ge 150 ]; then w "Free disk ${DISK_FREE_GB} GB: enough for Phase 1 only (est. 120–180 GB); consider moving RSP_HOME"
elif [ "$WSL_NOW" = 1 ] && [ "$DISK_FREE_GB" -ge 100 ]; then w "Free disk ${DISK_FREE_GB} GB: enough for the run_wsl_headless.sh set (est. 60-100 GB), not for Phase 1+2"
else f "Free disk ${DISK_FREE_GB} GB at $RSP_HOME (needs >=150 GB for Phase 1)"; fi
echo "  Estimated additional disk by phase: P1 120–180 GB, P2 +80–120 GB, P3 +80–150 GB"
echo "  /dev/shm: $(df -h --output=size /dev/shm | tail -n1 | tr -d ' ') (PyTorch dataloaders / Docker use shared memory)"

echo; log "== NVIDIA driver / GPU =="
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p | tr -d ' ')"
  DRV_CUDA="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | awk '{print $3}' || echo '?')"
  # Summarize all GPUs
  GPU_LINES="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | sed 's/, /,/g')"
  GPU_COUNT="$(printf '%s\n' "$GPU_LINES" | grep -c . || echo 0)"
  echo "  GPUs: $GPU_COUNT device(s) detected"
  i=0
  while IFS=, read -r NAME VRAMMiB CAP; do
    i=$((i+1)); printf '    [%d] %s  VRAM: %s  compute: %s\n' "$i" "$NAME" "$VRAMMiB" "$CAP"
  done <<<"$GPU_LINES"
  echo "  Driver: $DRV  (reports CUDA runtime: $DRV_CUDA)"
  ver_ge "$DRV" "$DRIVER_TESTED_ISAAC" && ok "Driver >= $DRIVER_TESTED_ISAAC (Isaac Sim 6.2 / Isaac Lab 3.0 tested driver; also covers cu130 PyTorch)" \
    || { ver_ge "$DRV" "$DRIVER_MIN_CU130" && w "Driver $DRV < $DRIVER_TESTED_ISAAC (Isaac tested version) but >= cu130 floor" \
         || f "Driver $DRV below cu130 floor $DRIVER_MIN_CU130 (Isaac Lab 3.0 torch 2.11 cu130 needs it)"; }
  ver_ge "$DRV" "$DRIVER_MIN_CU128" || f "Driver below $DRIVER_MIN_CU128 needed by LeRobot/GR00T cu128 wheels"
  # Accept 24 GB class (RTX 3090 / 4090 / 5090 Laptop)
  FIRST_VRAM="$(printf '%s\n' "$GPU_LINES" | sed -n 1p | awk -F, '{gsub(/ MiB/,"",$2);print $2+0}')"
  [ "${FIRST_VRAM:-0}" -ge 23000 ] && ok "Per-GPU VRAM ${FIRST_VRAM} MiB (24 GB class)" || w "Per-GPU VRAM ${FIRST_VRAM} MiB < 24 GB; the VRAM-fit table assumes 24 GB"
  [ "$GPU_COUNT" -lt "${RSP_EXPECT_GPUS:-1}" ] && w "Profile $PROFILE expects ${RSP_EXPECT_GPUS} GPU(s), found $GPU_COUNT"
  CC_MAX="$(printf '%s\n' "$GPU_LINES" | awk -F, '{print $3}' | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -n1 || true)"
  case "${CC_MAX%%.*}" in
    '') w "compute capability not reported by nvidia-smi" ;;
    1[0-9])
      ok "Blackwell GPU (compute capability $CC_MAX, sm_${CC_MAX/./}) detected"
      echo "  Blackwell needs PyTorch built for CUDA >= 12.8 (torch >= 2.7, cu128 wheels) in every env:"
      echo "    OK (cu128)       : LeRobot (ACT/DP/SmolVLA/pi0.5), ManiSkill3, GR00T N1.7, Isaac Lab 3.0 (torch 2.11 cu130, untouched)"
      echo "    unverified       : openpi (JAX cuda12 serving; PyTorch path needs OPENPI_TORCH_CU128=1)"
      echo "    NOT expected     : OpenVLA-OFT (torch 2.2), SimpleVLA-RL (torch 2.4), RoboTwin (CUDA 12.1 stack) - no sm_120 kernels"
      echo "    CPU-only, fine   : openpi's Python 3.8 LIBERO client (torch 1.11 is not used on the GPU)"
      ver_ge "$DRV" 570 || f "Driver $DRV < 570: Blackwell + cu128 wheels need R570 or newer" ;;
    *) ok "Compute capability $CC_MAX (pre-Blackwell: every torch pinned by the kit has kernels for it)" ;;
  esac
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  running: /' || true
  PERS="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | sed -n 1p)"; echo "  persistence mode: $PERS"
else
  if [ "${WSL_NOW:-0}" = 1 ]; then
    f "nvidia-smi not working inside WSL. Install/update the NVIDIA driver on WINDOWS (not in WSL), run 'wsl --update' and 'wsl --shutdown' in PowerShell, reopen Ubuntu."
  else
    f "nvidia-smi not working: install/repair the NVIDIA driver first (this kit never touches the driver)"
  fi
fi

echo; log "== CUDA toolkit (nvcc) =="
# Wheels bring their own CUDA runtime; nvcc is only needed to BUILD extensions
# (flash-attn for OpenVLA-OFT/SimpleVLA-RL, CuRobo/pytorch3d for RoboTwin). nvcc major must match torch's CUDA major.
ls -d /usr/local/cuda-* 2>/dev/null | sed 's/^/  found: /' || true
if have nvcc; then echo "  nvcc on PATH: $(nvcc --version | grep -oE 'release [0-9.]+' )"
elif [ -x /usr/local/cuda/bin/nvcc ]; then echo "  /usr/local/cuda/bin/nvcc: $(/usr/local/cuda/bin/nvcc --version | grep -oE 'release [0-9.]+')"
else w "No nvcc: fine for inference/most training; needed for flash-attn / CuRobo builds (see README 'CUDA toolkits')"; fi
if ! ls -d /usr/local/cuda-12* >/dev/null 2>&1; then
  w "No CUDA 12.x toolkit. OpenVLA-OFT (torch 2.2 cu121), SimpleVLA-RL (cu124), RoboTwin (CUDA 12.1 rec.) builds want a 12.x nvcc."
fi
if is_blackwell; then
  if find_cuda_home "${RSP_MIN_NVCC_BLACKWELL:-12.8}" >/dev/null; then ok "CUDA toolkit >= ${RSP_MIN_NVCC_BLACKWELL:-12.8} found ($(find_cuda_home "${RSP_MIN_NVCC_BLACKWELL:-12.8}")): can build sm_120 extensions"
  else echo "  No CUDA >= ${RSP_MIN_NVCC_BLACKWELL:-12.8} toolkit: flash-attn builds are skipped on Blackwell (optional: INSTALL_CUDA_TOOLKIT=12-8 bash scripts/01_base_deps.sh)"; fi
fi

echo; log "== Headless rendering =="
if ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia.so'; then ok "NVIDIA EGL present (MUJOCO_GL=egl)"
elif [ "$WSL_NOW" = 1 ]; then w "No NVIDIA EGL (normal on WSL2). MuJoCo evals (LIBERO, ALOHA) probe EGL and fall back to MUJOCO_GL=osmesa (CPU, slower)."
else f "libEGL_nvidia.so not found (MuJoCo/LIBERO headless EGL)"; fi
if ldconfig -p 2>/dev/null | grep -q 'libOSMesa.so'; then ok "OSMesa present (fallback MUJOCO_GL=osmesa)"
elif [ "$WSL_NOW" = 1 ]; then f "OSMesa missing: on WSL2 it is the MuJoCo fallback renderer. Run scripts/01_base_deps.sh (installs libosmesa6)"
else w "OSMesa missing (01_base_deps.sh installs libosmesa6-dev)"; fi
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then ok "NVIDIA Vulkan ICD present (SAPIEN: ManiSkill/SimplerEnv/RoboTwin)"
elif [ "$WSL_NOW" = 1 ]; then w "No NVIDIA Vulkan ICD (normal on WSL2). SAPIEN rendering may fail; ManiSkill PPO falls back to state-only (no rendering), SimplerEnv/RoboTwin evals may not run."
else f "NVIDIA Vulkan ICD json missing (SAPIEN rendering will fail)"; fi
if have vulkaninfo; then
  VK="$(vulkan_status)"; echo "  vulkan devices: $VK"
  timeout 60 vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName' | sed 's/^/  /' | head -n6 || true
  [ "$WSL_NOW" = 1 ] && [ "$VK" != nvidia ] && echo "  (WSL2: SAPIEN needs a real GPU Vulkan device; '$VK' means GPU rendering in ManiSkill is not expected to work)"
else w "vulkaninfo not installed (apt: vulkan-tools)"; fi
[ -n "${DISPLAY:-}" ] && echo "  DISPLAY=$DISPLAY (GUI available; kit scripts never open windows)" || echo "  No DISPLAY: headless mode (expected over SSH)"

echo; log "== Tooling =="
for t in git git-lfs curl ffmpeg cmake gcc uv docker; do
  if have "$t"; then echo "  $t: $($t --version 2>&1 | sed -n 1p)"; else w "$t missing"; fi
done
FF="$(ffmpeg -version 2>/dev/null | sed -n 1p | grep -oE 'version [^ ]+' | awk '{print $2}' || true)"
[ -n "$FF" ] && echo "  ffmpeg $FF (torchcodec for GR00T needs FFmpeg 4-7; Ubuntu 22.04 ships 4.4, 24.04 ships 6.1)"
if conda_base >/dev/null 2>&1; then echo "  conda: $(conda_base)"; else w "conda/Miniforge missing (needed for OFT, SimplerEnv, RoboTwin, SimpleVLA-RL)"; fi
if have docker; then
  docker info 2>/dev/null | grep -qi 'nvidia' && ok "docker has nvidia runtime" || w "docker present; NVIDIA container toolkit not detected (only needed for Docker options)"
fi
echo "  python3 (system): $(python3 --version 2>&1) (never pip-installed into; PEP 668 on 24.04)"

echo; log "== Environment pollution =="
[ -n "${PYTHONPATH:-}" ] && w "PYTHONPATH is set ($PYTHONPATH). Isaac Sim's setup_*_env.sh can set this; it breaks other envs. Kit scripts unset it." || ok "PYTHONPATH not set"
[ -n "${CONDA_PREFIX:-}" ] && w "A conda env is active ($CONDA_PREFIX). Run kit scripts from a clean shell." || true
echo "${LD_LIBRARY_PATH:-}" | grep -qiE 'isaac|omni|kit' && w "LD_LIBRARY_PATH contains Isaac paths" || true
grep -nE 'setup_conda_env|setup_python_env|isaac' "$HOME/.bashrc" 2>/dev/null | sed 's/^/  ~\/.bashrc: /' || true

echo; log "== Existing Isaac Lab / Isaac Sim (detect only) =="
isaaclab_detect
isaaclab_summary
if [ "$ISAACLAB_MODE" = none ]; then w "Isaac Lab not auto-detected; set ISAACLAB_PATH and run 02_isaaclab_validate.sh"
else ok "Isaac Lab detected (mode=$ISAACLAB_MODE). Run scripts/02_isaaclab_validate.sh for the full check."; fi

echo; log "== Summary =="
echo "  FAIL: $FAILS   WARN: $WARNS   report: $REPORT"
if [ "${RSP_DRY_RUN:-0}" = 1 ]; then log "RSP_DRY_RUN=1: exiting 0 (would have exited $FAILS)"; exit 0; fi
exit "$FAILS"
