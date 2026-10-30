#!/usr/bin/env bash
# 00_check_system.sh - read-only system audit (driver, CUDA, GPU, RAM, disk, rendering, tooling, Isaac).
# Safe to run any time; changes nothing. Exit code = number of FAIL items (0 = ready).
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
if [ -r /etc/os-release ]; then . /etc/os-release; echo "  $PRETTY_NAME"; fi
[ "${VERSION_ID:-}" = "22.04" ] && ok "Ubuntu 22.04 (tested by openpi; supported by Isaac Sim 6.2 but marked deprecated there)" \
  || w "Not Ubuntu 22.04 (found ${VERSION_ID:-?}); kit was written for 22.04"
GLIBC="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)"
ver_ge "$GLIBC" 2.35 && ok "glibc $GLIBC (Isaac Sim wheels need >= 2.35)" || f "glibc $GLIBC < 2.35"
echo "  kernel $(uname -r)"

echo; log "== CPU / RAM / disk =="
echo "  CPU: $(lscpu 2>/dev/null | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2;exit}') ($(nproc) threads)"
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$RAM_GB" -ge 64 ]; then ok "RAM ${RAM_GB} GB (Isaac Sim 'good' = 64 GB)"
elif [ "$RAM_GB" -ge 32 ]; then w "RAM ${RAM_GB} GB: meets Isaac Sim minimum (32 GB); VLA data loaders + sim may swap"
else f "RAM ${RAM_GB} GB < 32 GB (Isaac Sim minimum)"; fi
SWAP_GB=$(awk '/SwapTotal/{printf "%d", $2/1024/1024}' /proc/meminfo); echo "  swap ${SWAP_GB} GB"
DISK_FREE_GB=$(df -BG --output=avail "$RSP_HOME" | tail -n1 | tr -dc 0-9)
# Planning estimates (not official): Phase 1 (~120–180 GB), Phase 2 (+80–120 GB), Phase 3 (+80–150 GB)
if [ "$DISK_FREE_GB" -ge 300 ]; then ok "Free disk at $RSP_HOME: ${DISK_FREE_GB} GB"
elif [ "$DISK_FREE_GB" -ge 250 ]; then ok "Free disk ${DISK_FREE_GB} GB (Phase 1 + most of Phase 2 fit; monitor usage)"
elif [ "$DISK_FREE_GB" -ge 150 ]; then w "Free disk ${DISK_FREE_GB} GB: enough for Phase 1 only (est. 120–180 GB); consider moving RSP_HOME"
else f "Free disk ${DISK_FREE_GB} GB at $RSP_HOME (needs >=150 GB for Phase 1)"; fi
echo "  Estimated additional disk by phase: P1 120–180 GB, P2 +80–120 GB, P3 +80–150 GB"
echo "  /dev/shm: $(df -h --output=size /dev/shm | tail -n1 | tr -d ' ') (PyTorch dataloaders / Docker use shared memory)"

echo; log "== NVIDIA driver / GPU =="
if have nvidia-smi && nvidia-smi >/dev/null 2>&1; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
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
  # Accept 24 GB class (e.g., RTX 3090/4090)
  FIRST_VRAM="$(printf '%s\n' "$GPU_LINES" | head -n1 | awk -F, '{gsub(/ MiB/,"",$2);print $2+0}')"
  [ "${FIRST_VRAM:-0}" -ge 23000 ] && ok "Per-GPU VRAM ${FIRST_VRAM} MiB (24 GB class)" || w "Per-GPU VRAM ${FIRST_VRAM} MiB < 24 GB; the VRAM-fit table assumes 24 GB"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  running: /' || true
  PERS="$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -n1)"; echo "  persistence mode: $PERS"
else
  f "nvidia-smi not working: install/repair the NVIDIA driver first (this kit never touches the driver)"
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

echo; log "== Headless rendering =="
ldconfig -p 2>/dev/null | grep -q 'libEGL_nvidia.so' && ok "NVIDIA EGL present (MUJOCO_GL=egl)" || f "libEGL_nvidia.so not found (MuJoCo/LIBERO headless EGL)"
ldconfig -p 2>/dev/null | grep -q 'libOSMesa.so' && ok "OSMesa present (fallback MUJOCO_GL=osmesa)" || w "OSMesa missing (01_base_deps.sh installs libosmesa6-dev)"
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then ok "NVIDIA Vulkan ICD present (SAPIEN: ManiSkill/SimplerEnv/RoboTwin)"
else f "NVIDIA Vulkan ICD json missing (SAPIEN rendering will fail)"; fi
if have vulkaninfo; then
  vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName' | sed 's/^/  /' | head -n4 || w "vulkaninfo failed"
else w "vulkaninfo not installed (apt: vulkan-tools)"; fi
[ -n "${DISPLAY:-}" ] && echo "  DISPLAY=$DISPLAY (GUI available)" || echo "  No DISPLAY: headless mode (expected over SSH)"

echo; log "== Tooling =="
for t in git git-lfs curl ffmpeg cmake gcc uv docker; do
  if have "$t"; then echo "  $t: $($t --version 2>&1 | head -n1)"; else w "$t missing"; fi
done
FF="$(ffmpeg -version 2>/dev/null | head -n1 | grep -oE 'version [^ ]+' | awk '{print $2}' || true)"
[ -n "$FF" ] && echo "  ffmpeg $FF (torchcodec for GR00T needs FFmpeg 4-7; Ubuntu 22.04 ships 4.4)"
if conda_base >/dev/null 2>&1; then echo "  conda: $(conda_base)"; else w "conda/Miniforge missing (needed for OFT, SimplerEnv, RoboTwin, SimpleVLA-RL)"; fi
if have docker; then
  docker info 2>/dev/null | grep -qi 'nvidia' && ok "docker has nvidia runtime" || w "docker present; NVIDIA container toolkit not detected (only needed for Docker options)"
fi
echo "  python3 (system): $(python3 --version 2>&1)"

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
exit "$FAILS"
