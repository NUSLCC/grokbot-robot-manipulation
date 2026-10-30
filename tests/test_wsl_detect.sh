#!/usr/bin/env bash
# Unit tests for WSL / OS / GPU / profile detection in scripts/lib/common.sh. No GPU needed.
#   bash tests/test_wsl_detect.sh      (exit code = number of failed assertions)
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"; K="$(cd "$T/.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
export RSP_HOME="$TMP/rsp"
nfail=0
check() { if [ "$2" = "$3" ]; then echo "ok   - $1"; else echo "FAIL - $1: expected '$3', got '$2'"; nfail=$((nfail+1)); fi; }
# run SNIPPET with extra env, sourcing common.sh in a clean subshell; prints the snippet output
run() { local snip="$1"; shift; env -u WSL_DISTRO_NAME -u PROFILE -u RSP_GPU_DETECTED -u RSP_GPU_ARCH -u TORCH_CUDA "$@" \
          bash -c "source '$K/scripts/lib/common.sh' >/dev/null 2>&1; set +e; $snip" 2>/dev/null; }

echo "Linux version 6.6.87.2-microsoft-standard-WSL2 (root@x) #1 SMP" > "$TMP/proc_wsl"
echo "Linux version 6.8.0-49-generic (buildd@lcy02) #49~22.04.1-Ubuntu SMP" > "$TMP/proc_native"
printf 'PRETTY_NAME="Ubuntu 24.04.1 LTS"\nVERSION_ID="24.04"\n' > "$TMP/os2404"
printf 'PRETTY_NAME="Ubuntu 22.04.5 LTS"\nVERSION_ID="22.04"\n' > "$TMP/os2204"
NOGPU="$TMP/nogpu"; mkdir -p "$NOGPU"   # PATH without nvidia-smi
BASEPATH="/usr/local/bin:/usr/bin:/bin"

check "WSL from /proc/version"        "$(run 'is_wsl && echo y || echo n' RSP_PROC_VERSION_FILE="$TMP/proc_wsl" PATH="$BASEPATH")" y
check "native from /proc/version"     "$(run 'is_wsl && echo y || echo n' RSP_PROC_VERSION_FILE="$TMP/proc_native" PATH="$BASEPATH")" n
check "WSL from WSL_DISTRO_NAME"      "$(run 'is_wsl && echo y || echo n' RSP_PROC_VERSION_FILE="$TMP/proc_native" WSL_DISTRO_NAME=Ubuntu-24.04 PATH="$BASEPATH")" y
check "RSP_FORCE_WSL=0 wins"          "$(run 'is_wsl && echo y || echo n' RSP_FORCE_WSL=0 WSL_DISTRO_NAME=Ubuntu-24.04 PATH="$BASEPATH")" n
check "RSP_FORCE_WSL=1 wins"          "$(run 'is_wsl && echo y || echo n' RSP_FORCE_WSL=1 RSP_PROC_VERSION_FILE="$TMP/proc_native" PATH="$BASEPATH")" y
check "os_version_id 24.04"           "$(run 'os_version_id' RSP_OS_RELEASE_FILE="$TMP/os2404" PATH="$BASEPATH")" 24.04
check "os_version_id 22.04"           "$(run 'os_version_id' RSP_OS_RELEASE_FILE="$TMP/os2204" PATH="$BASEPATH")" 22.04

FAKE="$T/fake-bin:$BASEPATH"
check "5090 Laptop -> rtx5090_laptop_wsl" "$(run 'echo $PROFILE' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 5090 Laptop GPU" FAKE_GPU_CC=12.0)" rtx5090_laptop_wsl
check "5090 Laptop is Blackwell"      "$(run 'is_blackwell && echo y || echo n' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 5090 Laptop GPU" FAKE_GPU_CC=12.0)" y
check "5090 Laptop VRAM class 24"     "$(run 'echo $RSP_VRAM_CLASS_GB' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 5090 Laptop GPU" FAKE_GPU_CC=12.0)" 24
check "5090 Laptop torch cu128"       "$(run 'echo $TORCH_CUDA $LEROBOT_TORCH_BACKEND' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 5090 Laptop GPU" FAKE_GPU_CC=12.0)" "cu128 cu128"
check "2x3090 -> dual3090"            "$(run 'echo $PROFILE $RSP_GPU_COUNT' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 3090" FAKE_GPU_CC=8.6 FAKE_GPU_COUNT=2)" "dual3090 2"
check "3090 not Blackwell"            "$(run 'is_blackwell && echo y || echo n' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 3090" FAKE_GPU_CC=8.6)" n
check "3090 keeps PR #1 cu128"        "$(run 'echo $LEROBOT_TORCH_BACKEND' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 3090" FAKE_GPU_CC=8.6)" cu128
check "4090 -> rtx4090"               "$(run 'echo $PROFILE $RSP_GPU_ARCH' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 4090" FAKE_GPU_CC=8.9)" "rtx4090 ada"
check "desktop 5090 -> generic+Blackwell" "$(run 'echo $PROFILE $RSP_GPU_ARCH' PATH="$FAKE" FAKE_GPU_NAME="NVIDIA GeForce RTX 5090" FAKE_GPU_CC=12.0 FAKE_GPU_MEM=32607)" "generic blackwell"
check "no GPU -> generic/unknown"     "$(run 'echo $PROFILE $RSP_GPU_ARCH $RSP_GPU_COUNT' PATH="$BASEPATH")" "generic unknown 0"
check "PROFILE override"              "$(run 'echo $PROFILE $RSP_GPU_ARCH' PATH="$BASEPATH" PROFILE=rtx5090_laptop_wsl)" "rtx5090_laptop_wsl blackwell"
check "user TORCH_CUDA wins"          "$(run 'echo $TORCH_CUDA' PATH="$BASEPATH" PROFILE=rtx5090_laptop_wsl TORCH_CUDA=cu129)" cu129
check "bad PROFILE dies"              "$(run 'echo alive' PATH="$BASEPATH" PROFILE=nope || echo died)" died
check "skip exit code is 77"          "$(run 'skip_exit test' PATH="$BASEPATH"; echo $?)" 77

# 00_check_system.sh in simulated WSL + dry run must finish with exit 0 and print the WSL section
out="$(env -u PROFILE PATH="$FAKE" RSP_FORCE_WSL=1 RSP_DRY_RUN=1 WSL_DISTRO_NAME=Ubuntu-24.04 bash "$K/scripts/00_check_system.sh" 2>&1)"; rc=$?
check "00_check_system dry-run exit 0" "$rc" 0
check "00 prints WSL2 section"        "$(grep -c '== WSL2 ==' <<<"$out")" 1
check "00 flags Blackwell"            "$(grep -c 'Blackwell GPU (compute capability 12.0' <<<"$out")" 1
check "00 laptop note"                "$(grep -c 'Laptop GPU: plug in' <<<"$out")" 1
echo; [ "$nfail" = 0 ] && echo "ALL PASS" || echo "$nfail FAILED"
exit "$nfail"
