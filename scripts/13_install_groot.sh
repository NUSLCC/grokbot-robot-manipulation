#!/usr/bin/env bash
# 13_install_groot.sh - NVIDIA Isaac-GR00T N1.7 (tag n1.7-release) + its LIBERO / SimplerEnv sim clients.
# Server env: $RSP_HOME/src/Isaac-GR00T/.venv (uv sync --python 3.12; CUDA 12.8 wheels, flash-attn/TensorRT included).
# Each sim benchmark has its own setup script that builds a separate client venv (official design).
# GR00T's backbone nvidia/Cosmos-Reason2-2B is GATED: request access, then `uv run hf auth login`.
#   GROOT_SIMS="LIBERO SimplerEnv"   which sim clients to set up (names of gr00t/eval/sim/<X>/setup_*.sh)
# Blackwell: GR00T's own CUDA 12.8 wheels include sm_120 (per its README dGPU stack); not tested by this kit.
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv
: "${GROOT_SIMS:=LIBERO}"

SRC="$RSP_SRC/Isaac-GR00T"
have git-lfs || die "git-lfs required (01_base_deps.sh)"
check_disk_space 50 "$RSP_HOME"
git_sync https://github.com/NVIDIA/Isaac-GR00T.git "$SRC" "$GROOT_REF"
cd "$SRC"
git lfs pull || warn "git lfs pull failed (demo_data parquet files)"

FF="$(ffmpeg -version 2>/dev/null | head -n1 | grep -oE 'version [0-9]+' | awk '{print $2}' || echo 0)"
[ "$FF" -ge 4 ] && [ "$FF" -le 7 ] && ok "ffmpeg major $FF (torchcodec 0.8.0 needs 4-7)" || warn "ffmpeg major '$FF' not in 4-7; torchcodec may fail"

log "uv sync --python $GROOT_PY"
uv sync --python "$GROOT_PY"
uv run python -c "import gr00t, torch; print('GR00T installed;', 'torch', torch.__version__, torch.version.cuda, torch.cuda.is_available())"

if [ -z "${CUDA_HOME:-}" ]; then
  warn "CUDA_HOME unset. Fine-tuning may fail with 'CUDA_HOME is unset' (upstream note): export CUDA_HOME=/usr/local/cuda-12.8 if you have it."
fi
hf_check_access model nvidia/Cosmos-Reason2-2B >/dev/null 2>&1 || true

for s in $GROOT_SIMS; do
  f="$(ls gr00t/eval/sim/"$s"/setup_*.sh 2>/dev/null | head -n1 || true)"
  if [ -n "$f" ]; then log "Sim client setup: $f"; bash "$f"; else warn "No setup script for sim '$s' under gr00t/eval/sim/"; fi
done
[ -f scripts/eval/check_sim_eval_ready.py ] && uv run python scripts/eval/check_sim_eval_ready.py || true
ok "GR00T ready at $SRC"
