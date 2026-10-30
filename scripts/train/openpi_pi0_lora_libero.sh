#!/usr/bin/env bash
# openpi LoRA fine-tune of pi0 on LIBERO (config pi0_libero_low_mem_finetune: gemma_2b_lora + gemma_300m_lora, EMA off).
# openpi README: LoRA fine-tuning needs > 22.5 GB (24 GB class e.g., RTX 3090/4090) -> fits, but tightly; close other GPU apps.
#   CONFIG=pi0_fast_libero_low_mem_finetune also exists.  (pi05_libero in openpi is a FULL fine-tune -> > 70 GB.)
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"; clean_python_env; apply_gpu_selection; ensure_uv
: "${CONFIG:=pi0_libero_low_mem_finetune}"; : "${EXP:=rsp_lora_$(stamp)}"; : "${EXTRA_ARGS:=}"
SRC="$RSP_SRC/openpi"; [ -d "$SRC/.venv" ] || die "Run scripts/11_install_openpi.sh first"
cd "$SRC"
UVRUN=(uv run); [ -f .rsp_no_sync ] && UVRUN=(uv run --no-sync)   # see 11_install_openpi.sh OPENPI_TORCH_CU128
if ! ls assets/"$CONFIG"/*/norm_stats.json >/dev/null 2>&1; then
  log "Computing norm stats for $CONFIG (downloads physical-intelligence/libero LeRobot dataset)"
  "${UVRUN[@]}" scripts/compute_norm_stats.py --config-name "$CONFIG"
fi
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
gpu_monitor_start "$RSP_LOGS/openpi_$EXP.gpu.csv"
# shellcheck disable=SC2086
"${UVRUN[@]}" scripts/train.py "$CONFIG" --exp-name="$EXP" --overwrite $EXTRA_ARGS
echo "Peak GPU memory: $(gpu_monitor_stop) MiB"
ok "Checkpoints: $SRC/checkpoints/$CONFIG/$EXP ; serve with SERVER_ARGS=\"policy:checkpoint --policy.config=$CONFIG --policy.dir=checkpoints/$CONFIG/$EXP/<step>\" bash scripts/eval/openpi_pi05_libero.sh"
