#!/usr/bin/env bash
# 11_install_openpi.sh - Physical Intelligence openpi (pi0, pi0-FAST, pi0.5; JAX + PyTorch) + LIBERO client env.
# Server env: $RSP_HOME/src/openpi/.venv (uv sync; Python >=3.11, jax[cuda12]==0.5.3, torch==2.7.1).
# Client env: $RSP_HOME/src/openpi/examples/libero/.venv (Python 3.8, torch cu113) - as in examples/libero/README.md.
# Set OPENPI_PYTORCH=1 to also apply the transformers patches needed for the PyTorch models.
# Blackwell (RTX 5090): the JAX path (pi0.5 serving, LoRA training) uses jax[cuda12]==0.5.3 - sm_120 support is
# UNVERIFIED. The PyTorch path pins torch 2.7.1; OPENPI_TORCH_CU128=1 swaps in the 2.7.1+cu128 wheel (has sm_120)
# and writes .rsp_no_sync so the kit's openpi scripts use `uv run --no-sync` (plain `uv run` would re-sync it back).
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_uv

SRC="$RSP_SRC/openpi"
check_disk_space 30 "$RSP_HOME"
git_sync https://github.com/Physical-Intelligence/openpi.git "$SRC" "$OPENPI_REF"
cd "$SRC"

log "uv sync (server env)"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
rm -f .rsp_no_sync   # uv sync just restored openpi's locked torch; re-created below only with OPENPI_TORCH_CU128=1

if [ "${OPENPI_PYTORCH:-0}" = 1 ]; then
  TV="$(uv pip show transformers 2>/dev/null | awk '/^Version/{print $2}')"
  [ "$TV" = "4.53.2" ] || warn "openpi PyTorch path expects transformers 4.53.2, found $TV"
  SITE="$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')"
  log "Applying openpi transformers patches into $SITE/transformers (UV_LINK_MODE=$UV_LINK_MODE avoids editing the uv cache)"
  cp -r ./src/openpi/models_pytorch/transformers_replace/* "$SITE/transformers/"
fi

if is_blackwell; then
  warn "Blackwell GPU: openpi's JAX stack (jax[cuda12]==0.5.3) on sm_120 is unverified; the PyTorch path pins torch $OPENPI_TORCH_VERSION."
  if [ "${OPENPI_TORCH_CU128:-0}" = 1 ]; then
    log "OPENPI_TORCH_CU128=1: installing torch==$OPENPI_TORCH_VERSION from the cu128 index into .venv"
    uv pip install --python .venv/bin/python --torch-backend cu128 "torch==$OPENPI_TORCH_VERSION"
    touch .rsp_no_sync
    warn "Wrote $SRC/.rsp_no_sync: kit scripts now call 'uv run --no-sync' here. Delete it and rerun this script to undo."
  else
    log "  (PyTorch models only: rerun with OPENPI_TORCH_CU128=1 to use the cu128 torch build.)"
  fi
fi

log "LIBERO client env (Python 3.8)"
# The client's torch 1.11+cu113 has no Ampere+/Blackwell-specific needs: the client only runs the MuJoCo sim and
# talks to the policy server over a websocket, so it does not use the GPU for compute.
if [ ! -x examples/libero/.venv/bin/python ]; then uv venv --python 3.8 examples/libero/.venv; fi
uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python -e third_party/libero
# LIBERO asks interactively for a dataset path on first import; answer "N" (use defaults) once.
select_mujoco_gl "$SRC/examples/libero/.venv/bin/python"
printf 'N\n' | PYTHONPATH="$SRC/third_party/libero" examples/libero/.venv/bin/python -c "from libero.libero import benchmark; print('LIBERO ok')" \
  || warn "LIBERO first import failed; try MUJOCO_GL=glx or osmesa (see README troubleshooting)"

log "Server sanity check"
.venv/bin/python -c "import jax, openpi; print('jax', jax.__version__, jax.devices())"
ok "openpi ready. Checkpoints download from gs://openpi-assets to \$OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
