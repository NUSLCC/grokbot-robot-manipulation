#!/usr/bin/env bash
# 11_install_openpi.sh - Physical Intelligence openpi (pi0, pi0-FAST, pi0.5; JAX + PyTorch) + LIBERO client env.
# Server env: $RSP_HOME/src/openpi/.venv (uv sync; Python >=3.11, jax[cuda12]==0.5.3, torch==2.7.1).
# Client env: $RSP_HOME/src/openpi/examples/libero/.venv (Python 3.8, torch cu113) - as in examples/libero/README.md.
# Set OPENPI_PYTORCH=1 to also apply the transformers patches needed for the PyTorch models.
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

if [ "${OPENPI_PYTORCH:-0}" = 1 ]; then
  TV="$(uv pip show transformers 2>/dev/null | awk '/^Version/{print $2}')"
  [ "$TV" = "4.53.2" ] || warn "openpi PyTorch path expects transformers 4.53.2, found $TV"
  SITE="$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')"
  log "Applying openpi transformers patches into $SITE/transformers (UV_LINK_MODE=$UV_LINK_MODE avoids editing the uv cache)"
  cp -r ./src/openpi/models_pytorch/transformers_replace/* "$SITE/transformers/"
fi

log "LIBERO client env (Python 3.8)"
if [ ! -x examples/libero/.venv/bin/python ]; then uv venv --python 3.8 examples/libero/.venv; fi
uv pip sync --python examples/libero/.venv/bin/python examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
uv pip install --python examples/libero/.venv/bin/python -e third_party/libero
# LIBERO asks interactively for a dataset path on first import; answer "N" (use defaults) once.
printf 'N\n' | PYTHONPATH="$SRC/third_party/libero" MUJOCO_GL=egl examples/libero/.venv/bin/python -c "from libero.libero import benchmark; print('LIBERO ok')" \
  || warn "LIBERO first import failed; try MUJOCO_GL=glx or osmesa (see README troubleshooting)"

log "Server sanity check"
.venv/bin/python -c "import jax, openpi; print('jax', jax.__version__, jax.devices())"
ok "openpi ready. Checkpoints download from gs://openpi-assets to \$OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
