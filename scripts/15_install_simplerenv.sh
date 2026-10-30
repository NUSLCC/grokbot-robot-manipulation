#!/usr/bin/env bash
# 15_install_simplerenv.sh - SimplerEnv (real-to-sim eval for Google Robot / WidowX; SAPIEN + ManiSkill2_real2sim).
# Env: conda prefix $RSP_HOME/envs/conda/simplerenv (Python 3.10, numpy==1.24.4 per README). Minimal install only:
# the RT-1/Octo "full installation" (tensorflow) is NOT done here. For VLA evals on SimplerEnv prefer the policy
# repo's own client (e.g. GR00T: GROOT_SIMS=SimplerEnv bash scripts/13_install_groot.sh).
set -euo pipefail
source "$(dirname "$0")/lib/common.sh"
clean_python_env; ensure_conda

PREFIX="$RSP_ENVS/conda/simplerenv"; SRC="$RSP_SRC/SimplerEnv"
conda_env_create "$PREFIX" "$SIMPLER_PY"; PY="$PREFIX/bin/python"
git_sync https://github.com/simpler-env/SimplerEnv.git "$SRC" "$SIMPLER_REF"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install 'setuptools<70'
"$PY" -m pip install "numpy==$SIMPLER_NUMPY"
"$PY" -m pip install -e "$SRC/ManiSkill2_real2sim"
"$PY" -m pip install -e "$SRC"
"$PY" - <<'PY'
import simpler_env
print("SimplerEnv import ok; skipping env.make() on CPU-only host")
PY
ok "SimplerEnv env: $PREFIX"
