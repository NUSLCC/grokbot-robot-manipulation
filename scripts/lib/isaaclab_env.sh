#!/usr/bin/env bash
# Detect an EXISTING Isaac Lab 3.0 / Isaac Sim 6.2 install and expose runner functions.
# This file never installs or modifies anything in the Isaac environment.
#
# Overrides (export before sourcing):
#   ISAACLAB_PATH=/path/to/IsaacLab        checkout root (contains isaaclab.sh / VERSION)
#   ISAACLAB_MODE=uv|sh|python             how to launch (auto-detected)
#   ISAAC_PY=/path/to/python               interpreter that has isaaclab installed (python mode)
#   ISAACLAB_CONDA_ENV=name_or_prefix      conda env to activate before isaaclab.sh (sh mode)
#
# Isaac Lab 3.0 facts (verified from release/3.0.0 source):
#   * `isaaclab.sh` still exists but prints a deprecation warning ("removed in 3.1, use `uv run isaaclab`").
#   * isaaclab.sh picks python in this order: $VIRTUAL_ENV, $CONDA_PREFIX, $ISAACLAB_PATH/env_isaaclab,
#     $ISAACLAB_PATH/_isaac_sim/python.sh, then system python3.
#   * CLI subcommands: train, play, list_envs, zero_agent, random_agent, teleop, demo, example, benchmark, leapp.
#   * Default physics backend is Newton; Isaac Sim/PhysX is selected with the `physics=isaacsim_physx` preset.

isaaclab_find_root() {
  local c
  if [ -n "${ISAACLAB_PATH:-}" ]; then
    [ -f "$ISAACLAB_PATH/VERSION" ] || [ -f "$ISAACLAB_PATH/isaaclab.sh" ] || return 1
    echo "$ISAACLAB_PATH"; return 0
  fi
  for c in "$HOME/IsaacLab" "$HOME/isaaclab" "$HOME/workspace/IsaacLab" "$HOME/work/IsaacLab" \
           "$HOME/projects/IsaacLab" "$HOME/code/IsaacLab" "$HOME/src/IsaacLab" /opt/IsaacLab /opt/isaaclab; do
    if [ -f "$c/isaaclab.sh" ] && [ -d "$c/source/isaaclab" ]; then echo "$c"; return 0; fi
  done
  # Bounded search (depth 4 under $HOME, depth 3 under /opt)
  c="$( { find "$HOME" -maxdepth 4 -name isaaclab.sh -path '*IsaacLab*' 2>/dev/null; \
          find /opt -maxdepth 3 -name isaaclab.sh 2>/dev/null; } | head -n1 || true)"
  if [ -n "$c" ]; then dirname "$c"; return 0; fi
  return 1
}

isaacsim_find_root() {
  local c
  for c in "${ISAACSIM_PATH:-}" "${ISAACLAB_PATH:-/nonexistent}/_isaac_sim" "$HOME/isaacsim" "$HOME/isaac-sim" \
           "$HOME/isaacsim-6.2.0" /opt/isaacsim /isaac-sim; do
    [ -n "$c" ] && [ -f "$c/python.sh" ] && { readlink -f "$c"; return 0; }
  done
  return 1
}

# Decide launch mode and the python used for diagnostics.
isaaclab_detect() {
  ISAACLAB_PATH="$(isaaclab_find_root)" || { ISAACLAB_PATH=""; }
  export ISAACLAB_PATH
  ISAACSIM_ROOT="$(isaacsim_find_root || true)"
  if [ -z "${ISAACLAB_MODE:-}" ]; then
    if [ -n "$ISAACLAB_PATH" ] && [ -x "$ISAACLAB_PATH/.venv/bin/python" ] && have uv; then
      ISAACLAB_MODE=uv
    elif [ -n "${ISAAC_PY:-}" ]; then
      ISAACLAB_MODE=python
    elif [ -n "$ISAACLAB_PATH" ] && [ -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
      ISAACLAB_MODE="sh"
    else
      ISAACLAB_MODE=none
    fi
  fi
  export ISAACLAB_MODE
  if [ -z "${ISAAC_PY:-}" ]; then
    case "$ISAACLAB_MODE" in
      uv) ISAAC_PY="$ISAACLAB_PATH/.venv/bin/python" ;;
      sh)
        if [ -n "${ISAACLAB_CONDA_ENV:-}" ] && conda_base >/dev/null 2>&1; then
          local b; b="$(conda_base)"
          if [ -x "$ISAACLAB_CONDA_ENV/bin/python" ]; then ISAAC_PY="$ISAACLAB_CONDA_ENV/bin/python"
          else ISAAC_PY="$b/envs/$ISAACLAB_CONDA_ENV/bin/python"; fi
        elif [ -x "$ISAACLAB_PATH/env_isaaclab/bin/python" ]; then ISAAC_PY="$ISAACLAB_PATH/env_isaaclab/bin/python"
        elif [ -f "$ISAACLAB_PATH/_isaac_sim/python.sh" ]; then ISAAC_PY="$ISAACLAB_PATH/_isaac_sim/python.sh"
        else ISAAC_PY=""; fi ;;
      *) ISAAC_PY="" ;;
    esac
  fi
  export ISAAC_PY ISAACSIM_ROOT
}

# Run an `isaaclab <subcommand>` in the existing env WITHOUT syncing/changing it.
isaaclab_cli() {
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}" ACCEPT_EULA="${ACCEPT_EULA:-Y}"
  case "$ISAACLAB_MODE" in
    uv)     (cd "$ISAACLAB_PATH" && uv run --no-sync isaaclab "$@") ;;
    sh)     ( if [ -n "${ISAACLAB_CONDA_ENV:-}" ]; then conda_activate "$ISAACLAB_CONDA_ENV"; fi
              cd "$ISAACLAB_PATH" && ./isaaclab.sh "$@" ) ;;
    python) (cd "${ISAACLAB_PATH:-.}" && "$ISAAC_PY" -m isaaclab "$@") ;;
    *) die "Isaac Lab not detected. Set ISAACLAB_PATH (and ISAACLAB_MODE/ISAAC_PY if needed)." ;;
  esac
}

# Run a python script (path relative to ISAACLAB_PATH) in the existing env.
isaaclab_python() {
  export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}" ACCEPT_EULA="${ACCEPT_EULA:-Y}"
  case "$ISAACLAB_MODE" in
    uv)     (cd "$ISAACLAB_PATH" && uv run --no-sync python "$@") ;;
    sh)     ( if [ -n "${ISAACLAB_CONDA_ENV:-}" ]; then conda_activate "$ISAACLAB_CONDA_ENV"; fi
              cd "$ISAACLAB_PATH" && ./isaaclab.sh -p "$@" ) ;;
    python) (cd "${ISAACLAB_PATH:-.}" && "$ISAAC_PY" "$@") ;;
    *) die "Isaac Lab not detected." ;;
  esac
}

isaaclab_summary() {
  echo "  ISAACLAB_PATH = ${ISAACLAB_PATH:-<not found>}"
  echo "  ISAACLAB_MODE = ${ISAACLAB_MODE:-none}"
  echo "  ISAAC_PY      = ${ISAAC_PY:-<none>}"
  echo "  ISAACSIM_ROOT = ${ISAACSIM_ROOT:-<not found; may be pip-installed as the isaacsim wheel>}"
  if [ -n "${ISAACLAB_PATH:-}" ] && [ -f "$ISAACLAB_PATH/VERSION" ]; then
    echo "  Isaac Lab VERSION file = $(cat "$ISAACLAB_PATH/VERSION")"
  fi
}
