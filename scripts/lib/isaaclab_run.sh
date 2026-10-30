#!/usr/bin/env bash
# isaaclab_run.sh cli|python ARGS...  - run inside the detected existing Isaac Lab env (no syncing).
# Lets callers wrap Isaac runs with `timeout`.
set -euo pipefail
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/isaaclab_env.sh"
isaaclab_detect
kind="$1"; shift
case "$kind" in
  cli) isaaclab_cli "$@" ;;
  python) isaaclab_python "$@" ;;
  *) die "usage: isaaclab_run.sh cli|python ARGS..." ;;
esac
