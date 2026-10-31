#!/usr/bin/env python3
"""Headless check that a LeRobot sim env package imports, builds and renders.

usage: lerobot_sim_probe.py ENV_TYPE [TASK]      ENV_TYPE: aloha | pusht (package gym_<type>)
exit 0: ok (prints "sim probe ok: ...")   3: package not installed   4: package imports but make/reset/render failed

Run before a LeRobot train/eval so a missing or broken simulator is found in seconds, not after training.
MUJOCO_GL / PYOPENGL_PLATFORM (aloha) and SDL_VIDEODRIVER (pusht) come from the caller.
"""
import importlib
import os
import sys

DEFAULT_TASK = {"aloha": "AlohaTransferCube-v0", "pusht": "PushT-v0"}


def main() -> int:
    env_type = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else DEFAULT_TASK.get(env_type, "")
    pkg = f"gym_{env_type}"
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        importlib.import_module(pkg)
    except ModuleNotFoundError as e:
        print(f"sim probe: {pkg} is not installed ({e})")
        return 3
    try:
        import gymnasium as gym

        env = gym.make(f"{pkg}/{task}", render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        env.close()
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) != 3:
            raise RuntimeError(f"render() returned {type(frame).__name__} {shape}, expected an HxWx3 image")
    except Exception as e:  # noqa: BLE001 - report any make/render failure as one line
        print(f"sim probe: {pkg}/{task} imports but make/reset/render failed: {type(e).__name__}: {e}")
        return 4
    gl = os.environ.get("MUJOCO_GL", "-") if env_type == "aloha" else os.environ.get("SDL_VIDEODRIVER", "-")
    print(f"sim probe ok: {pkg}/{task} render {tuple(shape)} ({'MUJOCO_GL' if env_type == 'aloha' else 'SDL_VIDEODRIVER'}={gl})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
