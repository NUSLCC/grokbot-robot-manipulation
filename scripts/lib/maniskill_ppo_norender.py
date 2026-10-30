"""Run ManiSkill's examples/baselines/ppo/ppo.py with rendering disabled (no Vulkan needed, e.g. WSL2).

ppo.py hard-codes render_mode="rgb_array" (ManiSkill v3.0.1). This wrapper patches gymnasium.make so every env
is created with render_backend="none" and render_mode=None, then runs ppo.py unchanged. Pass
--no-capture-video (videos need rendering). Usage: python maniskill_ppo_norender.py /path/to/ppo.py [ppo args...]
"""
import runpy
import sys

import gymnasium as gym

_orig_make = gym.make


def _make_norender(*args, **kwargs):
    kwargs["render_backend"] = "none"
    kwargs["render_mode"] = None
    return _orig_make(*args, **kwargs)


gym.make = _make_norender
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
print("[rsp] ManiSkill rendering disabled (render_backend='none'); state observations only", flush=True)
runpy.run_path(script, run_name="__main__")
