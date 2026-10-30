"""ManiSkill3 rendering probe.  usage: python maniskill_probe.py gpu|none [ENV_ID]

gpu  : state obs + rgb_array render through SAPIEN's Vulkan renderer (what ppo.py does by default)
none : render_backend="none" (no Vulkan needed), state obs, GPU PhysX sim when CUDA is available
Exit 0 on success, 1 on failure. Prints one line: MANISKILL_PROBE <mode> OK|FAIL <reason>
"""
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "gpu"
env_id = sys.argv[2] if len(sys.argv) > 2 else "PushCube-v1"
try:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers envs)
    import torch

    sim = "physx_cuda" if torch.cuda.is_available() else "physx_cpu"
    kw = dict(num_envs=2 if sim == "physx_cuda" else 1, obs_mode="state", sim_backend=sim)
    if mode == "none":
        kw.update(render_backend="none")
    else:
        kw.update(render_mode="rgb_array")
    env = gym.make(env_id, **kw)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    if mode != "none":
        env.render()
    env.close()
    print(f"MANISKILL_PROBE {mode} OK sim={sim}")
    sys.exit(0)
except Exception as e:  # noqa: BLE001
    print(f"MANISKILL_PROBE {mode} FAIL {type(e).__name__}: {e}")
    sys.exit(1)
