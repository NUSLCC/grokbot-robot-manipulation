# Isaac Lab 3.0 / Isaac Sim 6.2 in this platform

Your workstation already has **Isaac Sim 6.2.0 + Isaac Lab 3.0.0 (release/3.0.0)**. The kit treats that install as a fixed,
read-only part of the platform. It detects it, validates it, and runs workflows against it. It never reinstalls or upgrades it.

## What Isaac Lab 3.0 / Isaac Sim 6.2 require (verified 2026-10-29)

| Item | Requirement | Source |
|---|---|---|
| Python | **3.12** (`requires-python >=3.12,<3.13`) | IsaacLab release/3.0.0 `pyproject.toml` |
| PyTorch | **torch==2.11.0, torchvision==0.26.0** from the **cu130** index | same |
| Other pins | warp-lang 1.17.0, transformers 5.10.4, rsl-rl-lib 5.5.1, numpy>=2, `isaacsim[all,extscache]==6.2.0.0` (extra `isaacsim`) | same |
| Driver | Linux x86_64 **595.58.03** (Isaac Sim 6.2 tested and Isaac Lab doc); default CUDA 13 stack needs Turing+ | Isaac Lab install doc; Isaac Sim 6.2 requirements |
| Isaac Sim HW | min: 32 GB RAM, 50 GB SSD, RTX 4080-class, 16 GB VRAM. Good: 64 GB RAM, 500 GB SSD. glibc ≥ 2.35 for the wheels | Isaac Sim 6.2 requirements |
| OS | Ubuntu 22.04/24.04. **Isaac Sim marks 22.04 as deprecated** (still supported in 6.2) | Isaac Sim 6.2 requirements |
| Default physics | **Newton (Warp)**. Isaac Sim is optional and selected with `physics=isaacsim_physx` | Isaac Lab install doc / README |

### How it may be installed on your machine (the kit handles all of these)

| Install style | How to recognise it | How the kit launches it |
|---|---|---|
| uv from the checkout (recommended in 3.0) | `$ISAACLAB_PATH/.venv` + `uv.lock` | `cd $ISAACLAB_PATH && uv run --no-sync isaaclab …` (`--no-sync` = never modifies the env) |
| `isaaclab.sh` (deprecated in 3.0, to be removed in 3.1) | `isaaclab.sh` uses `$VIRTUAL_ENV` → `$CONDA_PREFIX` → `env_isaaclab/` → `_isaac_sim/python.sh` → system python3 | `./isaaclab.sh …` (set `ISAACLAB_CONDA_ENV=<env>` if it lives in conda) |
| Isaac Sim **binary/workstation** build linked as `_isaac_sim` | `$ISAACLAB_PATH/_isaac_sim/python.sh` | via isaaclab.sh (it sources `_isaac_sim/setup_*_env.sh`) |
| pip wheels into your own venv/conda | `isaacsim` and `isaaclab` packages present in that Python | `ISAACLAB_MODE=python ISAAC_PY=/path/to/python` |

If auto-detection misses your setup, export `ISAACLAB_PATH=/path/to/IsaacLab` (plus `ISAACLAB_CONDA_ENV` or `ISAAC_PY`).

## Validation: `scripts/02_isaaclab_validate.sh`

1. Locates the checkout (`VERSION` should read `3.0.0`) and the Isaac Sim root or wheel.
2. Probes the Isaac Python (read-only). It checks the Python version, torch/CUDA, a GPU matmul, the isaacsim wheel version and whether
   rsl_rl, isaaclab_mimic and robomimic are present.
3. `isaaclab list_envs`, saving all Franka, Allegro and Mimic manipulation task IDs to `manipulation_tasks.txt`.
4. Headless RL smoke tests: `isaaclab train --rl_library rsl_rl --task {Isaac-Reach-Franka, Isaac-Lift-Franka}
   --num_envs 64 --max_iterations 3`. Each runs with the default Newton backend **and** `physics=isaacsim_physx` (the run that exercises Isaac Sim).
   The peak GPU memory of each run is recorded.
5. Prints a PASS/WARN/FAIL table. The exit code is the number of FAILs. Logs go to `~/robot-sim/logs/isaaclab_validate_*`.

Tune it with `SMOKE_TASKS`, `SMOKE_NUM_ENVS`, `SMOKE_ITERS`, `SMOKE_PHYSICS="default"` (Newton only), or `SKIP_SMOKE=1`.

## Manipulation task IDs in release/3.0.0 (from source)

Isaac Lab 3.0 renamed tasks: they no longer carry the `-v0` suffix, and community tasks moved to `IsaacContrib-*`.

* Core: `Isaac-Reach-Franka`, `Isaac-Reach-Franka-OSC`, `Isaac-Reach-UR10`, `Isaac-Lift-Franka`,
  `Isaac-Open-Drawer-Franka`, `Isaac-Open-Drawer-Franka-Direct`, `Isaac-Lift-KukaAllegro(-Camera)`, `Isaac-Reorient-KukaAllegro(-Camera)`,
  deformables `Isaac-Lift-{Soft,Cloth,Cable}-Franka(-Camera)`.
* Contrib: `IsaacContrib-Lift-Cube-Franka(-IK-Abs|-IK-Rel)`, `IsaacContrib-Reach-Franka-IK-Abs`,
  `IsaacContrib-Open-Drawer-Franka-IK-{Abs,Rel}`, `IsaacContrib-Stack-Cube-Franka(-IK-Rel, -IK-Rel-Visuomotor, -IK-Rel-Visuomotor-Cosmos, -IK-Rel-Skillgen, …)`, plus OpenArm variants.
* Mimic (data generation): `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`, `…-Visuomotor-Mimic-v0`, `…-IK-Abs-Mimic-v0`,
  `Isaac-PickPlace-GR1T2-Abs-Mimic-v0`, `Isaac-NutPour-GR1T2-Pink-IK-Abs-Mimic-v0`, `Isaac-Locomanipulation-G1-Abs-Mimic-v0`, Galbot and Agibot variants.

Always confirm on your machine with `bash scripts/lib/isaaclab_run.sh cli list_envs`.

## Workflows relevant to the review

### A. RL on manipulation tasks (fits on a 24 GB 3090-class GPU)
`TASK=Isaac-Lift-Franka NUM_ENVS=1024 bash scripts/train/isaaclab_rl_franka.sh` trains with rsl_rl and then plays the policy.
Other options: `RL_LIB=skrl|sb3|rl_games` (only if that extra is installed in your env), `PHYSICS=isaacsim_physx`, `VIDEO=1`.
Upstream equivalents: `uv run isaaclab train --rl_library rsl_rl --task … [physics=…]` and `isaaclab play …`.

### B. Imitation learning with Isaac Lab Mimic (data multiplication + robomimic BC)
`bash scripts/train/isaaclab_mimic_stack.sh` follows the official tutorial without teleoperation:
download the official Franka stack human dataset (`…/Isaac/6.2/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5`),
run `annotate_demos.py --auto`, run `generate_dataset.py` (small test, then `N_GEN` demos), train `robomimic/train.py --algo bc`,
and evaluate with `robomimic/play.py --num_rollouts 50`. The visuomotor variant uses the `…-Visuomotor` task IDs.
It needs the `isaacsim` and `mimic` extras in your Isaac env. The script refuses to add them unless `ALLOW_ISAAC_ENV_CHANGES=1`.
Related upstream features: SkillGen (cuRobo-based), humanoid (GR1T2/G1) imitation, Cosmos-augmented visuomotor data,
and teleop recording (`isaaclab teleop record/replay`, keyboard/SpaceMouse/XR).
**Bridge to the VLAs in the review:** Mimic-generated HDF5 can be converted to LeRobot format to train ACT, DP, SmolVLA or GR00T.
Isaac Lab ships a GR00T conversion script for the locomanipulation SDG data (`scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py`).
A generic Mimic→LeRobot converter for the Franka stack task is **not** part of Isaac Lab 3.0 (unverified; we did not find one).

### C. RL post-training of VLAs inside Isaac Lab (RLinf; official experimental feature)
Isaac Lab 3.0 includes `--rl_library rlinf` (`docs/source/experimental-features/rlinf_vla_posttraining.rst`). It fine-tunes GR00T
(example task `assemble_trocar`, checkpoint `nvidia/Assemble_Trocar`) with PPO, Actor-Critic or SAC:
`uv run --no-sync isaaclab train --rl_library rlinf --config_name isaaclab_ppo_gr00t_assemble_trocar --model_path …`.
**Caution:** the documented install runs `uv sync --inexact --extra rlinf --extra video`, then installs `rlinf==0.2.0dev2`,
`transformers==4.51.3` and Isaac-GR00T (pinned commit) with `--no-deps` and flash-attn 2.8.3 **into the Isaac env**. That replaces
Isaac Lab's transformers 5.10.4. Safer options are to clone your checkout to a second directory and `uv sync` it into its own `.venv`
before following the doc, or to snapshot the env first (`cp -a .venv .venv.bak`). The doc recommends multiple GPUs, and single-4090 feasibility is unverified.
Our kit does not automate this.

### D. GR00T and LeRobot with Isaac
* **Isaac Lab-Arena** (`19_install_isaaclab_arena.sh`, alpha v0.3, main targets Isaac Lab 3.0 / Isaac Sim 6.2) is NVIDIA's policy
  evaluation framework. It runs GR00T, π0.5 or custom policies behind a server with a common observation/action contract, and registers Arena
  environments with Isaac Lab for RL and data generation. It builds its own `.venv`.
* **LeRobot EnvHub `isaaclab_arena`** (`lerobot-eval --env.type=isaaclab_arena …`, released checkpoints `nvidia/smolvla-arena-gr1-microwave`,
  `nvidia/pi05-arena-gr1-microwave`). Its guide currently pins **Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11**, which is **incompatible**
  with your 6.2/3.0 stack. Only use it in a separate env built from that guide, never in your Isaac env.
* GR00T N1.7 itself ships sim examples for LIBERO, SimplerEnv, RoboCasa and RoboLab (`scripts/eval/groot_libero.sh`).

### E. Sim-to-real
Isaac Lab 3.0 docs under `policy_deployment/` cover gear-assembly sim-to-real, a reach policy, an SO-101 typing policy, COMPASS
navigation with NuRec, and **LEAPP** export/deploy of trained policies (`isaaclab leapp export|deploy`). Domain randomisation presets come from
`presets=` in the task config. Use these for policies trained in workflow A/B. For VLAs, deployment follows each framework's
server/client path (openpi websocket, GR00T ZMQ `PolicyClient`).

## Isolation rules (keep the Isaac env healthy)
* All other frameworks live in `~/robot-sim/envs/*` or their own repo `.venv`. Nothing is installed with the Isaac Python.
* Do not source `_isaac_sim/setup_conda_env.sh` or `setup_python_env.sh` in `~/.bashrc`. It leaks `PYTHONPATH` into other envs.
* Use `uv run --no-sync` (not plain `uv run`) in the Isaac checkout. Otherwise uv may re-sync the lockfile and undo local changes.
* Before any change you do make to the Isaac env (extras such as `mimic`, `skrl`, `rlinf`), snapshot it: `cp -a $ISAACLAB_PATH/.venv $ISAACLAB_PATH/.venv.bak-$(date +%F)`.
