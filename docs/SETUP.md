# Setup and design

This page holds the kit's architecture, install steps, version pins, and how a first validation run is meant to go. For a short overview see the [README](../README.md). Hardware results are in [VALIDATION.md](VALIDATION.md).
## Target hardware

| Target | Value |
|---|---|
| OS | Ubuntu 22.04 (native). WSL2 Ubuntu 24.04 is supported but is not being tested for now ([WSL2.md](WSL2.md)). |
| GPU | RTX 3090 (24 GB, sm_86, 1 or 2 GPUs) or RTX 4090 (24 GB, sm_89) |
| Driver | 595 |
| Already installed | **Isaac Sim 6.2.0 + Isaac Lab 3.0.0.** The kit never reinstalls or changes them unless you set `ALLOW_ISAAC_ENV_CHANGES=1`. |

Methods outside the 14-step headless runner (OpenVLA-OFT, GR00T, SimplerEnv, RoboTwin, RLinf, SimpleVLA-RL, Arena, the VLA fine-tuning scripts) have not yet been run to completion on these GPUs. Items marked **(unverified)** are our best reading of the sources, not measured facts.

## Architecture

```
                         ┌────────────── RTX 3090 / 4090 (24 GB) ─────────────────┐
  Policy frameworks      │  LeRobot env     openpi env       OpenVLA-OFT env   GR00T env      RLinf env │
  (one env each,         │  py3.12/cu128    py3.11/JAX+cu12  py3.10/torch2.2   py3.12/cu128   py3.11    │
   never shared)         │  ACT, DP,        pi0, pi0-FAST,   OpenVLA-OFT 7B    GR00T N1.7 3B  PPO/GRPO  │
                         │  SmolVLA, pi0/   pi0.5 (server)                     (server)       on VLAs   │
                         │  pi0.5, GR00T                                                              │
                         ├──────────────── websocket / ZMQ policy servers  ◄──►  sim clients ─────────┤
  Simulators /           │  MuJoCo (EGL):  LIBERO, ALOHA-sim, PushT                                   │
  benchmarks             │  SAPIEN (Vulkan): ManiSkill3, SimplerEnv, RoboTwin 2.0                      │
                         │  Isaac (existing): Isaac Lab 3.0 RL / Mimic IL, Isaac Sim 6.2 PhysX+RTX     │
                         └────────────────────────────────────────────────────────────────────────────┘
  Everything the kit creates lives in ~/robot-sim/{src,envs,data,checkpoints,logs,hf_cache}  (RSP_HOME)
```

Design rules:
1. **One environment per framework.** Their pins conflict. Examples: torch 2.2 for OFT, 2.7.1 for openpi, 2.11
   for Isaac Lab and ≥2.10 for LeRobot. transformers is a fork for OFT, 4.53.2 for openpi, 5.4–5.5 for LeRobot and 5.10.4 for Isaac Lab.
   numpy is <2 for openpi and SimplerEnv but ≥2 for Isaac Lab.
2. **The Isaac environment is read-only to this kit.** Isaac runs go through `uv run --no-sync`, `isaaclab.sh` or the
   detected interpreter, and nothing is ever installed into that environment unless you set `ALLOW_ISAAC_ENV_CHANGES=1`.
3. **Server/client evaluation.** openpi, GR00T and RLinf already split the policy and the simulator into separate processes.
   That keeps the policy environment and the old simulator environments (Python 3.8 for LIBERO) apart.
4. Each script uses `set -euo pipefail`, can be rerun safely, writes to `~/robot-sim/logs`, and logs the commit of every repo it clones
   to `logs/commits.txt`.

## Step-by-step setup

```bash
# clone and enter this repository, then:
# git clone <this-repo-url> && cd <repo>
bash scripts/00_check_system.sh          # read-only audit; exit code = number of FAIL items
bash scripts/01_base_deps.sh             # apt packages, uv, Miniforge (asks for sudo). Driver/Isaac untouched.
bash scripts/02_isaaclab_validate.sh     # finds your Isaac Lab 3.0 / Isaac Sim 6.2 and runs a headless smoke test
bash scripts/10_install_lerobot.sh       # phase 1 core
bash scripts/10b_install_lerobot_libero.sh   # LIBERO env (MuJoCo 3.3.2; >=3.4 breaks libero_spatial task 5)
bash scripts/11_install_openpi.sh        # phase 1 core
bash scripts/12_install_openvla_oft.sh   # phase 1 core (SKIP_FLASH_ATTN=1 for inference only)
bash scripts/13_install_groot.sh         # phase 2 (needs gated nvidia/Cosmos-Reason2-2B access)
bash scripts/14_install_maniskill.sh     # phase 2
bash scripts/15_install_simplerenv.sh    # phase 2
bash scripts/16_install_robotwin.sh      # phase 2 (builds CuRobo, so it needs a CUDA 12.x nvcc)
bash scripts/17_install_rlinf.sh         # phase 3 (RLINF_USE_DOCKER=1 for the official image)
bash scripts/18_install_simplevla_rl.sh  # phase 3, optional: code/eval only on 24 GB
I_UNDERSTAND_ARENA_IS_ALPHA=1 bash scripts/19_install_isaaclab_arena.sh   # optional
```
Or run phase 1 end to end: `QUICK=1 bash scripts/run_phase1.sh`.

### Multi-GPU (optional, Ampere 3090 x2)
- Set GPUs for any script with GPUS or NUM_GPUS:
  - Example: `GPUS=0,1 bash scripts/11_install_openpi.sh` (JAX will use both devices); `NUM_GPUS=2 bash scripts/train/isaaclab_rl_franka.sh` (limits visibility to the first 2).
  - The kit applies `CUDA_VISIBLE_DEVICES` automatically from these variables; defaults remain single-GPU.
- Safe concurrency: run two independent jobs, one per card:
  - Terminal A: `CUDA_VISIBLE_DEVICES=0 bash scripts/train/lerobot_act_aloha.sh`
  - Terminal B: `CUDA_VISIBLE_DEVICES=1 TRIALS=5 SUITE=libero_object bash scripts/eval/openpi_pi05_libero.sh`

**Before downloading models:** run `~/robot-sim/envs/lerobot/bin/hf auth login` and accept the licenses for the gated repos
`google/paligemma-3b-pt-224` (π0/π0.5 in LeRobot) and `nvidia/Cosmos-Reason2-2B` (all GR00T N1.7 checkpoints).

Useful overrides: `RSP_HOME=/data/robot-sim` (put everything on a big disk), `HF_HOME`, any pin in
`scripts/lib/versions.env` (for example `LEROBOT_REF=main`), and `ISAACLAB_PATH` / `ISAACLAB_CONDA_ENV` / `ISAAC_PY`.

### Pinned versions (from official sources, 2026-10-29)

| Component | Pin | Python | Torch / CUDA wheel | Install method |
|---|---|---|---|---|
| LeRobot | tag `v0.6.1` (latest release) | 3.12 | torch ≥2.7,<2.12 from the **cu128** index (driver ≥ 570.86) | uv venv, editable + extras |
| openpi | `main` (no tags; commit logged) | ≥3.11 (uv) | jax[cuda12]==0.5.3, torch==2.7.1, transformers==4.53.2 | `uv sync` |
| openpi LIBERO client | — | **3.8** | torch cu113 | separate uv venv (upstream design) |
| OpenVLA-OFT | `main` | 3.10 | torch==2.2.0 (cu121 wheel), transformers fork, flash-attn==2.5.5 | conda |
| Isaac-GR00T | tag `n1.7-release` | 3.12 | CUDA 12.8 wheels, transformers 4.57.3, torchcodec 0.8.0 (FFmpeg 4–7) | `uv sync` |
| ManiSkill3 | `mani_skill==3.0.1` | 3.11 (our choice) | latest torch, cu128 | uv venv |
| SimplerEnv | `main` | 3.10 | numpy==1.24.4 | conda |
| RoboTwin 2.0 | `main` | 3.10 | CUDA 12.1 recommended | conda + `scripts/_install.sh` |
| RLinf | tag `v0.4` | 3.11.14 default | per install target | `requirements/install.sh` or Docker |
| SimpleVLA-RL | `main` + veRL `v0.2.x` | 3.10 | torch==2.4.0 cu124 | conda |
| **Isaac Lab (existing)** | `3.0.0` | **3.12** | **torch==2.11.0 cu130**, warp 1.17.0, transformers 5.10.4, rsl-rl-lib 5.5.1 | yours (uv / isaaclab.sh / pip) |
| **Isaac Sim (existing)** | `6.2.0` (`isaacsim[all,extscache]==6.2.0.0`) | 3.12 | — | yours |

### Driver 595 and CUDA/PyTorch conflicts

* **The driver is not the conflict.** Driver 595 meets every wheel used here: cu113, cu121, cu124, cu128 (floor 570.86),
  cu130 (floor 580.65), JAX cuda12, and the Isaac Sim 6.2 tested driver 595.58.03. CUDA drivers are backward compatible.
  Sub-versions under 595.58.03 trigger a WARN in `00_check_system.sh`.
* **The real conflicts are user-space and compile time:**
  1. **Isaac env torch 2.11 (cu130) versus everything else.** Never `pip install` LeRobot, openpi, GR00T etc. into the Isaac env.
     The kit gives each one its own env.
  2. **`nvcc` versions.** flash-attn (OFT, SimpleVLA-RL), CuRobo and pytorch3d (RoboTwin) compile against the *system* nvcc,
     and its major version must match torch's CUDA major (12.x). If `/usr/local/cuda` is 13.x, install a CUDA 12 toolkit
     next to it (`INSTALL_CUDA_TOOLKIT=12-4 bash scripts/01_base_deps.sh`, which installs the toolkit only and never the driver) and
     export `CUDA_HOME=/usr/local/cuda-12.4` for those scripts only. Do not put it first on your global PATH.
  3. **Isaac Sim environment scripts.** `setup_conda_env.sh` / `setup_python_env.sh` export `PYTHONPATH` and `LD_LIBRARY_PATH`.
     If your `~/.bashrc` sources them, every other env breaks. Kit scripts strip these (`clean_python_env`) and `00_check_system.sh` flags them.
  4. **openpi's PyTorch path** patches `transformers` files in place. The kit sets `UV_LINK_MODE=copy` so the uv cache is not modified.
  5. **The LeRobot "IsaacLab Arena" EnvHub guide targets Isaac Sim 5.1 / Isaac Lab 2.3 / Python 3.11.** Do **not** follow it on
     this machine, because it would downgrade an Isaac stack. Use Arena `main` (built for Isaac Lab 3.0) in its own venv (`19_…`).
  6. **RLinf inside Isaac Lab** (official Isaac Lab doc) installs `transformers==4.51.3` and `--no-deps` packages *into the Isaac env*.
     See [ISAAC_LAB.md](ISAAC_LAB.md) for a safer way to run it.

## First validation run (recommended order)

**Step 0: platform sanity (~15 min).** Run `bash scripts/02_isaaclab_validate.sh`. It locates Isaac Lab, checks Python 3.12,
torch 2.11 cu130 and CUDA from inside the Isaac env, then runs `list_envs` and a 3-iteration headless rsl_rl training on `Isaac-Reach-Franka` and
`Isaac-Lift-Franka`. Each runs with the default **Newton** backend and again with **`physics=isaacsim_physx`**, which is the run that actually exercises Isaac Sim 6.2.

**Step 1: evaluate a released VLA on LIBERO (SmolVLA, smallest and quickest).**
```bash
bash scripts/eval/lerobot_smolvla_libero.sh                    # libero_object, 2 episodes/task (quick)
SUITES=libero_spatial,libero_object,libero_goal,libero_10 N_EPISODES=10 bash scripts/eval/lerobot_smolvla_libero.sh  # protocol run (400 episodes)
```
Success: `eval_info.json` with per-suite `pc_success`, plus the Wilson 95% interval that LeRobot reports. Use 10 episodes per task
and average over 3 seeds before comparing against published numbers.

**Step 2: reproduce a published SOTA number (π0.5, openpi official path).**
```bash
bash scripts/11_install_openpi.sh
TRIALS=5 SUITE=libero_spatial bash scripts/eval/openpi_pi05_libero.sh   # official reference: 98.8% (spatial)
```
Then compare with `scripts/eval/openvla_oft_libero.sh` (OFT, ~16 GB) and `scripts/eval/lerobot_pi05_libero.sh`.

**Step 3: train an IL policy from scratch.**
```bash
bash scripts/train/lerobot_act_aloha.sh          # ACT, ALOHA transfer-cube (sim eval every 5k steps)
bash scripts/train/lerobot_diffusion_pusht.sh    # Diffusion Policy, PushT
```

**Step 4: fine-tune a VLA within 24 GB.**
```bash
bash scripts/train/lerobot_smolvla_libero.sh     # full FT of SmolVLA (BATCH=8)
bash scripts/train/openpi_pi0_lora_libero.sh     # pi0 LoRA (>22.5 GB, so close other GPU apps)
```

Every eval/train script prints the **peak GPU memory** it saw, from `nvidia-smi` sampling. Use that to fill in the "unverified" cells above.

## Phased plan

| Phase | Goal | Scripts | Exit criteria |
|---|---|---|---|
| **1: Core** (days 1–3) | Toolchain, Isaac validated, LIBERO evaluation of 3 VLA families, IL training | 00, 01, 02, 10, 11, 12; `eval/lerobot_smolvla_libero`, `eval/openpi_pi05_libero`, `eval/openvla_oft_libero`; `train/lerobot_act_aloha`, `train/lerobot_diffusion_pusht` | π0.5 / OFT LIBERO numbers within a few points of the published ones; ACT/DP checkpoints with nonzero success |
| **2: More benchmarks** (week 2) | Robustness and real-to-sim, bimanual, GPU-parallel, Isaac IL | 13 (GR00T LIBERO + SimplerEnv), 14 (ManiSkill3), 15 (SimplerEnv), 16 (RoboTwin 2.0); `train/isaaclab_rl_franka`, `train/isaaclab_mimic_stack`; optional 19 (Arena) | GR00T LIBERO eval; SimplerEnv/RoboTwin rollouts; Isaac Lift-Franka policy trained; Mimic BC policy evaluated |
| **3: RL fine-tuning** (week 3+) | Post-training with RL, as in the review | `train/maniskill_ppo` (fits), Isaac Lab RL (fits), 17 + `rl/rlinf_maniskill_quickstart` (experimental on 24 GB), 18 (SimpleVLA-RL, eval/code only) | PPO baselines converge; RLinf runs at least a few iterations or you decide to rent multi-GPU |

For phase 3 on one 24 GB GPU, start with methods whose trainable part is small: ManiSkill or Isaac Lab PPO, and RLinf's
DSRL for π0 (a small SAC agent in the diffusion noise space; RLinf docs). RL on 7B VLAs (SimpleVLA-RL, RLinf + OFT) was validated
upstream on 8× 80 GB GPUs.

## Headless rendering

On WSL2 (and whenever NVIDIA EGL or Vulkan is missing), MuJoCo evals probe EGL and fall back to `MUJOCO_GL=osmesa`.
ManiSkill PPO falls back to no-render state mode, and Isaac Lab runs with `--viz none --livestream 0` (Isaac Lab 3.0 rejects
`--headless`). See [WSL2.md](WSL2.md).

| Simulator | Backend | Setting |
|---|---|---|
| LIBERO / robosuite / ALOHA / PushT (MuJoCo) | EGL (NVIDIA) | `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (kit default). Fallbacks: `osmesa` (CPU, slow, needs `libosmesa6-dev`) or `glx` with a display or `xvfb-run`. |
| ManiSkill3, SimplerEnv, RoboTwin (SAPIEN) | Vulkan | needs `/usr/share/vulkan/icd.d/nvidia_icd.json` + `libvulkan1`; check with `vulkaninfo --summary`. In Docker add `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (RoboTwin docs). |
| Isaac Lab / Isaac Sim | headless by default (no `--viz`) | `OMNI_KIT_ACCEPT_EULA=YES` for first headless run; `--video` for recordings; `--viz kit` only with a display. |

## Docker

See [DOCKER.md](DOCKER.md). In short: openpi `examples/libero/compose.yml` (openpi's recommended LIBERO path; needs rootless Docker and the
NVIDIA container toolkit), the GR00T `docker/README.md`, the RLinf image `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`,
`huggingface/lerobot-gpu:latest`, ManiSkill `maniskill/base` (last pushed 2025-03, so it is old), the Isaac Lab `docker/` folder, and Arena `docker/run_docker.sh`.

## Caveats and unverified items

* VRAM cells marked "—" or "unverified" have no official figure. Measure them with the scripts' peak-memory output.
* `lerobot/smolvla_libero` rename map (`image→camera1`, `image2→camera2`) was inferred from its `config.json`.
* LeRobot docs were read from `main` (0.6.2.dev) while the kit pins `v0.6.1`. If a flag is missing, set `LEROBOT_REF=main`.
* Isaac Lab: the task IDs, CLI subcommands, `physics=isaacsim_physx` and the Mimic commands come from the `release/3.0.0` source and docs.
  Whether every task accepts every physics preset is unverified (check `list_envs`). `python -m isaaclab` (python mode) is inferred from the CLI code.
* Disk-space thresholds in `00_check_system.sh` (150/300 GB) are our planning estimates, not official numbers.
* SimpleVLA-RL and RLinf feasibility on one 24 GB GPU is unverified. Upstream test rigs are 8× 80 GB.

Sources: [SOURCES.md](SOURCES.md). Isaac Lab workflows: [ISAAC_LAB.md](ISAAC_LAB.md). Troubleshooting: [TROUBLESHOOTING.md](../TROUBLESHOOTING.md). VRAM / 24 GB fit: [MODELS.md](MODELS.md).
