# Robot Manipulation Simulation Platform (RTX 4090 / Ubuntu 22.04 / Isaac Lab 3.0)

## Quick start

```bash
git clone <this-repo-url>
cd <repo>
bash scripts/00_check_system.sh
bash scripts/01_base_deps.sh
bash scripts/02_isaaclab_validate.sh
QUICK=1 bash scripts/run_phase1.sh
```

A setup kit for a simulation **training and evaluation** platform that checks the recent manipulation methods
from the Notion review (VLAs, imitation learning, RL fine-tuning) on **one workstation**:

| Target | Value |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | 1x RTX 4090, 24 GB, Ada (compute capability 8.9) |
| Driver | 595 (you reported it) |
| Already installed | **Isaac Sim 6.2.0 + Isaac Lab 3.0.0 (release/3.0.0).** The kit never reinstalls or changes them. |

> **Status: not run on a GPU yet.** All scripts were written from the official repos and docs as of 2026-10-29
> and pass a `bash -n` syntax check. None of them has run on a GPU machine. Run them in order, and treat the
> first run of each one as the real test. Items marked **(unverified)** are our best reading of the sources, not measured facts.

---

## 1. Architecture

```
                         ┌───────────────────────────── RTX 4090 (24 GB) ─────────────────────────────┐
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

## 2. What fits on a 24 GB RTX 4090

The numbers come from each project's own README or docs. "—" means the project does not publish the figure.

| Method (review) | Framework / env | Inference VRAM | Fine-tune mode that fits 24 GB | Official sim benchmarks |
|---|---|---|---|---|
| **π0 / π0-FAST / π0.5** | openpi (JAX) | **> 8 GB** (README; example GPU RTX 4090) | **LoRA: > 22.5 GB** (README lists RTX 4090), so it fits but tightly. Full FT needs > 70 GB and does **not fit**. Configs: `pi0_libero_low_mem_finetune`, `pi0_fast_libero_low_mem_finetune`. openpi-PyTorch has **no LoRA**. | LIBERO (π0.5: 98.8/98.2/98.0/92.4), ALOHA-sim |
| π0 / π0.5 (PyTorch port) | LeRobot | — (unverified, expect similar to openpi) | Guide says "24 GB tight at BS 1" (24–40 GB group). Frozen VLM (`train_expert_only`) plus grad checkpointing at BS 1–4 is **unverified**. | LIBERO (LeRobot π0.5: 97.0/99.0/98.0/96.0) |
| **OpenVLA-OFT** (7B) | openvla-oft (conda) | **~16 GB** LIBERO (README) | Training at BS 1 needs **~25 GB** (LIBERO.md), so it does **not fit** per upstream numbers. Upstream offers 8/4-bit loading for inference only. | LIBERO 4 suites (97.1% avg), ALOHA (real) |
| **SmolVLA** (450M) | LeRobot | — (small; unverified) | **Full fine-tune fits.** About 10–16 GB at BS 8 (LeRobot HW guide). | LIBERO (`lerobot/smolvla_libero` released). The paper also reports Meta-World (unverified here). |
| **GR00T N1.7** (3B) | Isaac-GR00T (uv) | **16 GB+** (README lists RTX 4090) | Official minimum is 40 GB+. Default FT (projector + DiT) peaks around 35 GB, so it does **not fit** per upstream. A small-batch attempt is unverified. | LIBERO (97.65/97.5/98.45/94.35), SimplerEnv Bridge and Fractal, RoboCasa, RoboLab |
| **Diffusion Policy** | LeRobot / ManiSkill baseline | small | **Full training fits.** About 8–14 GB at BS 8 (LeRobot); ~2–4 h for 5 epochs on a 4090 (indicative). | PushT (LeRobot's DP reference env); ManiSkill3 DP baseline; any LeRobot sim env (e.g. LIBERO) |
| **ACT** | LeRobot | small | **Full training fits.** About 2–6 GB at BS 8; ~30–60 min (indicative). | ALOHA-sim (LeRobot's ACT reference env); any LeRobot sim env |
| **SimpleVLA-RL** (OFT + veRL) | conda | ~16 GB (OFT) | RL tested on **8× A800 80 GB**, so it does **not fit**. Eval-only (`val_only`) might work (unverified). | LIBERO, RoboTwin 1.0/2.0 |
| **RLinf** (VLA RL) | uv or Docker | model-dependent | Single 24 GB GPU **not documented** (unverified). Supports OFT, π0/π0.5, GR00T on ManiSkill, LIBERO and Isaac Lab. | ManiSkill3, LIBERO, LIBERO-Pro/Plus, Isaac Lab |
| PPO (state / RGB) | ManiSkill3 | — | **Fits.** README says its RGB-PPO settings use < 15 GB. | ManiSkill3 tasks |
| RL (rsl_rl / skrl / sb3) | **Isaac Lab 3.0** (existing) | — | **Fits.** VRAM scales with `num_envs` and cameras. Isaac Sim minimum is 16 GB VRAM. | Isaac-Lift/Reach/Open-Drawer-Franka, KukaAllegro lift/reorient, … |
| Mimic + robomimic BC | **Isaac Lab 3.0** (existing) | — | **Fits** (state-based BC-RNN is small; unverified figure). | Franka Stack-Cube (+ visuomotor), GR1T2, G1 |

What this means in practice: on this GPU you can **evaluate** every released checkpoint above, **LoRA-fine-tune π0 / π0-FAST**
(openpi), **fully fine-tune SmolVLA**, train **ACT and Diffusion Policy** from scratch, and run **ManiSkill and Isaac Lab RL**.
Full fine-tuning of 3B–7B VLAs (OFT, GR00T, π0.5 full) and SimpleVLA-RL training need a larger GPU or a cloud job.

## 3. Step-by-step setup

```bash
# clone and enter this repository, then:
# git clone <this-repo-url> && cd <repo>
bash scripts/00_check_system.sh          # read-only audit; exit code = number of FAIL items
bash scripts/01_base_deps.sh             # apt packages, uv, Miniforge (asks for sudo). Driver/Isaac untouched.
bash scripts/02_isaaclab_validate.sh     # finds your Isaac Lab 3.0 / Isaac Sim 6.2 and runs a headless smoke test
bash scripts/10_install_lerobot.sh       # phase 1 core
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
     See `docs/ISAAC_LAB.md` for a safer way to run it.

## 4. First validation run (recommended order)

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

## 5. Phased plan

| Phase | Goal | Scripts | Exit criteria |
|---|---|---|---|
| **1: Core** (days 1–3) | Toolchain, Isaac validated, LIBERO evaluation of 3 VLA families, IL training | 00, 01, 02, 10, 11, 12; `eval/lerobot_smolvla_libero`, `eval/openpi_pi05_libero`, `eval/openvla_oft_libero`; `train/lerobot_act_aloha`, `train/lerobot_diffusion_pusht` | π0.5 / OFT LIBERO numbers within a few points of the published ones; ACT/DP checkpoints with nonzero success |
| **2: More benchmarks** (week 2) | Robustness and real-to-sim, bimanual, GPU-parallel, Isaac IL | 13 (GR00T LIBERO + SimplerEnv), 14 (ManiSkill3), 15 (SimplerEnv), 16 (RoboTwin 2.0); `train/isaaclab_rl_franka`, `train/isaaclab_mimic_stack`; optional 19 (Arena) | GR00T LIBERO eval; SimplerEnv/RoboTwin rollouts; Isaac Lift-Franka policy trained; Mimic BC policy evaluated |
| **3: RL fine-tuning** (week 3+) | Post-training with RL, as in the review | `train/maniskill_ppo` (fits), Isaac Lab RL (fits), 17 + `rl/rlinf_maniskill_quickstart` (experimental on 24 GB), 18 (SimpleVLA-RL, eval/code only) | PPO baselines converge; RLinf runs at least a few iterations or you decide to rent multi-GPU |

For phase 3 on one 24 GB GPU, start with methods whose trainable part is small: ManiSkill or Isaac Lab PPO, and RLinf's
DSRL for π0 (a small SAC agent in the diffusion noise space; RLinf docs). RL on 7B VLAs (SimpleVLA-RL, RLinf + OFT) was validated
upstream on 8× 80 GB GPUs.

## 6. Headless rendering

| Simulator | Backend | Setting |
|---|---|---|
| LIBERO / robosuite / ALOHA / PushT (MuJoCo) | EGL (NVIDIA) | `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` (kit default). Fallbacks: `osmesa` (CPU, slow, needs `libosmesa6-dev`) or `glx` with a display or `xvfb-run`. |
| ManiSkill3, SimplerEnv, RoboTwin (SAPIEN) | Vulkan | needs `/usr/share/vulkan/icd.d/nvidia_icd.json` + `libvulkan1`; check with `vulkaninfo --summary`. In Docker add `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (RoboTwin docs). |
| Isaac Lab / Isaac Sim | headless by default (no `--viz`) | `OMNI_KIT_ACCEPT_EULA=YES` for first headless run; `--video` for recordings; `--viz kit` only with a display. |

## 7. Docker options (official)

See `docs/DOCKER.md`. In short: openpi `examples/libero/compose.yml` (openpi's recommended LIBERO path; needs rootless Docker and the
NVIDIA container toolkit), the GR00T `docker/README.md`, the RLinf image `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`,
`huggingface/lerobot-gpu:latest`, ManiSkill `maniskill/base` (last pushed 2025-03, so it is old), the Isaac Lab `docker/` folder, and Arena `docker/run_docker.sh`.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `EGL`/`GLFW` errors in LIBERO | `MUJOCO_GL=egl`; check `ldconfig -p \| grep EGL_nvidia`; fall back to `MUJOCO_GL=osmesa` or `glx` (openpi README suggests glx for EGL errors) |
| LIBERO prompts "custom dataset path? (Y/N)" | answer N, or pre-create `~/.libero/config.yaml`; install scripts already answer N once |
| SAPIEN `vk::...` / segfault | missing NVIDIA Vulkan ICD or `graphics` capability in Docker; run `vulkaninfo` |
| `uv sync` conflicts (openpi) | `rm -rf .venv && uv sync`; `uv self update` (openpi README) |
| JAX OOM in openpi LoRA | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (default in script); close other GPU processes (`nvidia-smi`); disable EMA |
| openpi CUDA errors | openpi says system CUDA libs can conflict, since wheels bring CUDA; try removing system CUDA from `LD_LIBRARY_PATH` |
| flash-attn build fails | needs nvcc 12.x matching torch; `CUDA_HOME=/usr/local/cuda-12.x`, `MAX_JOBS=4`; `pip cache remove flash_attn`; inference works without it (OFT) |
| GR00T `GatedRepoError` / 401 | request access to `nvidia/Cosmos-Reason2-2B`, `uv run hf auth login` |
| GR00T `CUDA_HOME is unset` | `export CUDA_HOME=/usr/local/cuda` (or run its `scripts/deployment/dgpu/install_deps.sh`) |
| GR00T torchcodec "We support versions 4, 5, 6 and 7" | FFmpeg 8 on PATH; Ubuntu 22.04's apt FFmpeg 4.4 is fine |
| π0/π0.5 `QUANTILES normalization mode requires q01 and q99` | use MEAN_STD mapping (already in script) or `lerobot-edit-dataset … recompute_stats` |
| LeRobot feature-key mismatch | `--rename_map` (scripts set it for SmolVLA/LIBERO) |
| OFT results far below paper | `--center_crop True`; evaluate on the same GPU type you trained on, or merge LoRA on the eval GPU (LIBERO.md) |
| Isaac Lab smoke test fails only with `physics=isaacsim_physx` | Isaac Sim wheel or binary not visible in the env; check `02_…/report.txt`; first launch caches shaders (slow) |
| Other envs break after using Isaac | stray `PYTHONPATH`/`LD_LIBRARY_PATH` from Isaac setup scripts; open a clean shell |
| Disk fills up | set `RSP_HOME`/`HF_HOME` to a large disk; OFT RLDS LIBERO ≈ 10 GB, `lerobot/libero` 1.9 GB vs `HuggingFaceVLA/libero` 69.9 GB |

## 9. Caveats and unverified items

* **Not executed on GPU.** Syntax-checked only (`bash -n`). Upstream `main` branches (openpi, OFT, SimplerEnv, RoboTwin, Arena) can change.
  The kit logs exact commits so you can pin them once something works.
* VRAM cells marked "—" or "unverified" have no official figure. Measure them with the scripts' peak-memory output.
* `lerobot/smolvla_libero` rename map (`image→camera1`, `image2→camera2`) was inferred from its `config.json`.
* LeRobot docs were read from `main` (0.6.2.dev) while the kit pins `v0.6.1`. If a flag is missing, set `LEROBOT_REF=main`.
* Isaac Lab: the task IDs, CLI subcommands, `physics=isaacsim_physx` and the Mimic commands come from the `release/3.0.0` source and docs.
  Whether every task accepts every physics preset is unverified (check `list_envs`). `python -m isaaclab` (python mode) is inferred from the CLI code.
* Disk-space thresholds in `00_check_system.sh` (150/300 GB) are our planning estimates, not official numbers.
* SimpleVLA-RL and RLinf feasibility on one 24 GB GPU is unverified. Upstream test rigs are 8× 80 GB.

Sources: `docs/SOURCES.md`. Isaac Lab workflows: `docs/ISAAC_LAB.md`.
