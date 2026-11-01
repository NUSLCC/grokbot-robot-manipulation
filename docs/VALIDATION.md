# Validation: what was tested, how, and the results

This page documents the kit's validation run: the headless runner, the 14 steps it runs, the exact
quick and full-length settings (taken from the scripts), the results measured so far on the dual RTX 3090
workstation, the problems found and fixed on the way, and how to rerun everything.

!!! summary "Status (Nov 1, 2026)"
    * **Quick run (`QUICK=1`) on the dual RTX 3090 workstation: all 14 steps PASS** (Oct 31 – Nov 1, 2026).
      Quick-mode success rates are pipeline checks, not meaningful policy scores.
    * **Full-length run on the dual RTX 3090 workstation (Nov 1, 2026): all 10 steps PASS.** `isaac_mimic` first
      failed on robomimic's "overwrite?" prompt; the fix (`5d1f64e`) passed on rerun `20261101-221416` (88%, 44/50).
      See [Full-length run](#full-length-run-dual-rtx-3090).
    * RTX 4090 workstation and RTX 5090 Laptop (WSL2): not validated yet.

## Contents

1. [The runner](#the-runner)
2. [The 14 steps](#the-14-steps)
3. [Results: quick run, dual RTX 3090](#quick-run-dual-rtx-3090)
4. [Full-length run, dual RTX 3090](#full-length-run-dual-rtx-3090)
5. [Issues found and fixed during validation](#issues-found-and-fixed-during-validation)
6. [How to run](#how-to-run)
7. [Machine profiles](#machine-profiles)
8. [Docs site](#docs-site)

---

## The runner

[`scripts/run_wsl_headless.sh`](../scripts/run_wsl_headless.sh) is one curated, fully headless validation run.
It was made for WSL2 Ubuntu 24.04 and also runs on native Ubuntu 22.04 / 24.04. It calls the kit's own install,
train and eval scripts in a fixed order; a failing step never stops the others.

```bash
QUICK=1 bash scripts/run_wsl_headless.sh   # short settings for every step (pipeline check)
bash scripts/run_wsl_headless.sh           # full-length default training / eval settings
```

### Headless guarantees

Nothing opens a window and no step can wait for keyboard input:

| What | How (from the code) |
|---|---|
| No display | the runner does `unset DISPLAY WAYLAND_DISPLAY` |
| Toolkits without a window | `SDL_VIDEODRIVER=dummy` (pygame / PushT), `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg` |
| Isaac Sim / Isaac Lab | `LIVESTREAM=0`; Isaac Lab 3.0 launches get `--viz none --livestream 0` (`ISAACLAB_HEADLESS_FLAGS`; 3.0 rejects the old `--headless`) |
| MuJoCo (LIBERO, ALOHA) | `MUJOCO_GL=egl`, probed per Python env; falls back to `osmesa` (CPU) if EGL does not work, e.g. on WSL2 |
| SAPIEN (ManiSkill) | Vulkan is probed; if it is not available, state-based PPO runs with rendering disabled |
| Trackers / progress bars | `WANDB_MODE=offline` (unless set), `HF_HUB_DISABLE_PROGRESS_BARS=1`, `TQDM_MININTERVAL=30` |
| No interactive prompts | every step runs with `</dev/null`; LIBERO's first-run question is answered by writing its config file ([`scripts/lib/libero_config.py`](../scripts/lib/libero_config.py)) |
| sudo | asked once up front, only if `base_deps` will run (not with `SKIP_INSTALL=1`, `SKIP_APT=1` or when `base_deps` is excluded) |

### Outputs

Each run writes to `runs/<YYYYMMDD-HHMMSS>/` (or `$RUNS_DIR/<timestamp>/`):

| File | Content |
|---|---|
| `runner.log` | the whole console output |
| `NN_<step>.log` | full log of each step (`06_isaac_mimic.log`, ...) |
| `<step>.gpu.csv` | `nvidia-smi --query-gpu=index,memory.used -l 1` samples while the step ran |
| `SUMMARY.md` | platform, profile, GPU, mode, PASS/FAIL/SKIP counts, the results table, and for every failure its log path, hint and last 40 log lines |

The console ends with the same table: step, `PASS`/`FAIL`/`SKIP`, wall time, peak GPU MB, key metric, plus the
log path and a one-line hint for each failure ([`scripts/lib/results.sh`](../scripts/lib/results.sh)).

* **Peak GPU MB** is the highest `memory.used` on *any* GPU while the step ran (the busiest card, whole-card usage,
  sampled every second). It includes other processes on that card.
* **SKIP** reasons: not selected (`ONLY_STEPS`, `SKIP_STEPS`, `SKIP_INSTALL`, `SKIP_ISAAC`, optional Mimic), a step
  that exits with code 77 (skipped on purpose, e.g. no Vulkan for RGB PPO), or a dependency that FAILED / could not run
  (`needs X (FAILED)`). A dependency skipped by your choice is assumed to be done already, so its dependents still run.
* **Exit code** = number of FAILed steps (0 = everything that ran passed).

## The 14 steps

Settings are the exact values in the code: the runner's `QUICK=1` overrides
(`Q_*` arrays in [`run_wsl_headless.sh`](../scripts/run_wsl_headless.sh)) and, for the full-length run, the
defaults of each script (the runner passes only `PLAY=0` for RL and `SMOKE_TASKS=Isaac-Reach-Franka` for the
Isaac checks).

| # | Step | What it tests | Tool / model / data / sim | `QUICK=1` | Full length (defaults) | Metric | Needs |
|---|---|---|---|---|---|---|---|
| 01 | `system_check` | driver, GPUs, CUDA, RAM, disk, rendering, tools, Isaac (read-only) | [`00_check_system.sh`](../scripts/00_check_system.sh) | same | same | `FAIL: n WARN: n` | – |
| 02 | `base_deps` | system packages, uv, Miniforge; repairs an interrupted dpkg | [`01_base_deps.sh`](../scripts/01_base_deps.sh) (apt) | same | same | `installed` | – |
| 03 | `isaac_validate` | Isaac Lab detection, Python 3.12, torch/CUDA in the Isaac env, Newton/Warp pins, `list_envs`, headless rsl_rl smoke on **Newton** | [`02_isaaclab_validate.sh`](../scripts/02_isaaclab_validate.sh), Isaac Lab 3.0.0 + Isaac Sim 6.2.0 (existing, never changed) | `Isaac-Reach-Franka`, 32 envs, 2 iterations | `Isaac-Reach-Franka`, 64 envs, 3 iterations | `checks PASS=n FAIL=n` | – |
| 04 | `isaac_physx_smoke` | the same smoke with `physics=isaacsim_physx` (exercises Isaac Sim / Kit) | same script, `SMOKE_PHYSICS=isaacsim_physx` | 32 envs, 2 iterations | 64 envs, 3 iterations | `checks PASS=n FAIL=n` | isaac_validate |
| 05 | `isaac_rl_franka` | Isaac Lab RL training (rsl_rl PPO), headless | [`train/isaaclab_rl_franka.sh`](../scripts/train/isaaclab_rl_franka.sh) | `Isaac-Reach-Franka`, 512 envs, 30 iterations, no play | `Isaac-Lift-Franka`, 1024 envs, 1500 iterations, no play | mean reward, last iteration | isaac_validate |
| 06 | `isaac_mimic` (optional, `RUN_MIMIC=1`) | Isaac Lab Mimic: annotate subtasks, generate synthetic demos, train robomimic BC, play rollouts | [`train/isaaclab_mimic_stack.sh`](../scripts/train/isaaclab_mimic_stack.sh); NVIDIA's Franka cube-stack human dataset (Isaac 6.2 assets); `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` / `IsaacContrib-Stack-Cube-Franka-IK-Rel` | 20 generated demos (10 envs), 20 epochs, 5 rollouts | 1000 generated demos (100 envs), 1000 epochs, 50 rollouts | `play success x% (k/n rollouts)` | isaac_validate |
| 07 | `lerobot_install` | LeRobot env + extras; checks gym_aloha and gym_pusht import, build and render headless | [`10_install_lerobot.sh`](../scripts/10_install_lerobot.sh), LeRobot **v0.6.1** (Python 3.12, torch cu128) | same | same | `installed` | base_deps |
| 08 | `act_aloha` | train ACT from scratch, then sim eval | `lerobot-train`, dataset `lerobot/aloha_sim_transfer_cube_human`, sim `AlohaTransferCube-v0` (MuJoCo) | 2000 steps, batch 8, eval at 2000, 5 eval episodes | 20000 steps, batch 8, eval/save every 5000, 10 eval episodes | loss, eval success % | lerobot_install |
| 09 | `diffusion_pusht` | train Diffusion Policy from scratch, then sim eval | `lerobot-train`, dataset `lerobot/pusht`, sim PushT (pygame) | 2000 steps, **batch 32**, eval at 2000, 10 eval episodes | 50000 steps, **batch 64**, eval/save every 10000, 50 eval episodes | loss, eval success % | lerobot_install |
| 10 | `smolvla_libero_eval` | evaluate a released VLA on LIBERO | `lerobot-eval`, model `lerobot/smolvla_libero`, LIBERO (MuJoCo) | `libero_object`, `N_EPISODES=1` | `libero_object`, `N_EPISODES=2` | success % (`eval_info.json`) | lerobot_install |
| 11 | `openpi_install` | openpi (JAX + PyTorch) and its Python 3.8 LIBERO client | [`11_install_openpi.sh`](../scripts/11_install_openpi.sh), openpi `main` (commit recorded) | same | same | `installed` | base_deps |
| 12 | `openpi_pi05_libero_eval` | reproduce π0.5 on LIBERO with the official server/client | openpi `serve_policy.py --env LIBERO` (`gs://openpi-assets/checkpoints/pi05_libero`) + `examples/libero/main.py` | `libero_spatial`, 1 trial per task (10 episodes) | `libero_spatial`, 5 trials per task (50 episodes) | `success x% of n eps` | openpi_install |
| 13 | `maniskill_install` | ManiSkill3 env, SAPIEN | [`14_install_maniskill.sh`](../scripts/14_install_maniskill.sh), ManiSkill **3.0.1** (Python 3.11) | same | same | `installed` | base_deps |
| 14 | `maniskill_ppo` | GPU-parallel PPO baseline (state observations) | ManiSkill `examples/baselines/ppo/ppo.py`, `PushCube-v1` | 500,000 steps, 1024 envs | 2,000,000 steps, 2048 envs | eval `success_once`, return | maniskill_install |

Notes:

* The runner's Isaac checks run `Isaac-Reach-Franka` only. Run `bash scripts/02_isaaclab_validate.sh` directly for its
  default set: `Isaac-Reach-Franka` and `Isaac-Lift-Franka`, each on Newton and on Isaac Sim PhysX.
* `isaac_mimic` also always runs a small first generation pass (20 envs, 10 trials) before the main one.
* ACT and Diffusion save their checkpoint before the eval, so `EVAL_ONLY=1` can re-evaluate without retraining.
* LeRobot sim evals run in-process (`--eval.use_async_envs=false`), see [issue 7](#issues-found-and-fixed-during-validation).

## Quick run, dual RTX 3090

**Machine:** lcc-TRX40, native Ubuntu 22.04.5, 2× RTX 3090 24 GB (sm_86), NVIDIA driver 595, profile `dual3090`.
Isaac Sim 6.2.0 + Isaac Lab 3.0.0 (uv mode, `~/IsaacLab/.venv`, Python 3.12.15, torch 2.11.0+cu130).
**Mode:** `QUICK=1`, runs between **Oct 31 and Nov 1, 2026** (several reruns with `ONLY_STEPS` after fixes;
each row is the latest passing run).

| Step | Result | Key metric | Peak GPU | Notes |
|---|---|---|---|---|
| `system_check` | PASS | – | – | |
| `base_deps` | PASS | installed | – | after the dpkg repair ([issue 1](#issues-found-and-fixed-during-validation)) |
| `isaac_validate` | PASS | all checks pass | < 6 GB | 4 smoke tests (Reach + Lift × Newton + PhysX, direct `02_isaaclab_validate.sh` run), after Newton was downgraded to 1.6.1 ([issue 4](#issues-found-and-fixed-during-validation)) |
| `isaac_physx_smoke` | PASS | all checks pass | < 6 GB | Isaac Sim PhysX backend |
| `isaac_rl_franka` | PASS | mean reward −2.69 at iteration 29; rerun −2.76 | 1485 MiB | Reach, 30 iterations, 26 s (rerun after the Mimic install) |
| `isaac_mimic` | PASS | play success 0.0% (0/5 rollouts) | 2668 MiB (play), 4362 MiB (whole pipeline) | quick pipeline: annotate, generate, 20 training epochs (6 min 8 s, then the play stage failed: [issue 11](#issues-found-and-fixed-during-validation)); play-only rerun 147 s |
| `lerobot_install` | PASS | installed | – | LeRobot v0.6.1 |
| `act_aloha` | PASS | eval success 20.0% | 2653 MiB (eval) | 2000 training steps; eval-only rerun 44 s |
| `diffusion_pusht` | PASS | eval success 0.0% | 9236 MiB (eval); ~22 GB during training | 2000 steps at batch 32; eval-only rerun 56 s |
| `smolvla_libero_eval` | PASS | success 90% | 3.3 GB | |
| `openpi_install` | PASS | installed | – | |
| `openpi_pi05_libero_eval` | PASS | success 90% of 10 episodes | 21.7 GB | |
| `maniskill_install` | PASS | installed | – | |
| `maniskill_ppo` | PASS | eval success_once 0.875 | – | |

!!! warning "Quick-mode scores are not policy scores"
    `QUICK=1` trains for a tiny fraction of the normal length (ACT 2000 of 20000 steps, Diffusion 2000 of 50000,
    Mimic 20 of 1000 epochs, ManiSkill 0.5M of 2M steps) and evaluates only a few episodes. It checks that every
    pipeline installs, trains, saves, reloads and evaluates headless. Low numbers such as ACT 20%, Diffusion 0% and
    Mimic 0/5 are expected there. The pretrained-model evals (SmolVLA, π0.5) use few episodes, so their 90% has a wide
    error margin. Use the full-length run for real numbers.

"–" means the value was not recorded in the reports we have. Wall times other than those given were not recorded.

## Full-length run, dual RTX 3090

**Run `20261101-121944` (Nov 1, 2026, `QUICK=0`, `SKIP_INSTALL=1`), split across the two GPUs:** Isaac steps on
GPU 0 (`runs/gpu0`), LeRobot / openpi / ManiSkill on GPU 1 (`runs/gpu1`). **10/10 PASS** (`isaac_mimic` from Mimic-only rerun `20261101-221416`, GPU 0, after fix `5d1f64e`).
Numbers are copied from the two `SUMMARY.md` files.

| Step | Settings (full length) | GPU | Result | Wall time | Peak GPU | Key metric |
|---|---|---|---|---|---|---|
| `system_check` | – | 0 | PASS | 1 s | 1072 MiB | FAIL: 0, WARN: 2 |
| `isaac_validate` | Reach, 64 envs, 3 iterations (Newton) | 0 | PASS | 21 s | 1482 MiB | checks PASS=11 FAIL=0 |
| `isaac_physx_smoke` | Reach, 64 envs, 3 iterations (PhysX) | 0 | PASS | 17 s | 3516 MiB | checks PASS=11 FAIL=0 |
| `isaac_rl_franka` | Lift, 1024 envs, 1500 iterations | 0 | PASS | 19 m 58 s | 3726 MiB | mean reward 20.37, iter 1499 |
| `isaac_mimic` | 1000 demos, 1000 epochs, 50 rollouts | 0 | PASS (rerun `20261101-221416`; first run FAILED, [issue 13](#issues-found-and-fixed-during-validation)) | 29 m 05 s | 2769 MiB (GPU 0 only) | play success 88.0% (44/50 rollouts) |
| `act_aloha` | 20000 steps, batch 8, 10 eval episodes | 1 | PASS | 27 m 04 s | 5002 MiB | loss 0.109, eval success 40.0% |
| `diffusion_pusht` | 50000 steps, batch 64, 50 eval episodes | 1 | PASS | 1 h 57 m | 20929 MiB | loss 0.003, eval success 46.0% |
| `smolvla_libero_eval` | libero_object, `N_EPISODES=2` | 1 | PASS | 2 m 05 s | 1779 MiB | success 85.0% |
| `openpi_pi05_libero_eval` | libero_spatial, 5 trials per task (50 episodes) | 1 | PASS | 6 m 07 s | 19461 MiB | success 100.0% of 50 episodes |
| `maniskill_ppo` | PushCube-v1, 2M steps, 2048 envs | 1 | PASS | 2 m 50 s | 18409 MiB | eval success_once 1.0, return 37.92 |

* **Diffusion at batch 64 fit on one 24 GB RTX 3090** (passed; peak ≤ 20.9 GB); `BATCH=32/16` was not needed.
* **Peak GPU caveat:** in this run the sampler took the highest `memory.used` on *either* GPU, so a row can include
  the other lane's memory (e.g. the failed Mimic run showed 17983 MiB while Diffusion ran on GPU 1; the GPU-0-only rerun peaked at 2769 MiB). Since `6d43958`, a split run counts
  only its own `CUDA_VISIBLE_DEVICES` GPU. Per-GPU samples of this run are in `runs/gpu*/*/<step>.gpu.csv`.
* Install steps (`base_deps`, `*_install`) were skipped with `SKIP_INSTALL=1` because they had already passed.

## Issues found and fixed during validation

Every issue below came from the user's real runs on the dual 3090 workstation; each fix has a test with a
fixture made from the real log (`tests/`). Commits are on the PR #2 branch `cursor/wsl2-ubuntu2404-headless`.

| # | Problem | Cause | Fix | Commit |
|---|---|---|---|---|
| 1 | `base_deps` failed: `dpkg was interrupted` | an earlier apt/dpkg run had been interrupted | `01_base_deps.sh` detects it and runs `sudo dpkg --configure -a` once, logged (`RSP_AUTO_DPKG_CONFIGURE=0` only reports it); third-party apt sources failing `apt-get update` (e.g. `NO_PUBKEY` for a vendor repo) are a one-line note, never fatal | `21e154d` |
| 2 | WSL2 and Vulkan hints on native Linux | hints matched package names in apt lines (e.g. `vulkan-tools`) and used WSL wording everywhere | hints ignore apt package lists; WSL2 wording only on WSL2 | `21e154d` |
| 3 | steps "FAIL in 0 s" when a dependency had failed | dependency failures were reported as failures | they are `SKIP` with `needs X (FAILED)`; user-chosen skips still let dependents run | `21e154d` |
| 4 | Isaac smoke: `AttributeError: 'ModelBuilder' object has no attribute '_cloth_label'` | the Isaac Lab venv had newton 1.8.0.dev0 (from git); Isaac Lab v3.0.0 pins `newton[sim]==1.6.1`, `warp-lang==1.17.0`, `mujoco~=3.12.0`, `mujoco-warp~=3.12.0`, `newton-usd-schemas>=0.5.0` | detect-only "Newton/Warp pins" row in `isaac_validate` printing the exact `uv pip install --python ~/IsaacLab/.venv/bin/python ...` command; the user ran it (newton 1.6.1, mujoco 3.12.0) | `fe02270` |
| 5 | `lerobot-train: error: unrecognized arguments: --eval_freq` | LeRobot v0.6.0 renamed it to `--env_eval_freq` (and `main` later replaced `--wandb.*` with `--trackio.*`) | train scripts read `lerobot-train --help` and use whichever spelling the installed version accepts | `dd79231` |
| 6 | LIBERO `EOFError: EOF when reading a line` | LIBERO asks an interactive question when its `config.yaml` is missing | `scripts/lib/libero_config.py` writes a per-env config and sets `LIBERO_CONFIG_PATH` | `dd79231` |
| 7 | ACT/Diffusion eval: `NamespaceNotFound: Namespace gym_aloha / gym_pusht not found` after training | LeRobot 0.6.1 async eval workers (multiprocessing forkserver) never import the sim packages (huggingface/lerobot#4931); the packages were installed | evals run in-process (`--eval.use_async_envs=false`; `LEROBOT_ASYNC_EVAL=1` opts out), sims are probed before training, `EVAL_ONLY=1` re-evaluates a saved checkpoint | `cfdca2d` |
| 8 | Diffusion used ~22 GB during quick training | mostly PyTorch's allocator cache and cuDNN autotune workspaces, not a hard requirement | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (if unset); `BATCH=16` documented for shared 24 GB cards. Effect not measured on a GPU yet | `cfdca2d` |
| 9 | Mimic: `isaaclab_mimic` not importable | the Isaac Lab `mimic` extra was not installed (the kit never changes the Isaac env by itself) | user installed it: `cd ~/IsaacLab && source .venv/bin/activate && ./isaaclab.sh -i mimic; deactivate`; Newton stayed 1.6.1, `isaac_rl_franka` re-passed | user action (no commit) |
| 10 | `./isaaclab.sh -i mimic`: `ModuleNotFoundError: No module named 'tomllib'` | without an active venv, `isaaclab.sh` used Ubuntu 22.04's system Python 3.10 (`tomllib` exists from 3.11) | activate the Isaac Lab venv first (command above) | user action (no commit) |
| 11 | Mimic play: `[FAIL] No robomimic checkpoint found` after training succeeded | robomimic saves to `logs/robomimic/<task>/<experiment>/<timestamp>/models/*.pth`, one level deeper than the old glob | newest `*/models/*.pth` at any depth (task folder first); `MIMIC_STAGE=play` reruns only the play stage; play.py's success rate is the step metric | `7ba5a02` |
| 12 | wrong hint for issue 11: "OpenGL/EGL problem: try MUJOCO_GL=osmesa" | the Kit warning `Failed to startup plugin carb.windowing-glfw.plugin (interfaces: [carb::windowing::IGLContext ...])` (8× in the log, normal without a display) matched `GLContext` | hint matching ignores benign Kit `[Warning]` lines and egl-probe install lines; the EGL hint only looks at the end of the log; a specific checkpoint hint | `7ba5a02`, `012f02b` |
| 13 | full run, Mimic train: `EOFError` after `model directory (.../bc_rnn_low_dim_franka_stack) already exists! overwrite? (y/n)` | robomimic 0.4.0 `get_exp_dir()` asks with `input()` when the experiment folder exists (left by the quick run); Isaac Lab's `train.py` never passes `auto_remove_exp_dir` | each run trains into its own folder (`--name bc_rnn_low_dim_franka_stack-<timestamp>`); a reused `MIMIC_EXP_NAME` folder is moved to `<dir>.bak-<timestamp>`, never deleted; stdin is `/dev/null`; play uses this run's checkpoint; `train.py`'s swallowed `run failed with error` now fails the step; a rerun reuses `generated_dataset.hdf5` if it holds ≥ `N_GEN` demos (`REUSE_DATA=0` regenerates) | `5d1f64e` |

More detail for each: [TROUBLESHOOTING.md](../TROUBLESHOOTING.md).

## How to run

Run inside `tmux` (or `screen`) so a closed terminal or SSH session does not stop a long run:
`tmux new -s rsp`, detach with `Ctrl-b d`, reattach with `tmux attach -t rsp`.

### Quick (pipeline check)

```bash
cd ~/grokbot-robot-manipulation && git pull
QUICK=1 RUN_MIMIC=1 bash scripts/run_wsl_headless.sh
```

### Full length, one GPU

```bash
cd ~/grokbot-robot-manipulation && git pull
tmux new -s full
RUN_MIMIC=1 SKIP_INSTALL=1 bash scripts/run_wsl_headless.sh
```

`SKIP_INSTALL=1` skips `base_deps` and the `*_install` steps (they already passed) and avoids the sudo prompt.
The steps run one after another on the first visible GPU.

### Full length, two GPUs (dual 3090): two runner copies

Start two runners at the same time, each pinned to one card with `CUDA_VISIBLE_DEVICES` and with its own
`RUNS_DIR`, splitting the steps between them (Isaac on GPU 0, LeRobot / openpi / ManiSkill on GPU 1):

```bash
cd ~/grokbot-robot-manipulation && git pull
# terminal / tmux window 1
CUDA_VISIBLE_DEVICES=0 RUNS_DIR="$PWD/runs/gpu0" RUN_MIMIC=1 SKIP_INSTALL=1 \
  ONLY_STEPS="system_check isaac_validate isaac_physx_smoke isaac_rl_franka isaac_mimic" \
  bash scripts/run_wsl_headless.sh
# terminal / tmux window 2
CUDA_VISIBLE_DEVICES=1 RUNS_DIR="$PWD/runs/gpu1" SKIP_INSTALL=1 \
  ONLY_STEPS="act_aloha diffusion_pusht smolvla_libero_eval openpi_pi05_libero_eval maniskill_ppo" \
  bash scripts/run_wsl_headless.sh
```

Each copy writes its own `runs/gpu0/<timestamp>/SUMMARY.md` and `runs/gpu1/<timestamp>/SUMMARY.md`.

!!! note "GPU_MB with two runners"
    The `GPU_MB` column is the peak `memory.used` of the **busiest GPU on the machine** (`nvidia-smi` ignores
    `CUDA_VISIBLE_DEVICES`). With two runners in parallel, a step's number can therefore be the *other* card's
    usage. For per-card numbers, read the `index` column in `runs/gpuN/<timestamp>/<step>.gpu.csv`.

### Rerun options

| Option | Effect |
|---|---|
| `ONLY_STEPS="a b"` | run only these steps (steps not listed count as already done, so their dependents still run) |
| `SKIP_STEPS="a b"` | skip these steps |
| `SKIP_INSTALL=1` | skip `base_deps` and `*_install` |
| `EVAL_ONLY=1` (optional `CKPT=<.../pretrained_model>`) | `act_aloha` / `diffusion_pusht`: evaluate the newest saved checkpoint, no training |
| `MIMIC_STAGE=all\|annotate\|generate\|train\|play` | `isaac_mimic`: run one stage; `play` uses the newest robomimic checkpoint (`CHECKPOINT=` overrides) |
| `STEP_TIMEOUT=14400` | per-step timeout in seconds (default 0 = none); a timeout is rc 124 |
| `QUICK=1` | short settings |
| `SKIP_ISAAC=1`, `ISAAC_PHYSX=0`, `RUN_MIMIC=1` | Isaac steps off / PhysX smoke off / Mimic on |
| `RUNS_DIR=path` | where `runs/<timestamp>` goes |

Examples:

```bash
EVAL_ONLY=1 QUICK=1 ONLY_STEPS="act_aloha diffusion_pusht" bash scripts/run_wsl_headless.sh
RUN_MIMIC=1 QUICK=1 MIMIC_STAGE=play ONLY_STEPS=isaac_mimic bash scripts/run_wsl_headless.sh
```

### Known risks for the full-length run

* **Diffusion at batch 64 on a 24 GB card.** The quick run at batch 32 peaked at ~22 GB during training (mostly
  allocator cache, [issue 8](#issues-found-and-fixed-during-validation)); full length uses batch 64. If it runs out of
  memory, rerun that step directly with a smaller batch (the runner does not pass `BATCH` in full mode):
  ```bash
  CUDA_VISIBLE_DEVICES=1 BATCH=32 bash scripts/train/lerobot_diffusion_pusht.sh   # or BATCH=16
  ```
* **π0.5 eval** used 21.7 GB in the quick run; keep its card free of other jobs.
* **Durations are unmeasured.** Full-length ACT (20k steps), Diffusion (50k steps), Isaac Lift RL (1500 iterations) and
  Mimic (1000 generated demos, 1000 epochs) are long; set `STEP_TIMEOUT` only if you want a cap.

## Machine profiles

The profile is auto-selected from the GPU name ([`scripts/lib/profiles/`](../scripts/lib/profiles/dual3090.env); `PROFILE=<name>` overrides).

| Profile | Machine | Status |
|---|---|---|
| `dual3090` | lcc-TRX40: 2× RTX 3090 24 GB (sm_86), native Ubuntu 22.04, driver 595 | **Validated** (quick run, all 14 steps PASS); full-length run in progress |
| `rtx4090` | second workstation: RTX 4090 24 GB (sm_89), native Ubuntu 22.04, driver 595, 64 GB RAM | not validated yet |
| `rtx5090_laptop_wsl` | Windows laptop: RTX 5090 Laptop GPU 24 GB (Blackwell sm_120), WSL2 Ubuntu 24.04, Windows driver 596.36 | not validated yet (see [WSL2.md](WSL2.md)) |

## Docs site

The repo includes a minimal [MkDocs](https://www.mkdocs.org/) + Material config ([`mkdocs.yml`](../mkdocs.yml)) over
the README and the `docs/` pages. A small hook ([`mkdocs_hooks.py`](../mkdocs_hooks.py)) adds the root `README.md`
(home page) and `TROUBLESHOOTING.md`, and turns links to scripts into GitHub links, so the same Markdown works on
GitHub and on the site.

Build and preview locally:

```bash
python3 -m venv .venv-docs && .venv-docs/bin/pip install mkdocs-material
.venv-docs/bin/mkdocs build --strict   # output in site/ (git-ignored); fails on broken links/anchors
.venv-docs/bin/mkdocs serve            # http://127.0.0.1:8000
```

Publish to GitHub Pages (not done yet; GitHub Pages is not enabled for this repo):

```bash
.venv-docs/bin/mkdocs gh-deploy        # builds and pushes the site to the gh-pages branch
```

Then in GitHub: **Settings → Pages → Build and deployment → Deploy from a branch → `gh-pages` / root**. The site
appears at `https://nuslcc.github.io/grokbot-robot-manipulation/`. Rerun `mkdocs gh-deploy` to update it.
