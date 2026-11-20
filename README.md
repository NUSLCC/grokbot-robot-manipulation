# Robot Manipulation Simulation Platform

Headless kit to install, train and evaluate recent robot-manipulation methods (VLAs, imitation learning, RL) on one workstation, with Isaac Lab 3.0, LeRobot, openpi and ManiSkill.

## Machines

The hardware profile is auto-selected from the GPU name (`PROFILE=...` to override). WSL2 Ubuntu 24.04 is supported but is not being tested for now; see [docs/WSL2.md](docs/WSL2.md).

| Machine | Profile | OS / driver | GPU | Notes |
|---|---|---|---|---|
| lcc-TRX40 | `dual3090` | native Ubuntu 22.04, driver 595 | 2× RTX 3090 24 GB (sm_86) | Faulty RAM (memtester bit flips, Nov 2026): results are not trustworthy until the RAM is fixed and the runs are repeated |
| lcc-4090 | `rtx4090` | native Ubuntu 22.04, driver 595 | RTX 4090 24 GB (sm_89) | Isaac Lab 3.0 upgraded and validated, Nov 4, 2026 |

## Validation status

Full write-up: [docs/VALIDATION.md](docs/VALIDATION.md). Docs site: `mkdocs build --strict` / `mkdocs serve`.

| Machine | Quick run (`QUICK=1`) | Full-length run |
|---|---|---|
| lcc-TRX40 (`dual3090`) | **14/14 PASS** (Oct 31 – Nov 1, 2026) | **10/10 PASS** (Nov 1, 2026; Mimic via rerun after fix `5d1f64e`) |
| lcc-4090 (`rtx4090`) | **14/14 PASS** (Nov 4, 2026) | **Done** (Nov 4, 2026) |

lcc-TRX40 has faulty RAM (memtester bit flips, Nov 2026), so those results are not trustworthy until the RAM is fixed and the work is rerun.

lcc-4090 full-length highlights (Nov 4, 2026):

- Isaac Lab 3.0 upgraded and validated
- Mimic **86%** (43/50)
- ManiSkill PushCube **100%**
- Isaac Lift mean reward **23.5**
- Diffusion PushT **42%** (50 eval episodes)
- ACT aloha **30%** (10 eval episodes)
- π0.5 LIBERO **97.0%** (400 episodes)
- SmolVLA LIBERO **83.0%** (400 episodes, re-plan every 10 steps, dedicated MuJoCo 3.3.2 env)

Quick-mode success rates only check that each pipeline works; they are not policy scores.

## Quick start

```bash
git clone <this-repo-url>
cd <repo>
bash scripts/00_check_system.sh
bash scripts/01_base_deps.sh
bash scripts/02_isaaclab_validate.sh
bash scripts/10_install_lerobot.sh
bash scripts/10b_install_lerobot_libero.sh   # LIBERO env (MuJoCo 3.3.2; >=3.4 breaks libero_spatial task 5)
QUICK=1 bash scripts/run_wsl_headless.sh     # then open runs/<timestamp>/SUMMARY.md
```

Or `QUICK=1 bash scripts/run_phase1.sh` for the phase-1 installs and checks (it also runs `10b`). More install and validation commands: [docs/SETUP.md](docs/SETUP.md).

## Docs

| Page | What's there |
|---|---|
| [docs/VALIDATION.md](docs/VALIDATION.md) | Runs, settings, results, issues and how to rerun |
| [docs/SETUP.md](docs/SETUP.md) | Architecture, install list, pins, CUDA conflicts, first run, phases, headless rendering, caveats |
| [docs/MODELS.md](docs/MODELS.md) | What fits on a 24 GB GPU |
| [docs/WSL2.md](docs/WSL2.md) | WSL2 support (not currently being tested) |
| [docs/ISAAC_LAB.md](docs/ISAAC_LAB.md) | Isaac Lab 3.0 / Isaac Sim 6.2 workflows |
| [docs/DOCKER.md](docs/DOCKER.md) | Official Docker options |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common failures and fixes |
| [docs/SOURCES.md](docs/SOURCES.md) | Upstream sources |
