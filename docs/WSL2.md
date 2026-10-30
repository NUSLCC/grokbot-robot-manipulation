# WSL2 (Windows 11) + headless runs, and the three machines

This kit has **one codebase** for all three machines. A hardware profile (`scripts/lib/profiles/*.env`) is picked
automatically from the GPU name, and WSL2 is detected automatically. You don't need to pick a branch or machine.

| Machine | Profile (auto) | OS | GPU | Path the kit takes |
|---|---|---|---|---|
| lcc-TRX40 workstation | `dual3090` | native Ubuntu 22.04, driver 595 | 2x RTX 3090 24 GB (Ampere, sm_86) | native Linux, same torch/CUDA as PR #1 |
| second workstation | `rtx4090` | native Ubuntu 22.04, driver 595 | RTX 4090 24 GB (Ada, sm_89), 64 GB RAM | native Linux, same torch/CUDA as PR #1 |
| Windows laptop | `rtx5090_laptop_wsl` | **WSL2 Ubuntu 24.04**, Windows driver 596.36 | **RTX 5090 Laptop GPU 24 GB (Blackwell, sm_120)** | WSL2 + Blackwell: cu128 torch, rendering fallbacks |

How the profile is chosen: a GPU name containing `5090` and `Laptop` selects `rtx5090_laptop_wsl`, `4090` selects
`rtx4090` and `3090` selects `dual3090`. Anything else gets `generic`. A desktop RTX 5090 also gets `generic`, but its
Blackwell GPU is still detected from the compute capability. To override, set `PROFILE=rtx4090` (or another name).
Any variable you export (for example `TORCH_CUDA=cu129`) wins over the profile.
`bash scripts/00_check_system.sh` prints the profile it chose.

All three are 24 GB cards, so the README's "What fits on 24 GB" table applies unchanged. π0 LoRA stays **tight**,
because openpi says it needs more than 22.5 GB.

---

## 1. Fresh Windows 11 setup (laptop)

1. **NVIDIA driver on Windows only.** Install or keep the current Game Ready or Studio driver for Windows (you have
   596.36). WSL2 uses the Windows driver through `/usr/lib/wsl/lib`. **Never install a Linux NVIDIA driver
   (`nvidia-driver-*`, `cuda-drivers`, `cuda` metapackage) inside WSL.** `00_check_system.sh` fails loudly if one is installed.
2. **Install WSL2 + Ubuntu 24.04** in PowerShell (Admin):
   ```powershell
   wsl --install -d Ubuntu-24.04
   wsl --update
   ```
   Reboot if asked. Then open "Ubuntu 24.04" from the Start menu and create your Linux user.
3. **Give WSL enough memory.** By default WSL gets 50% of RAM. Create `%UserProfile%\.wslconfig` (for example
   `C:\Users\<you>\.wslconfig`) with:
   ```ini
   [wsl2]
   memory=48GB      # at most ~75% of the laptop's RAM (64 GB -> 48GB, 32 GB -> 24GB)
   swap=32GB
   processors=16    # leave a few cores for Windows
   ```
   Then run `wsl --shutdown` in PowerShell and reopen Ubuntu. On a laptop, keep `memory` sensible so Windows stays responsive.
4. **Laptop power.** Plug in the charger and set Windows *Settings > System > Power > Power mode* to **Best performance**.
   A laptop GPU has lower sustained clocks and throttles when it gets hot, so runs will be slower than on a desktop 24 GB card.
5. **Disk.** The Ubuntu disk is a virtual `ext4.vhdx` that grows on `C:`. Keep about 150 GB free on `C:` for the
   headless set. Everything lives on the Linux side under `~/robot-sim`.

## 2. Clone and run (inside Ubuntu, NOT under /mnt/c)

```bash
cd ~                                   # Linux filesystem. /mnt/c is 5-20x slower and breaks symlinks/permissions
git clone -b cursor/wsl2-ubuntu2404-headless https://github.com/NUSLCC/grokbot-robot-manipulation.git
cd grokbot-robot-manipulation
bash scripts/00_check_system.sh        # read-only: WSL, GPU (Blackwell), RAM/disk, rendering, Isaac detection
QUICK=1 bash scripts/run_wsl_headless.sh
# The runner's models are not gated (SmolVLA LIBERO from the Hub, pi0.5 LIBERO from openpi's public bucket),
# so no Hugging Face login is needed for it. For the gated ones later (pi0/pi0.5 in LeRobot, GR00T), see the README.
```

The runner asks for your sudo password once (apt packages), then runs every step without a GUI and writes:

```
runs/<timestamp>/SUMMARY.md          # the results table (open this)
runs/<timestamp>/NN_<step>.log       # full log per step
runs/<timestamp>/<step>.gpu.csv      # nvidia-smi samples every second (peak GPU memory)
runs/<timestamp>/runner.log          # everything the runner printed
```

**View the results from Windows** in Explorer or VS Code:
`\\wsl$\Ubuntu-24.04\home\<linux-user>\grokbot-robot-manipulation\runs\<timestamp>\SUMMARY.md`
(the newer form `\\wsl.localhost\Ubuntu-24.04\...` also works). Or run `explorer.exe runs` inside Ubuntu.

Useful options:

| Option | Effect |
|---|---|
| `QUICK=1` | short training/eval settings (pipeline check) |
| `SKIP_ISAAC=1` | skip all Isaac Sim / Isaac Lab steps |
| `ISAAC_PHYSX=0` | skip the extra Isaac Sim PhysX (Kit) smoke, which is the step most likely to fail on WSL2 |
| `RUN_MIMIC=1` | also run the Isaac Lab Mimic pipeline (needs the `mimic` extra in your Isaac env) |
| `ONLY_STEPS="act_aloha maniskill_ppo"` | rerun selected steps |
| `SKIP_INSTALL=1` | skip `base_deps` / `*_install` steps (already installed) |
| `STEP_TIMEOUT=14400` | per-step timeout in seconds |

Steps: `system_check`, `base_deps`, `isaac_validate` (Newton physics, Reach-Franka), `isaac_physx_smoke`,
`isaac_rl_franka`, `isaac_mimic` (optional), `lerobot_install`, `act_aloha`, `diffusion_pusht`, `smolvla_libero_eval`,
`openpi_install`, `openpi_pi05_libero_eval`, `maniskill_install`, `maniskill_ppo`. If one step fails, the rest still run.
A step whose install failed is marked SKIP with the reason, and the summary gives the log path and a hint for each failure.
Exit code = number of FAILed steps.

## 3. Isaac Sim 6.2 / Isaac Lab 3.0 on WSL2

* Your existing install is **detected and used as is**, the same way as on Ubuntu (`ISAACLAB_PATH`, `uv run --no-sync`,
  `isaaclab.sh`). The kit never reinstalls it or changes its torch. Isaac Lab 3.0 ships torch 2.11 cu130, which already has sm_120 kernels.
* Headless in Isaac Lab 3.0 means **`--viz none --livestream 0`**. The old `--headless` flag is **rejected** by Isaac Lab 3.0's
  launcher (checked in the release/3.0.0 source). The kit passes `ISAACLAB_HEADLESS_FLAGS` (default `--viz none --livestream 0`)
  to `isaaclab train/play` and the Mimic scripts.
* **NVIDIA does not officially support Isaac Sim on WSL2.** The kit prints that note and continues. Newton-physics RL
  (`isaac_validate`, `isaac_rl_franka`) is the most likely to work. Kit/PhysX (`isaac_physx_smoke`) needs RTX/Vulkan, which may fail on WSL2.

## 4. What works, what falls back, what is skipped (WSL2 + RTX 5090 Laptop)

**None of these rows has been run on WSL2 or a Blackwell GPU yet.** "Expected" comes from upstream docs and the kit's logic.

| Step / framework | Expected on WSL2 + Blackwell | Why |
|---|---|---|
| system check, base deps | works | apt names chosen per Ubuntu release (tested in a real Ubuntu 24.04 userland) |
| LeRobot ACT / Diffusion PushT training | works (GPU training) | torch cu128 wheels include sm_120; PushT renders with pygame (no GL) |
| ACT eval (ALOHA, MuJoCo) | **fallback**: EGL probe, then OSMesa | WSL2 has no NVIDIA EGL; OSMesa is CPU rendering, so evals run several times slower |
| SmolVLA / π0.5 LIBERO eval (MuJoCo) | **fallback**: EGL probe, then OSMesa | same as above; policy inference still on GPU |
| openpi π0.5 serving (JAX) | **unverified** | `jax[cuda12]==0.5.3` on sm_120 has not been tested; LIBERO client is CPU-only |
| ManiSkill3 PPO (state) | **fallback**: no-render state PPO | SAPIEN needs a Vulkan GPU device that WSL2 doesn't provide; PhysX GPU sim still runs on CUDA; no eval videos |
| ManiSkill3 PPO (rgb), SimplerEnv, RoboTwin | **skipped / not expected** | need SAPIEN rendering (Vulkan) |
| Isaac Lab RL (Newton) | try, not officially supported | see section 3 |
| Isaac Sim PhysX / Kit | try, likely fails | RTX/Vulkan on WSL2 |
| OpenVLA-OFT | **not expected** on Blackwell | pins torch 2.2.0 (no sm_120). `OFT_TORCH_OVERRIDE=1` swaps in torch 2.7.1+cu128 (experimental) |
| SimpleVLA-RL | **not expected** on Blackwell | torch 2.4.0 (no sm_120) |
| GR00T N1.7 | expected to work (unverified) | CUDA 12.8 stack |

## 5. Blackwell (RTX 5090 Laptop, sm_120)

* Every framework env needs **PyTorch built for CUDA ≥ 12.8 (torch ≥ 2.7, cu128 wheels)**. `TORCH_CUDA` (default `cu128`
  from the profile) picks the wheel index for LeRobot and ManiSkill. Both already use cu128 on the 3090 and 4090 boxes, so nothing changes there.
* Installers warn loudly when a framework pins a torch without sm_120 kernels (OFT 2.2, SimpleVLA-RL 2.4). The symptom at
  run time is `no kernel image is available for execution on the device`. The runner's hint points here.
* openpi: `OPENPI_TORCH_CU128=1 bash scripts/11_install_openpi.sh` installs torch 2.7.1+cu128 for its PyTorch models and
  makes the kit's openpi scripts use `uv run --no-sync` (otherwise uv would re-sync the locked torch back).
* flash-attn: needs a CUDA ≥ 12.8 toolkit to build for sm_120. Otherwise it is skipped on Blackwell. Optional:
  `INSTALL_CUDA_TOOLKIT=12-8 bash scripts/01_base_deps.sh` installs the toolkit only, from NVIDIA's **wsl-ubuntu** repo (no driver).
  The flash-attn version OFT pins (2.5.5) predates Blackwell, so a successful build isn't guaranteed.

## 6. Known issues on WSL2

* **No NVIDIA EGL or Vulkan ICD:** MuJoCo falls back to OSMesa and SAPIEN falls back to no-render or skips. `vulkaninfo --summary`
  usually shows only `llvmpipe` (CPU). Ubuntu 24.04's Mesa has no D3D12 ("dozen") Vulkan driver.
* **First Isaac launch is slow** (shader cache). Let it finish. Raise `SMOKE_TIMEOUT` if needed.
* **RAM:** WSL2 sees only what `.wslconfig` allows. Out-of-RAM kills show up as rc=137 / `Killed`, and the runner hints at `.wslconfig`.
* **Repo or RSP_HOME under `/mnt/c`:** very slow and breaks permissions. `00_check_system.sh` warns (repo) or fails (RSP_HOME).
* **Laptop throttling:** wall times vary with temperature and power mode. Compare runs only with the charger plugged in.
* **Two GPUs in Windows (iGPU + dGPU):** `nvidia-smi` lists only the NVIDIA GPU, which is what the kit uses.
