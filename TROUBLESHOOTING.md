Troubleshooting

This page lists common issues and fixes observed while hardening the kit. See also docs/ISAAC_LAB.md and README sections.

- Driver and CUDA toolkits
  - Symptom: Torch/OFT/GR00T wheels mismatch CUDA; flash-attn build fails; CuRobo/pytorch3d builds fail.
  - Fix: Do not change the NVIDIA driver. When a CUDA toolkit is required for building, install a side-by-side CUDA 12 toolkit only (never the driver): INSTALL_CUDA_TOOLKIT=12-4 bash scripts/01_base_deps.sh. For that script only, set CUDA_HOME=/usr/local/cuda-12.4.
- Ubuntu 22.04 vs 24.04
  - Both are supported. 01_base_deps.sh picks apt names per release (24.04: libgl1 + libglx-mesa0 instead of libgl1-mesa-glx) and skips any name apt does not know. System Python on 24.04 is 3.12 and PEP 668 "externally managed": the kit never pip-installs into it, and every framework gets its own Python via uv or conda (3.8 / 3.10 / 3.11 / 3.12).
- PYTHONPATH/LD_LIBRARY_PATH leaks from Isaac scripts
  - Symptom: Other envs break after using Isaac; import conflicts, missing GL libs.
  - Fix: Ensure your shell does not source setup_conda_env.sh or setup_python_env.sh automatically. The kit strips these via clean_python_env(). Open a fresh shell.
- Headless EGL (MuJoCo) errors
  - Symptom: EGL/GLFW errors; LIBERO fails to render.
  - Fix: MUJOCO_GL=egl (default). Check ldconfig -p | grep EGL_nvidia. As a fallback, use MUJOCO_GL=osmesa (CPU) or MUJOCO_GL=glx with a display/xvfb.
- SAPIEN/Vulkan errors (ManiSkill3, SimplerEnv, RoboTwin)
  - Symptom: vk::... ErrorExtensionNotPresent or segfault on first make().
  - Fix: Install libvulkan1 and verify /usr/share/vulkan/icd.d/nvidia_icd.json exists (00_check_system.sh checks). In Docker, add -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics. For CI/cpu-only smoke, the kit only imports modules without launching a renderer.
- Gated Hugging Face models
  - Symptom: 401/403 pulling nvidia/Cosmos-Reason2-2B (GR00T) or google/paligemma-3b-pt-224 (pi0/pi0.5 tokens).
  - Fix: ~/robot-sim/envs/lerobot/bin/hf auth login and accept the gated repos. The README links the exact IDs.
- Isaac Sim 6.2 headless first-launch slowness
  - Symptom: 02_isaaclab_validate.sh headless isaacsim_physx run is slow or times out.
  - Fix: First launch compiles and caches shaders. Re-run; check that the isaacsim wheel or binary is visible in the env.
- FFmpeg vs torchcodec (GR00T)
  - Symptom: torchcodec “We support versions 4, 5, 6 and 7”.
  - Fix: Ubuntu 22.04 apt ffmpeg 4.4 is OK. Ubuntu 24.04 ships 6.x; the kit warns but continues.
- Disk space
  - Symptom: Downloads fail or the machine fills up.
  - Fix: Set RSP_HOME to a large disk; consider setting HF_HOME as well. DATA: OFT RLDS LIBERO ≈ 10 GB; lerobot/libero 1.9 GB vs HuggingFaceVLA/libero 69.9 GB.

WSL2 (Windows 11, Ubuntu 24.04) - see docs/WSL2.md

- WSL2: nvidia-smi fails or libcuda is missing
  - Symptom: 00_check_system.sh FAILs on nvidia-smi or /usr/lib/wsl/lib.
  - Fix: Install or update the NVIDIA driver on WINDOWS only, run `wsl --update` and `wsl --shutdown` in PowerShell, then reopen Ubuntu. /usr/lib/wsl/lib must contain libcuda.so*. Never install nvidia-driver-*, cuda-drivers or the `cuda` metapackage inside WSL; if you did, `sudo apt-get remove --purge` them and run `wsl --shutdown`.
- WSL2: Vulkan / SAPIEN (ManiSkill, SimplerEnv, RoboTwin)
  - Symptom: `vk::...ErrorExtensionNotPresent`, `ErrorIncompatibleDriver`, or "failed to find a rendering device"; `vulkaninfo --summary` shows only llvmpipe.
  - Why: WSL2 exposes CUDA (libcuda/libnvidia-ml under /usr/lib/wsl/lib) but no NVIDIA Vulkan ICD, and Ubuntu 24.04's Mesa ships no D3D12 Vulkan driver.
  - Fix: train/maniskill_ppo.sh uses MS_RENDER=auto. It falls back to state-only PPO with render_backend="none" (GPU PhysX still runs on CUDA; no eval videos). RGB PPO, SimplerEnv and RoboTwin evals need a native Linux machine.
- WSL2: MuJoCo EGL / OSMesa slowness
  - Symptom: LIBERO/ALOHA evals say "Using MUJOCO_GL=osmesa" and take much longer.
  - Why: WSL2 has no NVIDIA EGL. OSMesa (or Mesa llvmpipe EGL) renders on the CPU.
  - Fix: Expected. Keep N_EPISODES/TRIALS small (QUICK=1), and give WSL more processors in .wslconfig. Force a backend with MUJOCO_GL=osmesa|egl. RSP_WSL_RENDER_FALLBACK=0 disables the fallback.
- WSL2: Isaac Sim shader cache / first launch
  - Symptom: The first isaac_* step is very slow or hits SMOKE_TIMEOUT.
  - Fix: First launch compiles shaders into the cache on the Linux side (~/.cache/ov, ~/.nv). Rerun it. Isaac Sim on WSL2 is not officially supported by NVIDIA; Newton-physics RL is the most likely to work, Kit/PhysX may fail (ISAAC_PHYSX=0 skips it).
- WSL2: processes killed / rc=137 / "Killed"
  - Fix: Raise memory and swap in %UserProfile%\.wslconfig (example in docs/WSL2.md), run `wsl --shutdown`, reopen Ubuntu. On a laptop keep memory at most ~75% of physical RAM.
- WSL2: everything is very slow under /mnt/c
  - Fix: Clone into the Linux home (`cd ~`), not /mnt/c. 00_check_system.sh warns about the repo and FAILs if RSP_HOME is under /mnt. Open results from Windows via \\wsl$\Ubuntu-24.04\home\<user>\...
- Blackwell (RTX 5090): "no kernel image is available for execution on the device"
  - Why: That env's torch has no sm_120 kernels (needs torch >= 2.7 built for CUDA >= 12.8). Affects OpenVLA-OFT (torch 2.2) and SimpleVLA-RL (torch 2.4).
  - Fix: Use cu128 envs (LeRobot, ManiSkill, GR00T, Isaac Lab are fine). Experimental: OFT_TORCH_OVERRIDE=1 bash scripts/12_install_openvla_oft.sh; OPENPI_TORCH_CU128=1 bash scripts/11_install_openpi.sh. flash-attn on Blackwell needs a CUDA >= 12.8 toolkit (INSTALL_CUDA_TOOLKIT=12-8 bash scripts/01_base_deps.sh); otherwise it is skipped.
- Laptop GPU slower than expected
  - Fix: Plug in the charger, set Windows power mode to "Best performance", and keep the laptop cool. Sustained clocks on a laptop GPU are lower than on a desktop card.
- Isaac Lab 3.0: "unrecognized arguments: --headless"
  - Fix: Isaac Lab 3.0 removed --headless. Use `--viz none` (the kit passes ISAACLAB_HEADLESS_FLAGS="--viz none --livestream 0").
