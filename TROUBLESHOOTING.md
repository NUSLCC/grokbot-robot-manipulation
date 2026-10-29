Troubleshooting

This page lists common issues and fixes observed while hardening the kit. See also docs/ISAAC_LAB.md and README sections.

- Driver and CUDA toolkits
  - Symptom: Torch/OFT/GR00T wheels mismatch CUDA; flash-attn build fails; CuRobo/pytorch3d builds fail.
  - Fix: Do not change the NVIDIA driver. When a CUDA toolkit is required for building, install a side-by-side CUDA 12 toolkit only (never the driver): INSTALL_CUDA_TOOLKIT=12-4 bash scripts/01_base_deps.sh. For that script only, set CUDA_HOME=/usr/local/cuda-12.4.
- Ubuntu 22.04 vs 24.04
  - Symptom: You’re on 24.04 and 00_check_system.sh warns; Isaac Sim 6.2 docs target 22.04.
  - Fix: It’s fine for this kit. Isaac Sim 6.2 is tested on 22.04; keep your existing Isaac install and run 02_isaaclab_validate.sh to verify.
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
