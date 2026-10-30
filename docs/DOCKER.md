# Official Docker options (verified 2026-10-29)

Prerequisites: Docker Engine (not snap Docker or Docker Desktop, since both break the NVIDIA runtime per openpi docs) and the NVIDIA Container Toolkit.
openpi ships `scripts/docker/install_docker_ubuntu22.sh` and `scripts/docker/install_nvidia_container_toolkit.sh`. openpi asks for rootless Docker.

| Project | Image / file | Command |
|---|---|---|
| openpi LIBERO (openpi's recommended path) | `examples/libero/compose.yml` (built locally) | `sudo xhost +local:docker; SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build` (prefix `MUJOCO_GL=glx` on EGL errors) |
| openpi generic | `scripts/docker/compose.yml` | `docker compose -f scripts/docker/compose.yml up --build` |
| Isaac-GR00T | `docker/README.md` in the repo | start the image, then clone/pull the repo inside it (upstream advice) |
| RLinf | `rlinf/rlinf:agentic-rlinf0.4-maniskill_libero` | `docker run -it --gpus all --shm-size 100g --net=host -e NVIDIA_DRIVER_CAPABILITIES=all <image>`; inside: `source switch_env openvla-oft` (or `RLINF_USE_DOCKER=1 bash scripts/17_install_rlinf.sh`) |
| LeRobot | `huggingface/lerobot-gpu:latest` (Docker Hub, updated 2026-10-29) | `docker run --gpus all -it huggingface/lerobot-gpu:latest` (entrypoint/details unverified) |
| ManiSkill3 | `maniskill/base` (last pushed 2025-03-06, so it is old) | prefer pip; if used, add `-e NVIDIA_DRIVER_CAPABILITIES=all` for Vulkan |
| RoboTwin 2.0 | no official image; Docker notes in the install doc | `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` is required (Vulkan) |
| Isaac Lab | `docker/` folder in the checkout (container install method in the docs) | you already have a native install, so it is not needed |
| Isaac Lab-Arena | `./docker/run_docker.sh` (`-g` adds GR00T deps) | mounts `$HOME/datasets`, `$HOME/models`, `$HOME/eval` |
