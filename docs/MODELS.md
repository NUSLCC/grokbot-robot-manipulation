# What fits on a 24 GB GPU (RTX 3090 / 4090)

VRAM figures come from each project's own README or docs, and apply to a 24 GB card (RTX 3090 or RTX 4090). "—" means the project does not publish the figure.

| Method (review) | Framework / env | Inference VRAM | Fine-tune mode that fits 24 GB | Official sim benchmarks |
|---|---|---|---|---|
| **π0 / π0-FAST / π0.5** | openpi (JAX) | **> 8 GB** (README; 24 GB class) | **LoRA: > 22.5 GB** (README 24 GB class), so it fits but tightly. Full FT needs > 70 GB and does **not fit**. Configs: `pi0_libero_low_mem_finetune`, `pi0_fast_libero_low_mem_finetune`. openpi-PyTorch has **no LoRA**. | LIBERO (π0.5: 98.8/98.2/98.0/92.4), ALOHA-sim |
| π0 / π0.5 (PyTorch port) | LeRobot | — (unverified, expect similar to openpi) | Guide says "24 GB tight at BS 1" (24–40 GB group). Frozen VLM (`train_expert_only`) plus grad checkpointing at BS 1–4 is **unverified**. | LIBERO (LeRobot π0.5: 97.0/99.0/98.0/96.0) |
| **OpenVLA-OFT** (7B) | openvla-oft (conda) | **~16 GB** LIBERO (README) | Training at BS 1 needs **~25 GB** (LIBERO.md), so it does **not fit** per upstream numbers. Upstream offers 8/4-bit loading for inference only. Works on Ampere (bf16) with flash-attn; our scripts build it automatically when a CUDA 12.x toolkit is present. | LIBERO 4 suites (97.1% avg), ALOHA (real) |
| **SmolVLA** (450M) | LeRobot | — (small; unverified) | **Full fine-tune fits.** About 10–16 GB at BS 8 (LeRobot HW guide). | LIBERO (`lerobot/smolvla_libero` released). The paper also reports Meta-World (unverified here). |
| **GR00T N1.7** (3B) | Isaac-GR00T (uv) | **16 GB+** (24 GB class noted in README) | Official minimum is 40 GB+. Default FT (projector + DiT) peaks around 35 GB, so it does **not fit** per upstream. A small-batch attempt is unverified. | LIBERO (97.65/97.5/98.45/94.35), SimplerEnv Bridge and Fractal, RoboCasa, RoboLab |
| **Diffusion Policy** | LeRobot / ManiSkill baseline | small | **Full training fits.** About 8–14 GB at BS 8 (LeRobot); ~2–4 h for 5 epochs on a 24 GB 3090-class GPU (indicative). | PushT (LeRobot's DP reference env); ManiSkill3 DP baseline; any LeRobot sim env (e.g. LIBERO) |
| **ACT** | LeRobot | small | **Full training fits.** About 2–6 GB at BS 8; ~30–60 min (indicative). | ALOHA-sim (LeRobot's ACT reference env); any LeRobot sim env |
| **SimpleVLA-RL** (OFT + veRL) | conda | ~16 GB (OFT) | RL tested on **8× A800 80 GB**, so it does **not fit**. Eval-only (`val_only`) might work (unverified). | LIBERO, RoboTwin 1.0/2.0 |
| **RLinf** (VLA RL) | uv or Docker | model-dependent | Single 24 GB GPU **not documented** (unverified). Supports OFT, π0/π0.5, GR00T on ManiSkill, LIBERO and Isaac Lab. | ManiSkill3, LIBERO, LIBERO-Pro/Plus, Isaac Lab |
| PPO (state / RGB) | ManiSkill3 | — | **Fits.** README says its RGB-PPO settings use < 15 GB. | ManiSkill3 tasks |
| RL (rsl_rl / skrl / sb3) | **Isaac Lab 3.0** (existing) | — | **Fits.** VRAM scales with `num_envs` and cameras. Isaac Sim minimum is 16 GB VRAM. | Isaac-Lift/Reach/Open-Drawer-Franka, KukaAllegro lift/reorient, … |
| Mimic + robomimic BC | **Isaac Lab 3.0** (existing) | — | **Fits** (state-based BC-RNN is small; unverified figure). | Franka Stack-Cube (+ visuomotor), GR1T2, G1 |

What this means in practice: on a 24 GB card you can **evaluate** every released checkpoint above, **LoRA-fine-tune π0 / π0-FAST**
(openpi), **fully fine-tune SmolVLA**, train **ACT and Diffusion Policy** from scratch, and run **ManiSkill and Isaac Lab RL**.
Full fine-tuning of 3B–7B VLAs (OFT, GR00T, π0.5 full) and SimpleVLA-RL training need a larger GPU or a cloud job.
