#!/usr/bin/env python3
"""Record a headless video of a trained SuperMimic policy on one LAFAN1 clip.

Example:
  PYTHONPATH= .venv/bin/python scripts/record_video.py \
      --ckpt logs/lafan1_anchor/ckpt_latest.pt --clip dance1_subject1 --steps 600
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.wrappers import VideoRecorder

from supermimic.envs import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--npz-dir", default="data/npz/lafan1")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    policy, cfg = load_policy_from_checkpoint(args.ckpt, args.device)
    files = discover_npz(Path(args.npz_dir), names=[args.clip])
    if not files:
        raise FileNotFoundError(f"clip {args.clip} not found under {args.npz_dir}")

    env_args = SuperMimicEnvCfgArgs(
        npz_files=[str(f) for f in files],
        num_envs=1,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode="start",
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=True,
    )
    env = ManagerBasedRlEnv(
        make_env_cfg(env_args), device=args.device, render_mode="rgb_array"
    )

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "videos" / f"{args.clip}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = VideoRecorder(
        env,
        video_folder=out.parent,
        step_trigger=lambda step: step == 0,
        video_length=args.steps,
        name_prefix=out.stem,
        disable_logger=False,
    )

    obs, _ = rec.reset()
    steps = 0
    try:
        for _ in range(args.steps):
            with torch.no_grad():
                action = policy(obs["policy"], obs["goal"])
            obs, _, term, trunc, _ = rec.step(action)
            steps += 1
            if bool(term[0]) or bool(trunc[0]):
                break
    finally:
        rec.close()

    produced = sorted(out.parent.glob(f"{out.stem}-step-*.mp4"))
    if produced:
        produced[-1].replace(out)
    print(f"recorded {steps} steps -> {out}")


if __name__ == "__main__":
    main()
