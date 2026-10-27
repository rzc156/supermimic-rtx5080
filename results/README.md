# SuperMimic · LAFAN1 40 clips · RTX 5080 16GB

**Live page:** <https://rzc156.github.io/supermimic-rtx5080/>

This directory contains the released results of an independent SuperMimic
reproduction on a **single RTX 5080 16GB**, trained for **3.3 h** with
**2048 envs / batch 2048** and anchor-relative observations.

Paper: Liao et al., *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model*, CoRL 2026, [arXiv:2610.28480](https://arxiv.org/abs/2610.28480).
No official code was released by the paper authors; this repository is adapted
from [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
(MIT, Copyright (c) 2026 Yuxi Liu).

---

## Headline

Corrected on 2026-10-26 to use the full T−1 denominator and exclude the extra
terminal reward. The 48k row is the mean of three evaluations.

| checkpoint | train time | mean norm. return | step-weighted | failed clips |
|---|---:|---:|---:|---:|
| 24k | 1.6 h | 0.7177 | 0.7320 | 9/40 |
| **40k** | **2.7 h** | **0.8515** | **0.8608** | **2/40** |
| 44k | 3.0 h | 0.8510 | 0.8563 | 3/40 |
| 48k | 3.3 h | 0.8375 (3-run mean) | 0.8504 | 4–6/40 |

- Paper Fig. 4a: **0.82** (1×H100, 1.2 h).
- Independent reproduction e3: 24k **0.824**, 48k **0.904** on 1×RTX 5090 with 4096/4096.
- Observations, env/batch, hardware and evaluator implementations differ, so these
  numbers do not establish a strict ranking or the cause of remaining gaps.
- Three 48k evaluations produced 0.8403, 0.8379 and 0.8343 (4/5/6 failures);
  three clips changed trajectories or failure status. Single-seed results do not
  establish that 48k is best.

Evaluation protocol: Appendix F; each clip is played from its first frame with
the deterministic policy, no observation/action noise, no reset noise, no DR,
foot friction 1.0. Normalized return = sum of rewards / (0.1 × (T−1)), where T
is the clip's frame count. Early termination does not reduce the denominator;
the extra terminal reward step is excluded. A clip fails if training termination fires.

---

## Key caveat: observations

The run uses **anchor-relative observations**: the paper's 93-dim proprio
observation plus 9 dims of anchor-relative pose in the goal group. The paper's
Table A.1 lists only the 93-dim proprio observation.

The independent reproduction found that with the paper's 93-dim observation,
long LAFAN1 clips drift (mean xy error 6.3 m, yaw 82° over 220 s), and the
benchmark saturates around **0.556–0.582**; adding anchor-relative pose brings
it to 0.824–0.904. This repository therefore reports the anchor-relative run as
the headline and documents the gap in [`../REPRODUCTION_ANALYSIS.md`](../REPRODUCTION_ANALYSIS.md).

---

## Files

```
results/lafan1_anchor/
├── summary.csv                     # 24k/40k/44k/48k table
├── config.json                     # exact training config
├── benchmark_ckpt_*_fullclip.json  # corrected per-clip results (Appendix F)
├── benchmark_ckpt_0048000_fullclip_rerun.json
├── benchmark_ckpt_*.json           # legacy biased-denominator results; not for comparison
├── training_curve.png              # train return, imagination survival, WM error
├── training_metrics.csv            # raw TensorBoard scalars
├── gifs/                           # 6 demo GIFs, 15 s excerpts at 384 px / 12 fps
│   ├── walk1.gif                   # walk1_subject1
│   ├── sprint1.gif                 # sprint1_subject2
│   ├── dance2.gif                  # dance2_subject3
│   ├── fight1.gif                  # fight1_subject5
│   ├── fight_sports.gif            # fightAndSports1_subject4
│   ├── fall_getup.gif              # fallAndGetUp3_subject1
│   └── index.html                  # GIF gallery
└── videos/                         # full 30-40 s rollouts, 960x540, 50 fps
    ├── walk1_subject1.mp4
    ├── sprint1_subject2.mp4
    ├── run1_subject2.mp4
    ├── dance2_subject3.mp4
    ├── fight1_subject5.mp4
    ├── fightAndSports1_subject4.mp4
    ├── fallAndGetUp3_subject1.mp4
    └── index.html                  # full-video page

Demo GIF gallery: https://rzc156.github.io/supermimic-rtx5080/results/lafan1_anchor/gifs/index.html
Full videos:      https://rzc156.github.io/supermimic-rtx5080/results/lafan1_anchor/videos/index.html
```

Both galleries were recorded with the **40k** checkpoint (best single evaluation:
mean 0.8515, 2/40 failures), not the 48k checkpoint. The videos start at an
offset inside each long LAFAN1 clip to show the characteristic motion; this is a
demo choice, not the benchmark protocol, which always plays a clip from frame 0.

---

## Reproduce

```bash
# 1) environment (uv)
uv sync --group dev
env -u PYTHONPATH uv run pytest tests -q

# 2) LAFAN1 G1 CSVs -> 50 Hz npz -> library
uv run python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 \
  --input-fps 30 --workers 16
uv run python scripts/data/build_library.py \
  --npz-dir data/npz/lafan1 --out data/library/lafan1.npz

# 3) 16GB config: 2048 envs / batch 2048, anchor obs, H=32, UTD=0.4
bash scripts/run_lafan1.sh lafan1_anchor_s0 1 2048 2048 3.5 48000 0

# 4) Appendix-F benchmark
bash scripts/eval_iterations.sh lafan1_anchor 24000 40000 44000 48000

# 5) videos
PYTHONPATH= .venv/bin/python scripts/record_video.py \
  --ckpt logs/lafan1_anchor/ckpt_0048000.pt --clip dance1_subject1 --steps 600
```

Data and checkpoints are **not** included (LAFAN1 is ~87 MB of CSVs; the
converted library is ~60 MB; checkpoints are 206 MB each). The conversion
scripts and exact config are included.

---

## Honest limitations

- Single training seed; no variance estimate across training seeds. Three 48k evaluation
  repeats differ slightly, so per-checkpoint rankings remain provisional.
- 2048/2048 instead of the 5090 reference's 4096/4096; its causal effect is untested.
- Anchor-relative observation deviates from the paper's Table A.1; the paper
  protocol was not run on this machine (the independent reproduction's e2/e7
  runs cover it).
- LAFAN1 evaluation protocol (whole-clip playback) is not fully specified in
  the paper; this repository uses whole-clip deterministic playback.
- No PPO-Sim / PPO-WM baseline, no H×critic ablation, no real-robot deployment.
- EGL emits `EGL_NOT_INITIALIZED` during process teardown; all rollouts complete
  normally.
