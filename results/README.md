# SuperMimic · LAFAN1 40 clips · RTX 5080 16GB

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

| checkpoint | train time | mean norm. return | step-weighted | failed clips |
|---|---:|---:|---:|---:|
| 24k | 1.6 h | 0.7969 | 0.8062 | 9/40 |
| 40k | 2.7 h | 0.8684 | 0.8788 | 3/40 |
| 44k | 3.0 h | 0.8693 | 0.8744 | 3/40 |
| **48k** | **3.3 h** | **0.8743** | **0.8837** | **3/40** |

- Paper Fig. 4a: **0.82** (1×H100, 1.2 h).
- Independent reproduction e3: 24k **0.824**, 48k **0.904** on 1×RTX 5090 with 4096/4096.
- This run: **0.874** at 48k; +5.4pp vs paper, -3.0pp vs the 5090 e3 run.
  The gap is consistent with half the batch/env count (2048 vs 4096).

Evaluation protocol: Appendix F; each clip is played from its first frame with
the deterministic policy, no observation/action noise, no reset noise, no DR,
foot friction 1.0. Normalized return = sum of rewards / (0.1 × steps). A clip
fails if the training termination fires.

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
├── benchmark_ckpt_0024000.json     # per-clip results (Appendix F)
├── benchmark_ckpt_0040000.json
├── benchmark_ckpt_0044000.json
├── benchmark_ckpt_0048000.json
├── training_curve.png              # train return, imagination survival, WM error
├── training_metrics.csv            # raw TensorBoard scalars
└── videos/
    ├── walk1_subject1.mp4          # 800 steps / 16 s
    ├── dance1_subject1.mp4         # 600 steps / 12 s
    ├── walk_sheet.png              # 4-frame contact sheet
    ├── dance_sheet.png
    └── index.html                  # side-by-side video page
```

White = policy, translucent blue = reference ghost.

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
bash scripts/run_lafan1.sh lafan1_anchor 1 2048 2048 3.3 320000

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

- Single seed; no variance estimate across training seeds.
- 2048/2048 instead of the 5090 reference's 4096/4096; the ~3pp gap is expected.
- Anchor-relative observation deviates from the paper's Table A.1; the paper
  protocol was not run on this machine (the independent reproduction's e2/e7
  runs cover it).
- LAFAN1 evaluation protocol (whole-clip playback) is not fully specified in
  the paper; this repository uses whole-clip deterministic playback.
- No PPO-Sim / PPO-WM baseline, no H×critic ablation, no real-robot deployment.
- EGL emits `EGL_NOT_INITIALIZED` during process teardown; all rollouts complete
  normally.
