#!/usr/bin/env python3
"""Export selected TensorBoard scalars + a training-curve figure for GitHub."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "logs" / "lafan1_anchor" / "tb"
OUT = ROOT / "results" / "lafan1_anchor"
TAGS = [
    "train/episode_norm_return",
    "policy/imag_survival_H",
    "wm/online_nse_1step",
    "wm/online_nse_8step",
    "wm/nse_1step",
    "wm/term_acc",
]


def smooth(y: np.ndarray, k: int = 50) -> np.ndarray:
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ea = EventAccumulator(str(TB), size_guidance={"scalars": 0})
    ea.Reload()
    data = {tag: ea.Scalars(tag) for tag in TAGS if tag in ea.Tags()["scalars"]}

    with (OUT / "training_metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "wall_time", "value"])
        for tag, scalars in data.items():
            for s in scalars:
                w.writerow([tag, s.step, s.wall_time, s.value])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    if "train/episode_norm_return" in data:
        s = data["train/episode_norm_return"]
        x = np.array([e.step for e in s])
        y = np.array([e.value for e in s])
        ax.plot(x, y, alpha=0.25, lw=0.8, color="#1f77b4")
        ys = smooth(y)
        ax.plot(x[len(x) - len(ys) :], ys, lw=1.8, color="#1f77b4", label="train return (50-iter mean)")
    if "policy/imag_survival_H" in data:
        s = data["policy/imag_survival_H"]
        ax.plot([e.step for e in s], [e.value for e in s], lw=1.2, color="#ff7f0e", label="imagined survival H")
    ax.set_xlabel("iteration")
    ax.set_ylabel("value")
    ax.set_title("Training return / imagination survival")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for tag, color, label in [
        ("wm/online_nse_1step", "#2ca02c", "online 1-step"),
        ("wm/online_nse_8step", "#d62728", "online 8-step"),
    ]:
        if tag in data:
            s = data[tag]
            ax.plot([e.step for e in s], [e.value for e in s], lw=1.4, color=color, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("normalized error")
    ax.set_title("World-model online prediction error")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("SuperMimic · LAFAN1 40 clips · RTX 5080 16GB · 2048 envs / batch 2048", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "training_curve.png", dpi=180)
    print(f"wrote {OUT / 'training_metrics.csv'} and {OUT / 'training_curve.png'}")


if __name__ == "__main__":
    main()
