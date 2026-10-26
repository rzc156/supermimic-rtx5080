import pytest

from supermimic.eval.benchmark import ClipResult, summarize
from supermimic.training.schedule import cosine_lr


def test_summary_weights_full_clip_length_after_early_termination():
    results = [
        ClipResult("complete", frames=101, steps=100, norm_return=0.8, failed=False, fail_step=-1),
        ClipResult("failed", frames=201, steps=50, norm_return=0.2, failed=True, fail_step=50),
    ]

    summary = summarize(results)

    assert summary["mean_norm_return"] == pytest.approx(0.5)
    assert summary["step_weighted_norm_return"] == pytest.approx(0.4)
    assert summary["failed"] == 1


def test_short_run_preserves_original_lr_schedule():
    for step in (0, 1000, 24000, 47999):
        original = cosine_lr(step, 320000, 2e-4)
        short = cosine_lr(step, 48000, 2e-4, decay_fraction=250000 / 48000)
        assert short == pytest.approx(original)
