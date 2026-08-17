#!/usr/bin/env python3
"""Fast isolated-stack checks for the long-memory amp accuracy lane."""

from __future__ import annotations

import tempfile
from pathlib import Path

import auraloss
import numpy as np
import torch

from research_audio_worker import (
    amp_perceptual_loss,
    build_perceptual_losses,
    build_tcn_fullness,
    candidate_metrics,
    write_dry_listening_auditions,
)


def main() -> None:
    sample_rate = 48_000
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, receptive = build_tcn_fullness(channels=8, levels=6, stacks=2, input_channels=3)
    model.to(device)
    assert receptive == 1 + 4 * sum(2**level for level in range(6))
    source = torch.randn(1, 3, receptive + 4096, device=device) * 0.05
    prediction = model(source)[:, :, -4096:]
    target = torch.tanh(source[:, :1, -4096:] * 5.0) * 0.3
    losses = build_perceptual_losses(auraloss, sample_rate, device, 4096)
    loss, components = amp_perceptual_loss(
        prediction,
        target,
        losses,
        sample_rate,
        profile="fullness-v2",
    )
    assert bool(torch.isfinite(loss).item())
    assert components["multiband_log_energy"].item() >= 0.0
    assert components["log_rms_envelope"].item() >= 0.0

    t = np.arange(sample_rate * 6, dtype=np.float32) / sample_rate
    di = (0.12 + 0.08 * np.sin(2.0 * np.pi * 0.4 * t)) * np.sin(2.0 * np.pi * 110.0 * t)
    amp = np.tanh(di * 6.0) * 0.32
    metrics = candidate_metrics(di, amp, amp.copy(), sample_rate, 1.25, 0.75)
    assert metrics["amp_tone_guard_passed"]
    assert metrics["listening_promotion_ready"]
    assert metrics["listening_section_pass_rate"] == 1.0

    with tempfile.TemporaryDirectory() as temporary_dir:
        outputs = write_dry_listening_auditions(
            Path(temporary_dir),
            "perfect_candidate",
            di,
            amp,
            amp,
            sample_rate,
            metrics,
            seconds=1.0,
        )
        assert len(outputs) == 5
        assert all(Path(item["output"]).exists() for item in outputs)

    print("Research accuracy-lane smoke checks passed.")


if __name__ == "__main__":
    main()
