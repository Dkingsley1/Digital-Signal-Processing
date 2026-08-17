#!/usr/bin/env python3
"""Smoke test for the amp-tone guard that blocks DI-gain-only renders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tone_capture_engine import (
    amp_tone_guard_metrics,
    apply_cabinet_guard_filter,
    apply_reference_spectral_imprint,
    enforce_amp_tone_regression_guard,
    match_reference_level,
    normalize_for_audition,
)


def main() -> None:
    sample_rate = 48_000
    seconds = 4.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float64) / sample_rate
    envelope = np.exp(-1.6 * np.mod(t, 0.5))
    di = (
        0.34 * np.sin(2.0 * np.pi * 73.416 * t)
        + 0.18 * np.sin(2.0 * np.pi * 146.832 * t)
        + 0.09 * np.sin(2.0 * np.pi * 293.665 * t)
        + 0.025 * np.sin(2.0 * np.pi * 2800.0 * t)
    ) * envelope

    amp_like = np.tanh(di * 5.5)
    target = apply_cabinet_guard_filter(
        amp_like,
        sample_rate,
        lowpass_hz=6800.0,
        highpass_hz=85.0,
        presence_db=4.0,
        air_db=0.5,
    )
    target = normalize_for_audition(target, peak=0.7)

    di_gain_only = match_reference_level(di, target, mode="rms")
    bad_metrics = amp_tone_guard_metrics(di, di_gain_only, target, sample_rate)
    if bad_metrics["passes"]:
        raise SystemExit("Guard failed: gain-matched DI passed as amp tone.")

    modeled = np.tanh(di * 4.0)
    modeled = apply_reference_spectral_imprint(
        modeled,
        target,
        sample_rate,
        strength=1.0,
        smoothing_bins=61,
        max_gain_db=18.0,
    )
    modeled = match_reference_level(modeled, target, mode="rms")
    _, good_metrics = enforce_amp_tone_regression_guard(
        modeled,
        di,
        target,
        sample_rate,
        repair_strength=1.0,
        smoothing_bins=61,
        max_gain_db=18.0,
    )
    if not good_metrics["passes"]:
        raise SystemExit("Guard failed: amp-shaped render did not pass.")

    print(
        "amp tone guard ok: "
        f"di_baseline={bad_metrics['di_gain_baseline_spectral_error_db']:.2f}dB "
        f"amp_render={good_metrics['render_spectral_error_db']:.2f}dB "
        f"movement={good_metrics['render_vs_di_spectral_distance_db']:.2f}dB"
    )


if __name__ == "__main__":
    main()
