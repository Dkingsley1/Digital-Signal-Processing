#!/usr/bin/env python3
"""Regression checks for the six production-modeling upgrades."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from rig_identity import rig_fingerprint, rig_identity_from_manifest  # noqa: E402
from tone_capture_engine import (  # noqa: E402
    align_pair_fractional,
    amp_model_promotion_decision,
    apply_frequency_gain_curve,
    blend_hammerstein_amp_layer,
    build_amp_window_features,
    detailed_tone_profile,
    heldout_amp_model_metrics,
    local_envelope_profile,
    match_detailed_tone_profile,
    match_reference_local_envelope,
    refresh_dataset_take_quality,
    select_pair_specs_by_rig_policy,
    source_match_feature_vector,
    source_matched_amp_transfer,
    source_matched_segment_profile,
    write_wav_float,
)


def fixed_rig_manifest(pedal: str, mic_position: str = "grille touch") -> dict:
    return {
        "audio_interface": {"sample_rate_hz": 96000},
        "di_box": {
            "amp_name": "Peavey 6505 Mini Head",
            "cabinet_name": "Egnater Tweaker 1x12 Celestion",
            "mic_name": "SM57",
        },
        "take_metadata": {
            "amp_channel": "rhythm",
            "mic_position": mic_position,
            "boost_pedal": pedal,
        },
    }


def check_rig_canonicalization_and_policy() -> None:
    label_a = "Maxon 808 drive 0 tone halfway balance full"
    label_b = "Maxon OD808 drive 0 tone 5 balance 10"
    identity_a = rig_identity_from_manifest(fixed_rig_manifest(label_a))
    identity_b = rig_identity_from_manifest(fixed_rig_manifest(label_b))
    assert identity_a == identity_b
    assert rig_fingerprint(identity_a) == rig_fingerprint(identity_b)

    other_identity = rig_identity_from_manifest(fixed_rig_manifest("none"))
    pairs = [
        {"di_path": Path("one.wav"), "target_path": Path("one_target.wav"), "rig_identity": identity_a},
        {"di_path": Path("two.wav"), "target_path": Path("two_target.wav"), "rig_identity": other_identity},
    ]
    selected, report = select_pair_specs_by_rig_policy(pairs, policy="conditioned")
    assert len(selected) == 2
    assert len(report["group_counts"]) == 2
    strict_rejected = False
    try:
        select_pair_specs_by_rig_policy(pairs, policy="strict")
    except SystemExit:
        strict_rejected = True
    assert strict_rejected


def check_fractional_alignment() -> None:
    sample_rate = 48000
    rng = np.random.default_rng(6505)
    source = sosfilt(
        butter(2, [80, 7500], btype="bandpass", fs=sample_rate, output="sos"),
        rng.normal(0.0, 0.08, sample_rate),
    )
    expected_lag = 73.35
    positions = np.arange(len(source), dtype=np.float64) - expected_lag
    delayed = np.interp(positions, np.arange(len(source), dtype=np.float64), source, left=0.0, right=0.0)
    _, _, measured_lag, polarity = align_pair_fractional(source, -delayed, 0.01, sample_rate)
    assert abs(abs(measured_lag) - expected_lag) < 0.75
    assert polarity == -1


def check_level_preserving_features() -> None:
    source = np.linspace(-0.4, 0.4, 4096, dtype=np.float64)
    indices = np.array([1024, 2048, 3072], dtype=np.int64)
    quiet = build_amp_window_features(source * 0.25, indices, 64, input_scale=1.0)
    loud = build_amp_window_features(source, indices, 64, input_scale=1.0)
    assert np.max(np.abs(quiet[:, :-4])) < np.max(np.abs(loud[:, :-4])) * 0.35


def check_dataset_quality_refresh() -> None:
    sample_rate = 48_000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    di = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    target = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        di_path = root / "quality_refresh_clean_di.wav"
        target_path = root / "quality_refresh_amp_mic_target.wav"
        write_wav_float(di_path, sample_rate, di)
        write_wav_float(target_path, sample_rate, target)
        dataset = {
            "takes": [
                {
                    "take_name": "quality_refresh",
                    "clean_di_wav": str(di_path),
                    "amp_mic_target_wav": str(target_path),
                    "recording_levels": {"level_profile": "normal"},
                    "preferred_for_training": False,
                    "usable_for_training": False,
                },
                {
                    "take_name": "archived_take",
                    "inactive_for_training": True,
                    "preferred_for_training": False,
                },
            ]
        }
        refreshed = refresh_dataset_take_quality(root / "dataset.json", dataset)
        assert refreshed == 1
        assert dataset["takes"][0]["preferred_for_training"]
        assert dataset["takes"][0]["usable_for_training"]
        assert not dataset["takes"][1]["preferred_for_training"]


def check_heldout_promotion_guard() -> None:
    sample_rate = 48000
    rng = np.random.default_rng(808)
    source = rng.normal(0.0, 0.06, sample_rate)
    target = sosfilt(
        butter(4, [90, 5400], btype="bandpass", fs=sample_rate, output="sos"),
        np.tanh(source * 5.0),
    ) * 0.28
    gain_only = source * (np.sqrt(np.mean(target**2)) / max(np.sqrt(np.mean(source**2)), 1e-12))
    good = target + rng.normal(0.0, 0.0002, len(target))
    bad_metrics = heldout_amp_model_metrics(source, target, gain_only, sample_rate, 12.0, -1.0, 4.0)
    good_metrics = heldout_amp_model_metrics(source, target, good, sample_rate, 12.0, -1.0, 4.0)
    assert not bad_metrics["passes"]
    assert not bad_metrics["amp_tone_guard"]["passes"]
    assert good_metrics["passes"]

    candidate = {
        "mean_spectral_error_db": 4.0,
        "mean_match_correlation": 0.7,
        "pass_rate": 1.0,
    }
    existing = {
        "mean_spectral_error_db": 4.5,
        "mean_match_correlation": 0.6,
        "pass_rate": 1.0,
    }
    accepted = amp_model_promotion_decision(
        candidate,
        existing,
        max_mean_spectral_error_db=8.0,
        min_mean_correlation=0.12,
        min_pass_rate=0.8,
        min_existing_improvement_db=0.1,
        max_pair_regression_db=1.0,
        candidate_pairs=[{"index": 1, "spectral_error_db": 4.0}],
        existing_pairs=[{"index": 1, "spectral_error_db": 4.5}],
    )
    assert accepted["accepted"]

    regressed = amp_model_promotion_decision(
        {**candidate, "mean_spectral_error_db": 5.8},
        existing,
        max_mean_spectral_error_db=8.0,
        min_mean_correlation=0.12,
        min_pass_rate=0.8,
        min_existing_improvement_db=0.1,
        max_pair_regression_db=1.0,
        candidate_pairs=[{"index": 1, "spectral_error_db": 5.8}],
        existing_pairs=[{"index": 1, "spectral_error_db": 4.5}],
    )
    assert not regressed["accepted"]


def check_dry_close_mic_render_guards() -> None:
    sample_rate = 48_000
    impulse = np.zeros(4096, dtype=np.float64)
    impulse[1024] = 1.0
    impulse[1025] = -1.0
    filtered = apply_frequency_gain_curve(
        impulse,
        sample_rate,
        np.array([20.0, 1000.0, 6000.0, 24000.0]),
        np.array([-12.0, 3.0, -5.0, -30.0]),
    )
    assert np.max(np.abs(filtered[:1024])) < 1e-10

    source = np.sin(2.0 * np.pi * 220.0 * np.arange(sample_rate) / sample_rate)
    features = source_match_feature_vector(source, sample_rate)
    anchor = {
        "enabled": True,
        "curve_freqs_hz": [100.0, 1000.0, 5000.0],
        "target_over_di_gain_db": [0.0, 0.0, 0.0],
        "source_feature_mean": features.tolist(),
        "source_feature_std": np.ones_like(features).tolist(),
        "per_take_transfer_bank": [
            {
                "take_name": "matching_rig",
                "quality_weight": 1.0,
                "rig_fingerprint": "rig-a",
                "source_features": features.tolist(),
                "target_over_di_gain_db": [0.0, 0.0, 0.0],
            },
            {
                "take_name": "other_mic_rig",
                "quality_weight": 1.0,
                "rig_fingerprint": "rig-b",
                "source_features": features.tolist(),
                "target_over_di_gain_db": [12.0, 12.0, 12.0],
            },
        ],
    }
    matched = source_matched_amp_transfer(
        anchor,
        source,
        sample_rate,
        rig_fingerprint_value="rig-a",
        mic_position="SM57 close",
    )
    assert matched["top_matches"][0]["take_name"] == "matching_rig"
    assert max(np.abs(matched["target_over_di_gain_db"])) < 0.1

    dry, diagnostics = blend_hammerstein_amp_layer(
        source,
        source,
        sample_rate,
        matched,
        mix_override=0.0,
    )
    assert np.array_equal(dry, source)
    assert diagnostics["reason"] == "disabled_for_dry_close_mic_render"

    t = np.arange(sample_rate * 2, dtype=np.float64) / sample_rate
    uneven = np.sin(2.0 * np.pi * 220.0 * t) * (0.08 + 0.92 * np.square(np.sin(2.0 * np.pi * 1.5 * t)))
    dense_reference = np.tanh(uneven * 5.0) * 0.20
    before_spread = local_envelope_profile(uneven, sample_rate)["spread_db"]
    matched_fullness, fullness = match_reference_local_envelope(uneven, dense_reference, sample_rate)
    after_spread = local_envelope_profile(matched_fullness, sample_rate)["spread_db"]
    target_spread = fullness["target"]["spread_db"]
    assert abs(after_spread - target_spread) < abs(before_spread - target_spread)

    rng = np.random.default_rng(6505)
    guitar_like = sosfilt(
        butter(3, [55.0, 10000.0], btype="bandpass", fs=sample_rate, output="sos"),
        rng.normal(0.0, 0.1, sample_rate * 2),
    )
    thin = sosfilt(
        butter(3, [420.0, 7200.0], btype="bandpass", fs=sample_rate, output="sos"),
        guitar_like,
    )
    full = sosfilt(
        butter(3, [75.0, 5600.0], btype="bandpass", fs=sample_rate, output="sos"),
        np.tanh(guitar_like * 3.0),
    )
    target_profile = detailed_tone_profile(full, sample_rate)
    before_profile = detailed_tone_profile(thin, sample_rate)
    body_matched, tone_match = match_detailed_tone_profile(
        thin,
        sample_rate,
        target_profile,
        iterations=3,
    )
    after_profile = detailed_tone_profile(body_matched, sample_rate)
    target_ratios = np.asarray(target_profile["energy_ratios"], dtype=np.float64)
    before_ratios = np.asarray(before_profile["energy_ratios"], dtype=np.float64)
    after_ratios = np.asarray(after_profile["energy_ratios"], dtype=np.float64)
    before_error = float(np.mean(np.abs(10.0 * np.log10((before_ratios + 1e-12) / (target_ratios + 1e-12)))))
    after_error = float(np.mean(np.abs(10.0 * np.log10((after_ratios + 1e-12) / (target_ratios + 1e-12)))))
    assert tone_match["active"]
    assert after_error < before_error * 0.65

    exact_features = source_match_feature_vector(guitar_like, sample_rate)
    segment_profile = detailed_tone_profile(full, sample_rate)
    segment_envelope = local_envelope_profile(full, sample_rate)
    segment_dynamics = {
        "rms": 0.1,
        "peak": 0.4,
        "crest_factor": 4.0,
        "peak_over_rms_db": 12.0,
        "transient_rms_ratio": 0.2,
    }
    segment_anchor = {
        "segment_transfer_bank": [
            {
                "di": "exact.wav",
                "quality_weight": 1.0,
                "start_seconds": 0.0,
                "source_features": (exact_features + 3.0).tolist(),
                "target_detailed_tone_profile": segment_profile,
                "target_local_envelope_profile": segment_envelope,
                "target_dynamic_fingerprint": segment_dynamics,
            },
            {
                "di": "exact.wav",
                "quality_weight": 1.0,
                "start_seconds": 40.0,
                "source_features": exact_features.tolist(),
                "target_detailed_tone_profile": segment_profile,
                "target_local_envelope_profile": segment_envelope,
                "target_dynamic_fingerprint": segment_dynamics,
            },
        ]
    }
    selected_segment = source_matched_segment_profile(
        segment_anchor,
        guitar_like,
        sample_rate,
        source_hint_path=Path("exact.wav"),
    )
    assert selected_segment["enabled"]
    assert selected_segment["top_segments"][0]["start_seconds"] == 40.0
    assert selected_segment["top_segments"][0]["weight"] >= 0.86


def main() -> None:
    check_rig_canonicalization_and_policy()
    check_fractional_alignment()
    check_level_preserving_features()
    check_dataset_quality_refresh()
    check_heldout_promotion_guard()
    check_dry_close_mic_render_guards()
    print("Modeling upgrade smoke checks passed.")


if __name__ == "__main__":
    main()
