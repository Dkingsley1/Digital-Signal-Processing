#!/usr/bin/env python3
"""Regression checks for controlled rig capture, alignment, and causal rendering."""

from __future__ import annotations

import sys
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from rig_capture_workflow import (  # noqa: E402
    accepted_validation,
    align_calibrated_pair,
    build_rig_probe,
    capture_response_diagnostics,
    init_tcn_params,
    numpy_model_params,
    predict_rig_model,
    require_mlx,
    rms,
    spectral_error,
    tcn_receptive_field,
    validation_metrics,
)
from cabinet_variant_workflow import (  # noqa: E402
    CABINET_VARIANT_VERSION,
    extract_cabinet_variant,
    run_build_separated_cabinet,
    save_cabinet_variant,
)
from virtual_studio_workflow import apply_virtual_studio_audio  # noqa: E402


def check_probe() -> None:
    probe, sections, validation_start = build_rig_probe(48000, -18.0, 6505)
    assert 54.0 < len(probe) / 48000 < 56.0
    assert abs(20.0 * np.log10(np.max(np.abs(probe))) + 18.0) < 0.02
    assert sections[-2]["role"] == "validation"
    assert 0 < validation_start < len(probe)
    nonlinear_target = np.tanh(probe * 4.0) * 0.28
    diagnostics = capture_response_diagnostics(
        probe,
        nonlinear_target,
        48000,
        {"sections": sections},
    )
    assert diagnostics["frequency_points_analyzed"] == 131073
    assert 55.0 <= diagnostics["cabinet_resonance_hz"] <= 180.0
    assert np.isfinite(diagnostics["dynamic_compression_slope"])


def check_alignment() -> None:
    rng = np.random.default_rng(12)
    source = rng.normal(0.0, 0.08, 48000)
    target = np.concatenate([np.zeros(137), np.tanh(source * 3.0) * 0.21])[: len(source)]
    aligned_source, aligned_target, latency, polarity = align_calibrated_pair(source, target, 48000)
    assert abs(latency - 137.0) < 0.6
    assert polarity == 1
    assert np.max(np.abs(aligned_source)) < 0.5
    assert np.max(np.abs(aligned_target)) < 0.25


def check_causal_chunking() -> None:
    mx = require_mlx()
    levels = 5
    params = init_tcn_params(mx, channels=4, levels=levels, seed=4)
    params_np = numpy_model_params(params)
    metadata = {
        "levels": levels,
        "receptive_field_samples": tcn_receptive_field(levels),
        "input_scale": 1.0,
        "target_scale": 0.2,
    }
    audio = np.random.default_rng(9).normal(0.0, 0.04, 5000)
    one_chunk = predict_rig_model(audio, metadata, params_np, chunk_samples=len(audio))
    many_chunks = predict_rig_model(audio, metadata, params_np, chunk_samples=513)
    assert one_chunk.shape == audio.shape
    assert np.all(np.isfinite(many_chunks))
    assert np.max(np.abs(one_chunk - many_chunks)) < 2e-6


def check_gain_only_guard() -> None:
    rng = np.random.default_rng(22)
    source = rng.normal(0.0, 0.08, 12000)
    target = np.tanh(source * 5.0) * 0.3
    gain_only = source * (np.dot(source, target) / np.dot(source, source))
    metrics = validation_metrics(source, target, gain_only, 48000)
    thresholds = SimpleNamespace(
        max_validation_esr=10.0,
        min_validation_correlation=-1.0,
        max_validation_spectral_error_db=100.0,
        min_amp_movement_db=1.5,
    )
    accepted, failures = accepted_validation(metrics, thresholds)
    assert not accepted
    assert any("amp movement" in failure for failure in failures)


def check_cabinet_variant() -> None:
    probe, sections, validation_start = build_rig_probe(48000, -18.0, 6505)
    reference = np.tanh(probe * 3.5) * 0.25
    variant = sosfilt(butter(3, 4200, btype="lowpass", fs=48000, output="sos"), reference) * 0.9
    impulse, details = extract_cabinet_variant(
        probe,
        reference,
        variant,
        sample_rate=48000,
        probe_manifest={"sections": sections},
        fft_size=65536,
        fir_length=2048,
        smoothing_bins=65,
        max_correction_db=18.0,
        level_mode="tone-only",
    )
    synthesized = fftconvolve(reference, impulse, mode="full")[: len(reference)]
    target = variant[validation_start:]
    baseline = reference[validation_start:]
    corrected = synthesized[validation_start:]
    baseline *= rms(target) / max(rms(baseline), 1e-12)
    corrected *= rms(target) / max(rms(corrected), 1e-12)
    assert details["frequency_points"] >= 131073
    assert spectral_error(target, corrected, 48000) < spectral_error(target, baseline, 48000)


def check_virtual_studio() -> None:
    sample_rate = 48000
    source = np.random.default_rng(6505).normal(0.0, 0.035, sample_rate)
    metadata = {
        "profile_version": CABINET_VARIANT_VERSION,
        "name": "smoke mic B",
        "sample_rate_hz": sample_rate,
        "variant": {
            "cabinet": "smoke cabinet",
            "microphone": "smoke microphone",
            "mic_position": "edge",
        },
        "validation": {"accepted": True, "failures": []},
    }
    with tempfile.TemporaryDirectory() as directory:
        profile = Path(directory) / "mic_b.npz"
        save_cabinet_variant(profile, metadata, np.array([0.55, 0.30, 0.15]))
        mic_a, metadata_a = apply_virtual_studio_audio(
            source,
            sample_rate,
            mic_b_profile=profile,
            mic_morph=0.0,
        )
        mic_b, metadata_b = apply_virtual_studio_audio(
            source,
            sample_rate,
            mic_b_profile=profile,
            mic_morph=1.0,
            variphi_ms=0.37,
            room_preset="studio",
            distance=0.6,
            room_mix=0.25,
            overload=0.2,
        )
        rejected_profile = Path(directory) / "rejected_mic.npz"
        rejected_metadata = {**metadata, "validation": {"accepted": False, "failures": ["smoke"]}}
        save_cabinet_variant(rejected_profile, rejected_metadata, np.array([1.0]))
        rejected_was_blocked = False
        try:
            apply_virtual_studio_audio(source, sample_rate, mic_a_profile=rejected_profile)
        except SystemExit:
            rejected_was_blocked = True
    assert mic_a.shape == (len(source), 2)
    assert mic_b.shape == mic_a.shape
    assert np.all(np.isfinite(mic_b))
    assert not np.allclose(mic_a, mic_b)
    assert metadata_a["controls"]["mic_a_weight"] == 1.0
    assert metadata_b["controls"]["mic_b_weight"] == 1.0
    assert metadata_b["controls"]["effective_room_mix"] > 0.0
    assert metadata_b["power_amp_stage"].startswith("off")
    assert rejected_was_blocked


def check_separated_cabinet_guard() -> None:
    base = {
        "sample_rate_hz": 96000,
        "capture_valid": True,
        "probe_manifest": {
            "probe_version": "controlled_rig_probe_1.0",
            "sample_rate_hz": 96000,
            "seed": 6505,
            "duration_s": 55.0,
        },
        "routing": {"send_trim_db": 0.0},
        "rig": {"pedal": "Maxon OD808", "amp": "Peavey 6505", "amp_settings": "rhythm"},
    }
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        preamp = directory / "preamp.json"
        amp_cab = directory / "amp_cab.json"
        preamp.write_text(json.dumps({**base, "capture_type": "amp-preamp"}), encoding="utf-8")
        changed_rig = {**base["rig"], "amp_settings": "lead"}
        amp_cab.write_text(
            json.dumps({**base, "capture_type": "amp-cab", "rig": changed_rig}),
            encoding="utf-8",
        )
        rejected = False
        try:
            run_build_separated_cabinet(
                SimpleNamespace(
                    preamp_capture_manifest=preamp,
                    amp_cab_capture_manifest=amp_cab,
                    profile=directory / "cab.npz",
                    name="smoke",
                    mic_axis="on-axis",
                    fft_size=65536,
                    fir_length=2048,
                    smoothing_bins=65,
                    max_correction_db=18.0,
                    level_mode="tone-only",
                    min_spectral_improvement_db=0.5,
                    comparison_output=None,
                    comparison_seconds=1.0,
                    allow_failed_validation=False,
                )
            )
        except SystemExit as exc:
            rejected = "must remain fixed" in str(exc)
    assert rejected


def main() -> None:
    check_probe()
    check_alignment()
    check_causal_chunking()
    check_gain_only_guard()
    check_cabinet_variant()
    check_separated_cabinet_guard()
    check_virtual_studio()
    print("Rig capture smoke checks passed.")


if __name__ == "__main__":
    main()
