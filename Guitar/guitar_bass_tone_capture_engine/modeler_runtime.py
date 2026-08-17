#!/usr/bin/env python3
"""Guarded runtime ideas shared by portable guitar-modeler performance rigs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirpeak, lfilter, resample_poly, sosfilt


MODELER_RUNTIME_VERSION = "modeler_runtime_1.0"

OUTPUT_DESTINATIONS = {
    "studio-frfr": "Studio, interface, or full-range monitor",
    "headphones": "Headphones or headphone amplifier",
    "amp-input": "Input of a physical guitar amplifier",
    "amp-return": "Effects return of a physical guitar amplifier",
    "power-amp-guitar-cab": "External power amp feeding a physical guitar cabinet",
}

SPEAKER_IMPEDANCE_CURVES = {
    "flat": {
        "resonance_hz": 100.0,
        "resonance_db": 0.0,
        "resonance_q": 1.2,
        "presence_hz": 2800.0,
        "presence_db": 0.0,
        "presence_q": 0.8,
    },
    "open-back-1x12": {
        "resonance_hz": 105.0,
        "resonance_db": 2.5,
        "resonance_q": 1.0,
        "presence_hz": 3200.0,
        "presence_db": 1.2,
        "presence_q": 0.7,
    },
    "closed-back-1x12": {
        "resonance_hz": 115.0,
        "resonance_db": 3.5,
        "resonance_q": 1.35,
        "presence_hz": 2900.0,
        "presence_db": 1.7,
        "presence_q": 0.8,
    },
    "closed-back-4x12": {
        "resonance_hz": 82.0,
        "resonance_db": 4.5,
        "resonance_q": 1.55,
        "presence_hz": 2400.0,
        "presence_db": 2.1,
        "presence_q": 0.9,
    },
    "custom": None,
}

SNAPSHOT_KEYS = {
    "model_morph",
    "primary_level_db",
    "secondary_level_db",
    "input_trim_db",
    "gate_threshold_dbfs",
    "bass_db",
    "middle_db",
    "treble_db",
    "speaker_drive",
    "cone_cry",
    "mic_position_morph",
    "output_trim_db",
    "destination",
}


def _range(name: str, value: float, minimum: float, maximum: float) -> float:
    number = float(value)
    if not minimum <= number <= maximum:
        raise SystemExit(f"{name} must be between {minimum:g} and {maximum:g}.")
    return number


def speaker_curve_config(
    name: str,
    *,
    resonance_hz: float,
    resonance_db: float,
    resonance_q: float,
    presence_hz: float,
    presence_db: float,
    presence_q: float,
) -> dict:
    if name not in SPEAKER_IMPEDANCE_CURVES:
        raise SystemExit(f"Unknown speaker impedance curve: {name}")
    configured = SPEAKER_IMPEDANCE_CURVES[name]
    values = dict(configured) if configured is not None else {
        "resonance_hz": float(resonance_hz),
        "resonance_db": float(resonance_db),
        "resonance_q": float(resonance_q),
        "presence_hz": float(presence_hz),
        "presence_db": float(presence_db),
        "presence_q": float(presence_q),
    }
    values["name"] = name
    _range("Speaker resonance frequency", values["resonance_hz"], 40.0, 400.0)
    _range("Speaker resonance amount", values["resonance_db"], -12.0, 12.0)
    _range("Speaker resonance Q", values["resonance_q"], 0.25, 12.0)
    _range("Speaker presence frequency", values["presence_hz"], 500.0, 10000.0)
    _range("Speaker presence amount", values["presence_db"], -12.0, 12.0)
    _range("Speaker presence Q", values["presence_q"], 0.25, 12.0)
    return values


def _resonant_adjustment(
    audio: np.ndarray,
    sample_rate: int,
    frequency_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    if abs(float(gain_db)) < 1e-9:
        return source
    frequency_hz = min(float(frequency_hz), sample_rate * 0.45)
    b, a = iirpeak(frequency_hz, float(q), fs=sample_rate)
    band = lfilter(b, a, source)
    return source + ((10.0 ** (float(gain_db) / 20.0)) - 1.0) * band


def apply_speaker_impedance_response(audio: np.ndarray, sample_rate: int, config: dict) -> np.ndarray:
    """Apply a stable resonance approximation, not a physical reactive load."""
    output = _resonant_adjustment(
        audio,
        sample_rate,
        float(config.get("resonance_hz", 100.0)),
        float(config.get("resonance_db", 0.0)),
        float(config.get("resonance_q", 1.2)),
    )
    return _resonant_adjustment(
        output,
        sample_rate,
        float(config.get("presence_hz", 2800.0)),
        float(config.get("presence_db", 0.0)),
        float(config.get("presence_q", 0.8)),
    )


def apply_dynamic_speaker(
    audio: np.ndarray,
    sample_rate: int,
    *,
    drive: float,
    cone_cry: float,
    cone_cry_hz: float,
    reference_level_dbfs: float,
) -> np.ndarray:
    """Apply oversampled, level-dependent speaker compression and resonant breakup."""
    drive = _range("Dynamic speaker drive", drive, 0.0, 1.0)
    cone_cry = _range("Cone cry", cone_cry, 0.0, 1.0)
    cone_cry_hz = _range("Cone cry frequency", cone_cry_hz, 700.0, 8000.0)
    reference_level_dbfs = _range("Speaker reference level", reference_level_dbfs, -60.0, -3.0)
    source = np.asarray(audio, dtype=np.float64)
    if drive <= 1e-9 and cone_cry <= 1e-9:
        return source

    oversampled = resample_poly(source, 2, 1)
    oversampled_rate = int(sample_rate * 2)
    detector_hz = min(90.0, oversampled_rate * 0.05)
    envelope = sosfilt(
        butter(1, detector_hz, btype="lowpass", fs=oversampled_rate, output="sos"),
        np.abs(oversampled),
    )
    reference = 10.0 ** (reference_level_dbfs / 20.0)
    activity = np.clip(envelope / max(reference, 1e-9), 0.0, 3.0)
    local_drive = 1.0 + (5.0 * drive * activity)
    normalized = oversampled / max(reference, 1e-9)
    saturated = (np.tanh(normalized * local_drive) / local_drive) * reference
    output = ((1.0 - drive) * oversampled) + (drive * saturated)

    if cone_cry > 1e-9:
        distortion = output - oversampled
        cry_hz = min(cone_cry_hz, oversampled_rate * 0.45)
        b, a = iirpeak(cry_hz, 8.0, fs=oversampled_rate)
        cry = lfilter(b, a, distortion)
        output += cry * (2.0 * cone_cry)

    restored = resample_poly(output, 1, 2)
    restored = restored[: len(source)]
    if not np.all(np.isfinite(restored)):
        raise SystemExit("Dynamic speaker stage produced non-finite audio.")
    return restored


def destination_plan(capture_type: str, destination: str, *, has_full_cabinet: bool) -> dict:
    if destination not in OUTPUT_DESTINATIONS:
        raise SystemExit(f"Unknown output destination: {destination}")
    capture_type = str(capture_type)
    studio_destination = destination in {"studio-frfr", "headphones"}

    if capture_type == "amp-cab" and not studio_destination:
        raise SystemExit(
            "An Amp & Cab capture already contains its cabinet and cannot be routed to an amp input, "
            "amp return, power amp, or physical guitar cabinet."
        )
    if capture_type == "amp-preamp":
        if studio_destination and not has_full_cabinet:
            raise SystemExit("Studio/headphone output from an Amp / Pre-Amp capture requires a full cabinet IR.")
        if destination == "amp-input":
            raise SystemExit("An Amp / Pre-Amp capture cannot feed another amplifier input without stacking preamps.")
    if capture_type == "pedal-only" and destination != "amp-input":
        raise SystemExit("A Pedal Only capture must feed a physical amplifier input in this performance-rig path.")

    return {
        "destination": destination,
        "label": OUTPUT_DESTINATIONS[destination],
        "apply_speaker_stage": capture_type == "amp-preamp" and studio_destination,
        "apply_cabinet_stage": studio_destination,
    }


def validate_input_impedance(
    captured_kohm: float,
    source_kohm: float | None,
    *,
    allow_mismatch: bool,
) -> None:
    captured = _range("Captured input impedance", captured_kohm, 10.0, 10000.0)
    if source_kohm is None:
        return
    source = _range("Source input impedance", source_kohm, 10.0, 10000.0)
    ratio = max(captured, source) / min(captured, source)
    if ratio > 1.25 and not allow_mismatch:
        raise SystemExit(
            f"Input impedance mismatch: preset={captured:g} kOhm, source={source:g} kOhm. "
            "Use matching interface hardware or --allow-input-impedance-mismatch; software cannot recreate "
            "the pickup loading that happened before A/D conversion."
        )


def load_snapshots(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Snapshots JSON not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read snapshots JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit("Snapshots JSON must be an object mapping snapshot names to settings.")
    snapshots: dict[str, dict] = {}
    for raw_name, settings in raw.items():
        name = str(raw_name).strip()
        if not name or not isinstance(settings, dict):
            raise SystemExit("Each snapshot needs a non-empty name and a JSON settings object.")
        unknown = sorted(set(settings) - SNAPSHOT_KEYS)
        if unknown:
            raise SystemExit(f"Snapshot {name!r} has unsupported settings: {', '.join(unknown)}")
        snapshots[name] = dict(settings)
    return snapshots


def blend_model_paths(
    primary: np.ndarray,
    secondary: np.ndarray | None,
    *,
    mode: str,
    balance: float,
    primary_level_db: float,
    secondary_level_db: float,
) -> np.ndarray:
    balance = _range("Model balance", balance, 0.0, 1.0)
    primary_level_db = _range("Primary path level", primary_level_db, -24.0, 12.0)
    secondary_level_db = _range("Secondary path level", secondary_level_db, -24.0, 12.0)
    first = np.asarray(primary, dtype=np.float64) * (10.0 ** (primary_level_db / 20.0))
    if secondary is None:
        if balance > 1e-9:
            raise SystemExit("A non-zero model balance requires a secondary accepted model.")
        return first
    length = min(len(first), len(secondary))
    first = first[:length]
    second = np.asarray(secondary[:length], dtype=np.float64) * (10.0 ** (secondary_level_db / 20.0))
    if mode == "morph":
        blended = ((1.0 - balance) * first) + (balance * second)
        expected_rms = ((1.0 - balance) * math.sqrt(float(np.mean(first * first)))) + (
            balance * math.sqrt(float(np.mean(second * second)))
        )
        actual_rms = math.sqrt(float(np.mean(blended * blended)))
        if expected_rms > 1e-12 and actual_rms > 1e-12:
            blended *= expected_rms / actual_rms
        return blended
    if mode == "parallel":
        angle = balance * math.pi * 0.5
        weight_a = math.cos(angle)
        weight_b = math.sin(angle)
        return ((weight_a * first) + (weight_b * second)) / max(weight_a + weight_b, 1e-12)
    raise SystemExit("Model path mode must be morph or parallel.")
