#!/usr/bin/env python3
"""Two notes-inspired virtual cabinet studio for measured rig renders."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cabinet_variant_workflow import apply_cabinet_variant_audio, filter_variant_output
from rig_capture_workflow import peak_dbfs, read_audio, write_audio


VIRTUAL_STUDIO_VERSION = "measured_virtual_studio_1.0"

# Deterministic early-reflection patterns. These are render effects, not measured
# cabinet responses and never enter the rig-model training target.
ROOM_PRESETS = {
    "off": (),
    "tight": (
        (7.0, 0.26, -0.55),
        (12.5, 0.18, 0.45),
        (19.0, 0.11, -0.15),
    ),
    "studio": (
        (11.0, 0.30, -0.65),
        (23.0, 0.22, 0.55),
        (37.0, 0.15, -0.30),
        (53.0, 0.09, 0.25),
    ),
    "live": (
        (17.0, 0.34, -0.75),
        (41.0, 0.25, 0.65),
        (73.0, 0.17, -0.35),
        (101.0, 0.11, 0.40),
    ),
}


def mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float64)
    return np.mean(array, axis=1) if array.ndim == 2 else array


def db_to_linear(db: float) -> float:
    return float(np.power(10.0, float(db) / 20.0))


def validate_range(name: str, value: float, minimum: float, maximum: float) -> float:
    value = float(value)
    if value < minimum or value > maximum:
        raise SystemExit(f"{name} must be between {minimum:g} and {maximum:g}.")
    return value


def ensure_accepted_variant(metadata: dict, path: Path) -> None:
    validation = dict(metadata.get("validation", {}))
    if validation and not bool(validation.get("accepted", False)):
        raise SystemExit(f"Virtual studio refuses rejected cabinet/mic profile: {path}")


def render_mic(
    audio: np.ndarray,
    sample_rate: int,
    profile_path: Path | None,
    variant_mix: float = 1.0,
    low_cut_hz: float = 0.0,
    high_cut_hz: float = 0.0,
) -> tuple[np.ndarray, dict]:
    if profile_path is None:
        output = filter_variant_output(audio, sample_rate, low_cut_hz, high_cut_hz)
        return output, {
            "name": "captured reference cabinet/microphone",
            "variant": {"cabinet": "captured reference", "microphone": "captured reference"},
        }
    output, metadata = apply_cabinet_variant_audio(
        audio,
        sample_rate,
        profile_path,
        mix=variant_mix,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
    )
    ensure_accepted_variant(metadata, profile_path)
    return output, metadata


def fractional_delay(audio: np.ndarray, delay_samples: float) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    if abs(float(delay_samples)) < 1e-9:
        return source.copy()
    indices = np.arange(len(source), dtype=np.float64)
    return np.interp(indices - float(delay_samples), indices, source, left=0.0, right=0.0)


def pan_mono(audio: np.ndarray, pan: float) -> np.ndarray:
    pan = validate_range("pan", pan, -1.0, 1.0)
    angle = (pan + 1.0) * np.pi / 4.0
    gains = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    return np.asarray(audio, dtype=np.float64)[:, None] * gains[None, :]


def speaker_overload(audio: np.ndarray, amount: float) -> np.ndarray:
    amount = validate_range("speaker overload", amount, 0.0, 1.0)
    source = np.asarray(audio, dtype=np.float64)
    if amount <= 0.0:
        return source.copy()
    drive = 1.0 + (12.0 * amount)
    compressed = np.tanh(source * drive) / drive
    return ((1.0 - amount) * source) + (amount * compressed)


def add_early_reflections(
    direct: np.ndarray,
    sample_rate: int,
    preset: str,
    distance: float,
    room_mix: float,
) -> tuple[np.ndarray, float]:
    if preset not in ROOM_PRESETS:
        raise SystemExit(f"Unknown room preset: {preset}")
    distance = validate_range("distance", distance, 0.0, 1.0)
    room_mix = validate_range("room mix", room_mix, 0.0, 1.0)
    reflections = ROOM_PRESETS[preset]
    effective_mix = room_mix * distance if reflections else 0.0
    if effective_mix <= 0.0:
        return np.asarray(direct, dtype=np.float64), 0.0

    direct = np.asarray(direct, dtype=np.float64)
    room_source = np.mean(direct, axis=1)
    wet = np.zeros_like(direct)
    delay_scale = 0.70 + (0.80 * distance)
    for delay_ms, gain, pan in reflections:
        delayed = fractional_delay(room_source, delay_ms * delay_scale * sample_rate / 1000.0)
        wet += pan_mono(delayed * gain, pan)
    wet_peak = float(np.max(np.abs(wet)) + 1e-12)
    if wet_peak > 1.0:
        wet /= wet_peak
    return ((1.0 - effective_mix) * direct) + (effective_mix * wet), effective_mix


def apply_virtual_studio_audio(
    audio: np.ndarray,
    sample_rate: int,
    mic_a_profile: Path | None = None,
    mic_b_profile: Path | None = None,
    mic_a_variant_mix: float = 1.0,
    mic_morph: float = 0.5,
    mic_a_level_db: float = 0.0,
    mic_b_level_db: float = 0.0,
    mic_a_pan: float = 0.0,
    mic_b_pan: float = 0.0,
    variphi_ms: float = 0.0,
    invert_mic_b: bool = False,
    room_preset: str = "off",
    distance: float = 0.0,
    room_mix: float = 0.20,
    overload: float = 0.0,
    low_cut_hz: float = 0.0,
    high_cut_hz: float = 0.0,
) -> tuple[np.ndarray, dict]:
    source = mono(audio)
    mic_morph = validate_range("mic morph", mic_morph, 0.0, 1.0)
    validate_range("Variphi delay", variphi_ms, -10.0, 10.0)
    mic_a, metadata_a = render_mic(
        source,
        sample_rate,
        mic_a_profile,
        variant_mix=validate_range("mic A variant mix", mic_a_variant_mix, 0.0, 1.0),
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
    )
    mic_a = speaker_overload(mic_a, overload)

    mic_b_enabled = mic_b_profile is not None
    if mic_b_enabled:
        mic_b, metadata_b = render_mic(
            source,
            sample_rate,
            mic_b_profile,
            variant_mix=1.0,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
        )
        mic_b = speaker_overload(mic_b, overload)
        mic_b = fractional_delay(mic_b, float(variphi_ms) * sample_rate / 1000.0)
        if invert_mic_b:
            mic_b *= -1.0
        weight_a = float(np.cos(mic_morph * np.pi / 2.0))
        weight_b = float(np.sin(mic_morph * np.pi / 2.0))
    else:
        mic_b = np.zeros_like(mic_a)
        metadata_b = None
        weight_a, weight_b = 1.0, 0.0

    direct = pan_mono(mic_a * db_to_linear(mic_a_level_db) * weight_a, mic_a_pan)
    if mic_b_enabled:
        direct += pan_mono(mic_b * db_to_linear(mic_b_level_db) * weight_b, mic_b_pan)
    output, effective_room_mix = add_early_reflections(
        direct,
        sample_rate,
        preset=room_preset,
        distance=distance,
        room_mix=room_mix,
    )
    metadata = {
        "version": VIRTUAL_STUDIO_VERSION,
        "power_amp_stage": "off: source is assumed to be a complete captured rig",
        "mic_a": metadata_a,
        "mic_b": metadata_b,
        "controls": {
            "mic_morph": mic_morph,
            "mic_a_weight": weight_a,
            "mic_b_weight": weight_b,
            "mic_a_level_db": float(mic_a_level_db),
            "mic_b_level_db": float(mic_b_level_db),
            "mic_a_pan": float(mic_a_pan),
            "mic_b_pan": float(mic_b_pan),
            "variphi_ms": float(variphi_ms),
            "invert_mic_b": bool(invert_mic_b),
            "room_preset": room_preset,
            "distance": float(distance),
            "room_mix": float(room_mix),
            "effective_room_mix": effective_room_mix,
            "speaker_overload": float(overload),
            "low_cut_hz": float(low_cut_hz),
            "high_cut_hz": float(high_cut_hz),
        },
    }
    return np.nan_to_num(output), metadata


def run_apply_virtual_studio(args) -> None:
    sample_rate, audio = read_audio(Path(args.input))
    output, metadata = apply_virtual_studio_audio(
        audio,
        sample_rate,
        mic_a_profile=Path(args.mic_a) if args.mic_a else None,
        mic_b_profile=Path(args.mic_b) if args.mic_b else None,
        mic_a_variant_mix=float(args.mic_a_variant_mix),
        mic_morph=float(args.mic_morph),
        mic_a_level_db=float(args.mic_a_level_db),
        mic_b_level_db=float(args.mic_b_level_db),
        mic_a_pan=float(args.mic_a_pan),
        mic_b_pan=float(args.mic_b_pan),
        variphi_ms=float(args.variphi_ms),
        invert_mic_b=bool(args.invert_mic_b),
        room_preset=str(args.room_preset),
        distance=float(args.distance),
        room_mix=float(args.room_mix),
        overload=float(args.speaker_overload),
        low_cut_hz=float(args.low_cut_hz),
        high_cut_hz=float(args.high_cut_hz),
    )
    output *= db_to_linear(float(args.output_trim_db))
    if float(np.max(np.abs(output))) >= 1.0:
        if args.limiter == "off":
            raise SystemExit("Virtual studio output clips. Reduce mic levels or --output-trim-db.")
        output = np.tanh(output) / np.tanh(1.0)
        output *= 0.98
    write_audio(Path(args.output), sample_rate, output)
    mic_b_label = "off"
    if metadata["mic_b"] is not None:
        mic_b_label = str(metadata["mic_b"].get("name", args.mic_b))
    print(f"Virtual studio mic A: {metadata['mic_a'].get('name', 'captured reference')}")
    print(f"Virtual studio mic B: {mic_b_label}")
    print(
        f"Morph={float(args.mic_morph):.2f} Variphi={float(args.variphi_ms):+.3f} ms "
        f"room={args.room_preset}/{metadata['controls']['effective_room_mix']:.2f} "
        f"overload={float(args.speaker_overload):.2f}"
    )
    print("Power amp stage: off (prevents stacking over the captured rig power amp)")
    print(f"Output: {args.output} | stereo {sample_rate} Hz | peak={peak_dbfs(output):.2f} dBFS")
