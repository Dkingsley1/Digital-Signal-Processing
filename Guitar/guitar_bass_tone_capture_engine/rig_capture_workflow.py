#!/usr/bin/env python3
"""Controlled reamp capture and causal MLX rig modeling workflow."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, chirp, correlate, sosfilt


RIG_MODEL_VERSION = "mlx_causal_rig_capture_1.0"
PROBE_VERSION = "rig_probe_1.0"
CAPTURE_TYPE_LABELS = {
    "amp-cab": "Amp & Cab",
    "amp-preamp": "Amp / Pre-Amp",
    "pedal-only": "Pedal Only",
}


def db_to_linear(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64))) + 1e-12))


def peak_dbfs(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(float(np.max(np.abs(audio))) + 1e-12))


def rms_dbfs(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(rms(audio) + 1e-12))


def remove_dc(audio: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return clean - float(np.mean(clean))


def read_audio(path: Path, always_2d: bool = False) -> tuple[int, np.ndarray]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("Rig capture requires soundfile from requirements-audio-advanced.txt.") from exc
    if not path.exists():
        raise SystemExit(f"Audio file not found: {path}. Generate the probe first when using the rig-capture workflow.")
    try:
        data, sample_rate = sf.read(path, dtype="float64", always_2d=always_2d)
    except (OSError, RuntimeError) as exc:
        raise SystemExit(f"Could not read audio file {path}: {exc}") from exc
    return int(sample_rate), np.nan_to_num(np.asarray(data, dtype=np.float64))


def write_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("Rig capture requires soundfile from requirements-audio-advanced.txt.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(np.abs(clean)) + 1e-12) > 1.0:
        raise SystemExit(f"Refusing to write clipped floating-point audio: {path}")
    sf.write(path, clean, sample_rate, subtype="FLOAT")


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float64)
    try:
        import soxr
    except ImportError as exc:
        raise SystemExit("Rig capture oversampling requires soxr from requirements-audio-advanced.txt.") from exc
    return np.asarray(soxr.resample(audio, source_rate, target_rate, quality="VHQ"), dtype=np.float64)


def json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def json_read(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fade_edges(audio: np.ndarray, sample_rate: int, milliseconds: float = 18.0) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float64).copy()
    count = min(len(result) // 2, max(1, int(round(sample_rate * milliseconds / 1000.0))))
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, count)) ** 2
    result[:count] *= ramp
    result[-count:] *= ramp[::-1]
    return result


def level_segment(audio: np.ndarray, peak: float) -> np.ndarray:
    clean = remove_dc(audio)
    current = float(np.max(np.abs(clean)) + 1e-12)
    return clean * (float(peak) / current)


def guitar_band_noise(rng: np.random.Generator, sample_rate: int, seconds: float) -> np.ndarray:
    count = int(round(sample_rate * seconds))
    noise = rng.normal(0.0, 1.0, count)
    sos = butter(6, [55.0, min(12000.0, sample_rate * 0.45)], btype="bandpass", fs=sample_rate, output="sos")
    return sosfilt(sos, noise)


def multisine(rng: np.random.Generator, sample_rate: int, seconds: float) -> np.ndarray:
    count = int(round(sample_rate * seconds))
    t = np.arange(count, dtype=np.float64) / sample_rate
    frequencies = np.geomspace(35.0, min(14000.0, sample_rate * 0.42), 73)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
    signal = np.zeros(count, dtype=np.float64)
    for frequency, phase in zip(frequencies, phases):
        guitar_weight = 1.0 / np.sqrt(max(1.0, frequency / 110.0))
        signal += guitar_weight * np.sin((2.0 * np.pi * frequency * t) + phase)
    return signal


def synthetic_guitar_excitation(rng: np.random.Generator, sample_rate: int, seconds: float) -> np.ndarray:
    count = int(round(sample_rate * seconds))
    result = np.zeros(count, dtype=np.float64)
    note_hz = [65.41, 73.42, 82.41, 98.00, 110.00, 130.81, 146.83, 164.81, 196.00]
    note_samples = int(round(0.62 * sample_rate))
    for note_index, start in enumerate(range(0, count, int(round(0.48 * sample_rate)))):
        frequency = note_hz[note_index % len(note_hz)]
        length = min(note_samples, count - start)
        if length <= 0:
            break
        t = np.arange(length, dtype=np.float64) / sample_rate
        envelope = (1.0 - np.exp(-t * 180.0)) * np.exp(-t * (3.2 + (note_index % 4)))
        pluck = np.zeros(length, dtype=np.float64)
        for harmonic in range(1, 13):
            phase = rng.uniform(-0.12, 0.12)
            pluck += (1.0 / harmonic) * np.sin(2.0 * np.pi * frequency * harmonic * t + phase)
        result[start : start + length] += pluck * envelope
    return result


def build_rig_probe(sample_rate: int, peak_dbfs_value: float, seed: int) -> tuple[np.ndarray, list[dict], int]:
    if sample_rate < 44100:
        raise SystemExit("Rig probes require at least 44.1 kHz.")
    if not -36.0 <= peak_dbfs_value <= -6.0:
        raise SystemExit("--peak-dbfs must be between -36 and -6 dBFS.")

    rng = np.random.default_rng(seed)
    full_peak = db_to_linear(peak_dbfs_value)
    sections: list[dict] = []
    pieces: list[np.ndarray] = []
    cursor = 0

    def append(label: str, signal: np.ndarray, role: str, level_dbfs: float | None = None) -> None:
        nonlocal cursor
        signal = np.asarray(signal, dtype=np.float64)
        start = cursor
        pieces.append(signal)
        cursor += len(signal)
        sections.append(
            {
                "label": label,
                "role": role,
                "start_sample": int(start),
                "end_sample": int(cursor),
                "start_s": float(start / sample_rate),
                "end_s": float(cursor / sample_rate),
                "peak_dbfs": level_dbfs,
            }
        )

    def silence(seconds: float, label: str = "silence") -> None:
        append(label, np.zeros(int(round(sample_rate * seconds))), "calibration")

    silence(1.5, "noise_floor_pre")

    impulses = np.zeros(int(round(sample_rate * 2.5)), dtype=np.float64)
    for position_s in (0.35, 1.35):
        position = int(round(position_s * sample_rate))
        impulses[position : position + 4] = full_peak * np.array([1.0, -0.45, 0.18, -0.06])
    append("latency_impulses", impulses, "calibration", peak_dbfs_value)
    silence(0.5)

    for offset_db in (-18.0, -6.0):
        seconds = 5.0
        t = np.arange(int(round(sample_rate * seconds)), dtype=np.float64) / sample_rate
        sweep = chirp(t, f0=25.0, f1=min(18000.0, sample_rate * 0.44), t1=seconds, method="logarithmic")
        level_db = peak_dbfs_value + offset_db
        append(
            f"log_sweep_{int(level_db)}dbfs",
            fade_edges(level_segment(sweep, db_to_linear(level_db)), sample_rate),
            "training",
            level_db,
        )
        silence(0.35)

    for offset_db in (-18.0, -9.0, 0.0):
        level_db = peak_dbfs_value + offset_db
        signal = multisine(rng, sample_rate, 5.0)
        append(
            f"multisine_{int(level_db)}dbfs",
            fade_edges(level_segment(signal, db_to_linear(level_db)), sample_rate),
            "training",
            level_db,
        )
        silence(0.35)

    noise = guitar_band_noise(rng, sample_rate, 7.0)
    envelope = np.zeros(len(noise), dtype=np.float64)
    block = max(1, int(round(0.45 * sample_rate)))
    levels = [0.08, 0.18, 0.36, 0.72, 1.0, 0.55, 0.22]
    for index, start in enumerate(range(0, len(noise), block)):
        envelope[start : start + block] = levels[index % len(levels)]
    dynamic_noise = noise * envelope
    append(
        "multilevel_guitar_band_noise",
        fade_edges(level_segment(dynamic_noise, full_peak), sample_rate),
        "training",
        peak_dbfs_value,
    )
    silence(0.5)

    guitar_signal = synthetic_guitar_excitation(rng, sample_rate, 10.0)
    append(
        "synthetic_guitar_transients",
        fade_edges(level_segment(guitar_signal, full_peak), sample_rate),
        "training",
        peak_dbfs_value,
    )
    silence(0.75)

    validation_start = cursor
    validation = 0.62 * multisine(rng, sample_rate, 4.5)
    validation += 0.38 * guitar_band_noise(rng, sample_rate, 4.5)
    append(
        "unseen_validation_excitation",
        fade_edges(level_segment(validation, full_peak * 0.82), sample_rate),
        "validation",
        peak_dbfs_value + 20.0 * math.log10(0.82),
    )
    silence(1.0, "noise_floor_post")

    probe = np.concatenate(pieces).astype(np.float64)
    if float(np.max(np.abs(probe))) > full_peak * 1.001:
        raise RuntimeError("Probe construction exceeded requested peak.")
    return probe, sections, int(validation_start)


def run_probe_generate(args) -> None:
    output = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output.with_name(f"{output.stem}_manifest.json")
    probe, sections, validation_start = build_rig_probe(
        sample_rate=int(args.sample_rate),
        peak_dbfs_value=float(args.peak_dbfs),
        seed=int(args.seed),
    )
    write_audio(output, int(args.sample_rate), probe)
    manifest = {
        "probe_version": PROBE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "probe_wav": str(output),
        "sample_rate_hz": int(args.sample_rate),
        "duration_s": float(len(probe) / int(args.sample_rate)),
        "peak_dbfs": peak_dbfs(probe),
        "validation_start_sample": validation_start,
        "sections": sections,
        "seed": int(args.seed),
        "calibration_note": "Set the reamp output to the desired guitar-level send; never connect an amplifier speaker output to an interface.",
    }
    json_write(manifest_path, manifest)
    print("Generated controlled rig probe.")
    print(f"Probe: {output}")
    print(f"Manifest: {manifest_path}")
    print(f"Sample rate: {args.sample_rate} Hz | duration={manifest['duration_s']:.1f}s | peak={manifest['peak_dbfs']:.2f} dBFS")


def parse_device(value: str | None):
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def clone_control_map(values: list[str] | None) -> dict[str, str]:
    controls: dict[str, str] = {}
    for raw in values or []:
        name, separator, value = str(raw).partition("=")
        name = name.strip()
        value = value.strip()
        if not separator or not name or not value:
            raise SystemExit("Each --clone-control must use NAME=VALUE, for example pre_gain=5.")
        controls[name] = value
    return controls


def capture_routing_lines(capture_type: str, args) -> list[str]:
    if capture_type == "amp-cab":
        return [
            f"Interface output channel {args.output_channel} -> REAMP BOX -> pedal/amp input",
            f"Amp -> cabinet -> {args.mic} -> interface input channel {args.target_channel}",
            "Amplifier speaker output -> compatible speaker cabinet only.",
        ]
    if capture_type == "amp-preamp":
        return [
            f"Interface output channel {args.output_channel} -> REAMP BOX -> amp/preamp input",
            f"Approved line-level preamp/FX-send output -> interface input channel {args.target_channel}",
            "A speaker output requires a speaker-rated load/DI with a line output and proper load; never connect it directly to the interface.",
        ]
    if capture_type == "pedal-only":
        return [
            f"Interface output channel {args.output_channel} -> REAMP BOX -> pedal input",
            f"Pedal output -> suitable interface instrument/line input channel {args.target_channel}",
            "No amplifier speaker output belongs in this return path.",
        ]
    raise SystemExit(f"Unsupported capture type: {capture_type}")


def estimate_latency(probe: np.ndarray, target: np.ndarray, sample_rate: int, max_lag_s: float = 0.25) -> tuple[int, float, int]:
    compare_len = min(len(probe), len(target), int(round(sample_rate * 8.0)))
    source = remove_dc(probe[:compare_len])
    returned = remove_dc(target[:compare_len])
    corr = correlate(returned, source, mode="full", method="fft")
    center = len(source) - 1
    radius = int(round(sample_rate * max_lag_s))
    lo = max(1, center - radius)
    hi = min(len(corr) - 1, center + radius + 1)
    magnitude = np.abs(corr[lo:hi])
    local = int(np.argmax(magnitude))
    index = lo + local
    lag = int(index - center)
    polarity = -1 if float(corr[index]) < 0.0 else 1
    fraction = 0.0
    if 0 < index < len(corr) - 1:
        left, middle, right = magnitude[max(0, local - 1)], magnitude[local], magnitude[min(len(magnitude) - 1, local + 1)]
        denominator = left - (2.0 * middle) + right
        if abs(float(denominator)) > 1e-20:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return lag, lag + fraction, polarity


def run_probe_record(args) -> None:
    probe_path = Path(args.probe)
    sample_rate, probe = read_audio(probe_path)
    if probe.ndim == 2:
        probe = probe[:, 0]
    probe_manifest = json_read(Path(args.probe_manifest) if args.probe_manifest else None)
    if probe_manifest and int(probe_manifest.get("sample_rate_hz", sample_rate)) != sample_rate:
        raise SystemExit("Probe manifest sample rate does not match the WAV.")

    input_device = parse_device(args.input_device)
    output_device = parse_device(args.output_device)
    capture_type = str(getattr(args, "capture_type", "amp-cab"))
    capture_label = CAPTURE_TYPE_LABELS.get(capture_type)
    if capture_label is None:
        raise SystemExit(f"Unsupported capture type: {capture_type}")
    input_impedance_kohm = float(getattr(args, "input_impedance_kohm", 1000.0))
    if not 10.0 <= input_impedance_kohm <= 10000.0:
        raise SystemExit("--input-impedance-kohm must be between 10 and 10000.")
    print("Rig capture routing:")
    print(f"  Capture type: {capture_label}")
    print(f"  Interface input impedance: {input_impedance_kohm:g} kOhm")
    for line in capture_routing_lines(capture_type, args):
        print(f"  {line}")
    print("  Never connect an amplifier speaker output directly to the interface.")
    print(f"  Probe peak after send trim: {peak_dbfs(probe) + float(args.send_trim_db):.2f} dBFS")
    if args.dry_run:
        print("Dry run complete. No audio was played and no files were written.")
        return
    if not args.confirm_reamp_routing:
        raise SystemExit("Refusing to play the probe without --confirm-reamp-routing.")

    if int(args.target_channel) < 1 or int(args.target_channel) > int(args.input_channels):
        raise SystemExit("--target-channel must fit within --input-channels.")
    if int(args.output_channel) < 1 or int(args.output_channel) > int(args.output_channels):
        raise SystemExit("--output-channel must fit within --output-channels.")

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit("Rig probe recording requires sounddevice from requirements-interface.txt.") from exc

    send = np.asarray(probe * db_to_linear(float(args.send_trim_db)), dtype=np.float32)
    if float(np.max(np.abs(send))) >= 1.0:
        raise SystemExit("Send trim clips the digital probe. Reduce --send-trim-db.")
    capture = np.zeros((len(send), int(args.input_channels)), dtype=np.float32)
    position = 0
    statuses: list[str] = []
    finished = threading.Event()

    def callback(indata, outdata, frames, time_info, status):
        nonlocal position
        if status:
            statuses.append(str(status))
        outdata.fill(0.0)
        remaining = len(send) - position
        count = min(frames, max(0, remaining))
        if count:
            outdata[:count, int(args.output_channel) - 1] = send[position : position + count]
            capture[position : position + count, :] = indata[:count, :]
            position += count
        if count < frames or position >= len(send):
            raise sd.CallbackStop

    stream = sd.Stream(
        samplerate=sample_rate,
        blocksize=int(args.blocksize),
        device=(input_device, output_device),
        channels=(int(args.input_channels), int(args.output_channels)),
        dtype=("float32", "float32"),
        latency=str(args.latency),
        callback=callback,
        finished_callback=finished.set,
    )
    print(f"Capturing {len(send) / sample_rate:.1f}s at {sample_rate} Hz...")
    with stream:
        if not finished.wait(timeout=(len(send) / sample_rate) + 20.0):
            raise SystemExit("Audio stream timed out before the probe completed.")

    target = capture[:, int(args.target_channel) - 1].astype(np.float64)
    lag_int, lag_float, polarity = estimate_latency(send.astype(np.float64), target, sample_rate)
    hot_percent = float(np.mean(np.abs(target) >= 0.999) * 100.0)

    output_dir = Path(args.output_dir)
    base = str(args.capture_name)
    probe_copy = output_dir / f"{base}_probe_input.wav"
    target_path = output_dir / f"{base}_target_return.wav"
    raw_path = output_dir / f"{base}_raw_inputs.wav"
    manifest_path = output_dir / f"{base}_rig_capture_manifest.json"
    write_audio(probe_copy, sample_rate, send)
    write_audio(target_path, sample_rate, target)
    write_audio(raw_path, sample_rate, capture)

    capture_manifest = {
        "capture_version": "controlled_reamp_capture_1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "capture_name": base,
        "capture_type": capture_type,
        "capture_type_label": capture_label,
        "probe_input_wav": str(probe_copy),
        "target_return_wav": str(target_path),
        "raw_input_channels_wav": str(raw_path),
        "probe_manifest": probe_manifest,
        "sample_rate_hz": sample_rate,
        "input_impedance_kohm": input_impedance_kohm,
        "routing": {
            "input_device": "system_default" if input_device is None else input_device,
            "output_device": "system_default" if output_device is None else output_device,
            "input_channels": int(args.input_channels),
            "target_channel": int(args.target_channel),
            "output_channels": int(args.output_channels),
            "output_channel": int(args.output_channel),
            "send_trim_db": float(args.send_trim_db),
        },
        "rig": {
            "pedal": str(args.pedal),
            "amp": str(args.amp),
            "amp_settings": str(args.amp_settings),
            "cabinet": str(args.cabinet),
            "microphone": str(args.mic),
            "mic_position": str(args.mic_position),
            "send_calibration_dbu": args.send_calibration_dbu,
            "return_calibration_dbu": args.return_calibration_dbu,
            "notes": str(args.notes),
            "clone_controls": clone_control_map(getattr(args, "clone_control", [])),
        },
        "latency": {
            "integer_samples": lag_int,
            "fractional_samples": lag_float,
            "milliseconds": float(1000.0 * lag_float / sample_rate),
            "polarity": polarity,
        },
        "return_levels": {
            "peak_dbfs": peak_dbfs(target),
            "rms_dbfs": rms_dbfs(target),
            "clipped_sample_percent": hot_percent,
        },
        "stream_statuses": statuses,
        "capture_valid": bool(not statuses and hot_percent == 0.0 and -30.0 <= peak_dbfs(target) <= -3.0),
    }
    json_write(manifest_path, capture_manifest)
    print(f"Probe input: {probe_copy}")
    print(f"Target return: {target_path}")
    print(f"Untouched raw inputs: {raw_path}")
    print(f"Capture manifest: {manifest_path}")
    print(
        f"Return peak={peak_dbfs(target):.2f} dBFS rms={rms_dbfs(target):.2f} dBFS "
        f"latency={lag_float:.2f} samples/{1000.0 * lag_float / sample_rate:.3f} ms "
        f"stream_events={len(statuses)}"
    )
    if not capture_manifest["capture_valid"]:
        print("CAPTURE WARNING: level, clipping, or stream status failed. Fix routing/gain and repeat before final training.")


def align_calibrated_pair(source: np.ndarray, target: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    lag_int, lag_float, polarity = estimate_latency(source, target, sample_rate)
    if lag_int > 0:
        target_aligned = target[lag_int:]
        source_aligned = source[: len(target_aligned)]
    elif lag_int < 0:
        source_aligned = source[-lag_int:]
        target_aligned = target[: len(source_aligned)]
    else:
        length = min(len(source), len(target))
        source_aligned = source[:length]
        target_aligned = target[:length]
    length = min(len(source_aligned), len(target_aligned))
    source_aligned = source_aligned[:length]
    target_aligned = target_aligned[:length] * polarity
    fraction = lag_float - lag_int
    if abs(fraction) > 1e-4 and length > 8:
        positions = np.arange(length, dtype=np.float64) + fraction
        target_aligned = np.interp(positions, np.arange(length, dtype=np.float64), target_aligned, left=0.0, right=0.0)
    return remove_dc(source_aligned), remove_dc(target_aligned), lag_float, polarity


def require_mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise SystemExit("Rig model training requires MLX from requirements-mlx.txt.") from exc
    probe = mx.array([0.0])
    mx.eval(probe)
    return mx


def init_tcn_params(mx, channels: int, levels: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def weight(shape, fan_in):
        return mx.array(rng.normal(0.0, math.sqrt(2.0 / max(1, fan_in)), shape).astype(np.float32))

    params = {
        "input_w": weight((channels, 1, 1), 1),
        "input_b": mx.zeros((channels,), dtype=mx.float32),
        "output_w": weight((1, 1, channels), channels),
        "output_b": mx.zeros((1,), dtype=mx.float32),
    }
    for level in range(levels):
        params[f"filter_w_{level}"] = weight((channels, 3, 1), 3)
        params[f"filter_b_{level}"] = mx.zeros((channels,), dtype=mx.float32)
        params[f"gate_w_{level}"] = weight((channels, 3, 1), 3)
        params[f"gate_b_{level}"] = mx.zeros((channels,), dtype=mx.float32)
        params[f"mix_w_{level}"] = weight((channels, 1, channels), channels)
        params[f"mix_b_{level}"] = mx.zeros((channels,), dtype=mx.float32)
    return params


def tcn_receptive_field(levels: int) -> int:
    return int(1 + 2 * sum(2**level for level in range(levels)))


def tcn_forward(mx, params: dict, x, levels: int):
    h = mx.conv1d(x, params["input_w"]) + params["input_b"]
    skip = mx.zeros_like(h)
    channels = int(h.shape[-1])
    for level in range(levels):
        dilation = 2**level
        padding = 2 * dilation
        padded = mx.pad(h, ((0, 0), (padding, 0), (0, 0)))
        filtered = mx.conv1d(
            padded,
            params[f"filter_w_{level}"],
            dilation=dilation,
            groups=channels,
        ) + params[f"filter_b_{level}"]
        gated = mx.conv1d(
            padded,
            params[f"gate_w_{level}"],
            dilation=dilation,
            groups=channels,
        ) + params[f"gate_b_{level}"]
        activation = mx.tanh(filtered) * mx.sigmoid(gated)
        mixed = mx.conv1d(activation, params[f"mix_w_{level}"]) + params[f"mix_b_{level}"]
        h = h + (0.25 * mx.tanh(mixed))
        skip = skip + activation
    return mx.conv1d(mx.tanh(skip / max(1, levels)), params["output_w"]) + params["output_b"]


def multi_resolution_spectral_loss(mx, prediction, target):
    pred = prediction.reshape((-1,))
    ref = target.reshape((-1,))
    losses = []
    length = int(pred.shape[0])
    for fft_size in (256, 512, 1024, 2048):
        frame_count = length // fft_size
        if frame_count < 1:
            continue
        usable = frame_count * fft_size
        window = mx.array(np.hanning(fft_size).astype(np.float32))
        pred_frames = pred[:usable].reshape((frame_count, fft_size)) * window
        ref_frames = ref[:usable].reshape((frame_count, fft_size)) * window
        pred_mag = mx.abs(mx.fft.rfft(pred_frames, axis=-1)) + 1e-6
        ref_mag = mx.abs(mx.fft.rfft(ref_frames, axis=-1)) + 1e-6
        log_loss = mx.mean(mx.square(mx.log(pred_mag) - mx.log(ref_mag)))
        convergence = mx.sqrt(mx.sum(mx.square(pred_mag - ref_mag)) + 1e-8) / (
            mx.sqrt(mx.sum(mx.square(ref_mag)) + 1e-8)
        )
        losses.append(log_loss + (0.35 * convergence))
    if not losses:
        return mx.mean(mx.square(pred - ref))
    result = losses[0]
    for item in losses[1:]:
        result = result + item
    return result / len(losses)


def rig_loss(mx, params: dict, x, y, levels: int, loss_samples: int):
    prediction = tcn_forward(mx, params, x, levels)[:, -loss_samples:, :]
    target = y[:, -loss_samples:, :]
    error = prediction - target
    mse = mx.mean(mx.square(error))
    esr = mse / (mx.mean(mx.square(target)) + 1e-7)
    pre = mx.mean(mx.square((prediction[:, 1:] - 0.97 * prediction[:, :-1]) - (target[:, 1:] - 0.97 * target[:, :-1])))
    envelope = mx.mean(mx.square(mx.abs(prediction) - mx.abs(target)))
    spectral = multi_resolution_spectral_loss(mx, prediction, target)
    total = (0.22 * mse) + (0.46 * esr) + (0.12 * pre) + (0.08 * envelope) + (0.12 * spectral)
    return total


def tree_zeros(mx, params: dict) -> dict:
    return {key: mx.zeros_like(value) for key, value in params.items()}


def clip_gradients(mx, grads: dict, max_norm: float) -> tuple[dict, float]:
    total = None
    for value in grads.values():
        item = mx.sum(mx.square(value))
        total = item if total is None else total + item
    norm = mx.sqrt(total + 1e-12)
    scale = mx.minimum(mx.array(1.0), float(max_norm) / (norm + 1e-12))
    clipped = {key: value * scale for key, value in grads.items()}
    mx.eval(norm)
    return clipped, float(norm.item())


def adam_update(mx, params, grads, first, second, step: int, learning_rate: float):
    next_params, next_first, next_second = {}, {}, {}
    for key, value in params.items():
        grad = grads[key]
        m = (0.9 * first[key]) + (0.1 * grad)
        v = (0.999 * second[key]) + (0.001 * mx.square(grad))
        m_hat = m / (1.0 - (0.9**step))
        v_hat = v / (1.0 - (0.999**step))
        next_params[key] = value - float(learning_rate) * m_hat / (mx.sqrt(v_hat) + 1e-8)
        next_first[key], next_second[key] = m, v
    return next_params, next_first, next_second


def sample_starts(audio: np.ndarray, start: int, end: int, segment_samples: int, hop: int) -> np.ndarray:
    starts = []
    for position in range(max(0, start), max(start, end - segment_samples + 1), max(1, hop)):
        loss_part = audio[position + segment_samples // 2 : position + segment_samples]
        if len(loss_part) and rms(loss_part) > 1e-5:
            starts.append(position)
    return np.asarray(starts, dtype=np.int64)


def numpy_model_params(params: dict) -> dict:
    return {key: np.array(value, dtype=np.float32) for key, value in params.items()}


def save_rig_model(path: Path, metadata: dict, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = numpy_model_params(params)
    np.savez_compressed(path, metadata=json.dumps(metadata, indent=2), **arrays)


def load_rig_model(path: Path) -> tuple[dict, dict]:
    if not path.exists():
        raise SystemExit(f"Rig model not found: {path}. Train and accept the controlled rig model first.")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("model_version") != RIG_MODEL_VERSION:
            raise SystemExit(f"Unsupported rig model version: {metadata.get('model_version')}")
        params = {key: archive[key].astype(np.float32) for key in archive.files if key != "metadata"}
    return metadata, params


def predict_rig_model(audio: np.ndarray, metadata: dict, params_np: dict, chunk_samples: int) -> np.ndarray:
    mx = require_mlx()
    params = {key: mx.array(value) for key, value in params_np.items()}
    levels = int(metadata["levels"])
    receptive = int(metadata["receptive_field_samples"])
    input_scale = float(metadata.get("input_scale", 1.0))
    target_scale = float(metadata.get("target_scale", 1.0))
    source = np.asarray(audio / input_scale, dtype=np.float32)
    result = np.zeros(len(source), dtype=np.float32)
    for start in range(0, len(source), int(chunk_samples)):
        end = min(len(source), start + int(chunk_samples))
        history_start = max(0, start - (receptive - 1))
        segment = source[history_start:end]
        leading = max(0, (receptive - 1) - start)
        if leading:
            segment = np.pad(segment, (leading, 0))
        x = mx.array(segment.reshape(1, -1, 1))
        prediction = tcn_forward(mx, params, x, levels)
        mx.eval(prediction)
        values = np.array(prediction).reshape(-1)
        result[start:end] = values[-(end - start) :]
    return result.astype(np.float64) * target_scale


def spectral_error(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    length = min(len(reference), len(candidate), int(sample_rate * 12.0))
    if length < 1024:
        return float("inf")
    window = np.hanning(length)
    ref = np.abs(np.fft.rfft(reference[:length] * window)) + 1e-8
    cand = np.abs(np.fft.rfft(candidate[:length] * window)) + 1e-8
    return float(np.sqrt(np.mean(np.square(20.0 * np.log10(cand / ref)))))


def validation_metrics(source: np.ndarray, target: np.ndarray, prediction: np.ndarray, sample_rate: int) -> dict:
    length = min(len(source), len(target), len(prediction))
    source, target, prediction = source[:length], target[:length], prediction[:length]
    error = prediction - target
    esr = float(np.mean(np.square(error)) / (np.mean(np.square(target)) + 1e-12))
    if np.std(prediction) > 1e-10 and np.std(target) > 1e-10:
        corr = float(np.corrcoef(prediction, target)[0, 1])
    else:
        corr = 0.0
    gain = float(np.dot(source, target) / (np.dot(source, source) + 1e-12))
    gain_source = source * gain
    return {
        "esr": esr,
        "correlation": corr,
        "spectral_error_db": spectral_error(target, prediction, sample_rate),
        "model_vs_gain_only_distance_db": spectral_error(gain_source, prediction, sample_rate),
        "target_peak_dbfs": peak_dbfs(target),
        "prediction_peak_dbfs": peak_dbfs(prediction),
    }


def capture_response_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    probe_manifest: dict,
) -> dict:
    sections = list(probe_manifest.get("sections", []))
    active_sections = []
    for section in sections:
        start = max(0, int(section.get("start_sample", 0)))
        end = min(len(source), len(target), int(section.get("end_sample", 0)))
        if end - start < 1024 or section.get("peak_dbfs") is None:
            continue
        source_part = source[start:end]
        target_part = target[start:end]
        active_sections.append(
            {
                "label": str(section.get("label", "section")),
                "role": str(section.get("role", "unknown")),
                "input_rms_dbfs": rms_dbfs(source_part),
                "target_rms_dbfs": rms_dbfs(target_part),
                "transfer_gain_db": rms_dbfs(target_part) - rms_dbfs(source_part),
                "input_crest_db": peak_dbfs(source_part) - rms_dbfs(source_part),
                "target_crest_db": peak_dbfs(target_part) - rms_dbfs(target_part),
            }
        )

    dynamic_sections = [item for item in active_sections if "multisine" in item["label"]]
    compression_slope = 1.0
    if len(dynamic_sections) >= 2:
        x = np.asarray([item["input_rms_dbfs"] for item in dynamic_sections], dtype=np.float64)
        y = np.asarray([item["target_rms_dbfs"] for item in dynamic_sections], dtype=np.float64)
        if float(np.ptp(x)) > 1.0:
            compression_slope = float(np.polyfit(x, y, 1)[0])

    sweep_sections = [
        section
        for section in sections
        if str(section.get("label", "")).startswith("log_sweep_") and section.get("peak_dbfs") is not None
    ]
    response_summary = {
        "frequency_points_analyzed": 0,
        "cabinet_resonance_hz": None,
        "cabinet_resonance_intensity_db": None,
        "high_frequency_rolloff_hz": None,
    }
    if sweep_sections:
        section = min(sweep_sections, key=lambda item: float(item.get("peak_dbfs", 0.0)))
        start = max(0, int(section.get("start_sample", 0)))
        end = min(len(source), len(target), int(section.get("end_sample", 0)))
        x = remove_dc(source[start:end])
        y = remove_dc(target[start:end])
        if len(x) >= 8192:
            fft_size = 262144
            window_length = min(len(x), fft_size)
            window = np.hanning(window_length)
            source_fft = np.fft.rfft(x[:window_length] * window, n=fft_size)
            target_fft = np.fft.rfft(y[:window_length] * window, n=fft_size)
            regularizer = max(float(np.max(np.abs(source_fft))) * 1e-7, 1e-12)
            response_db = 20.0 * np.log10((np.abs(target_fft) + regularizer) / (np.abs(source_fft) + regularizer))
            frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
            smoothing = np.ones(129, dtype=np.float64) / 129.0
            smooth_db = np.convolve(response_db, smoothing, mode="same")
            resonance_mask = (frequencies >= 55.0) & (frequencies <= 180.0)
            body_mask = (frequencies >= 220.0) & (frequencies <= 700.0)
            if np.any(resonance_mask) and np.any(body_mask):
                local_index = int(np.argmax(smooth_db[resonance_mask]))
                resonance_indices = np.flatnonzero(resonance_mask)
                resonance_index = int(resonance_indices[local_index])
                body_level = float(np.median(smooth_db[body_mask]))
                response_summary["cabinet_resonance_hz"] = float(frequencies[resonance_index])
                response_summary["cabinet_resonance_intensity_db"] = float(
                    smooth_db[resonance_index] - body_level
                )
                presence_mask = (frequencies >= 700.0) & (frequencies <= 2500.0)
                presence_level = float(np.median(smooth_db[presence_mask])) if np.any(presence_mask) else body_level
                rolloff_candidates = np.flatnonzero(
                    (frequencies >= 2500.0) & (smooth_db <= presence_level - 12.0)
                )
                if len(rolloff_candidates):
                    response_summary["high_frequency_rolloff_hz"] = float(frequencies[rolloff_candidates[0]])
            response_summary["frequency_points_analyzed"] = int(len(frequencies))
            response_summary["analysis_fft_size"] = int(fft_size)
            response_summary["source_section"] = str(section.get("label", "low_level_sweep"))

    return {
        "dynamic_compression_slope": compression_slope,
        "dynamic_compression_amount": float(1.0 - compression_slope),
        "sections": active_sections,
        **response_summary,
    }


def accepted_validation(metrics: dict, args) -> tuple[bool, list[str]]:
    failures = []
    if float(metrics["esr"]) > float(args.max_validation_esr):
        failures.append(f"ESR {metrics['esr']:.3f} > {args.max_validation_esr}")
    if float(metrics["correlation"]) < float(args.min_validation_correlation):
        failures.append(f"correlation {metrics['correlation']:.3f} < {args.min_validation_correlation}")
    if float(metrics["spectral_error_db"]) > float(args.max_validation_spectral_error_db):
        failures.append(
            f"spectral error {metrics['spectral_error_db']:.2f} dB > {args.max_validation_spectral_error_db} dB"
        )
    if float(metrics["model_vs_gain_only_distance_db"]) < float(args.min_amp_movement_db):
        failures.append(
            f"amp movement {metrics['model_vs_gain_only_distance_db']:.2f} dB < {args.min_amp_movement_db} dB"
        )
    return not failures, failures


def loudest_window_start(audio: np.ndarray, window_samples: int) -> int:
    if len(audio) <= window_samples:
        return 0
    hop = max(1, window_samples // 4)
    starts = range(0, len(audio) - window_samples + 1, hop)
    return max(starts, key=lambda start: rms(audio[start : start + window_samples]))


def calibrate_model_input_trim(
    source: np.ndarray,
    target: np.ndarray,
    metadata: dict,
    params_np: dict,
    chunk_samples: int,
) -> tuple[float, dict]:
    sample_rate = int(metadata["sample_rate_hz"])
    window_samples = min(len(source), max(int(sample_rate * 1.5), int(metadata["receptive_field_samples"]) * 2))
    start = loudest_window_start(source, window_samples)
    source_window = source[start : start + window_samples]
    target_window = target[start : start + window_samples]

    tested: list[dict] = []

    def evaluate(trim_db: float) -> float:
        prediction = predict_rig_model(
            source_window * db_to_linear(trim_db),
            metadata,
            params_np,
            chunk_samples=chunk_samples,
        )
        level_error = abs(rms_dbfs(prediction) - rms_dbfs(target_window))
        peak_error = abs(peak_dbfs(prediction) - peak_dbfs(target_window))
        score = float(level_error + (0.15 * peak_error))
        tested.append(
            {
                "trim_db": float(trim_db),
                "score": score,
                "model_rms_dbfs": rms_dbfs(prediction),
                "target_rms_dbfs": rms_dbfs(target_window),
            }
        )
        return score

    coarse = np.arange(-24.0, 24.01, 4.0)
    coarse_scores = [evaluate(float(trim)) for trim in coarse]
    coarse_best = float(coarse[int(np.argmin(coarse_scores))])
    fine = np.arange(max(-30.0, coarse_best - 4.0), min(30.0, coarse_best + 4.0) + 0.01, 1.0)
    fine_scores = [evaluate(float(trim)) for trim in fine]
    best_trim = float(fine[int(np.argmin(fine_scores))])
    best = min(tested, key=lambda item: item["score"])
    return best_trim, {
        "method": "model_return_level_match",
        "window_start_sample": int(start),
        "window_samples": int(window_samples),
        "selected_trim_db": best_trim,
        "selected_score": float(best["score"]),
        "tested": tested,
    }


def run_train_rig_capture(args) -> None:
    mx = require_mlx()
    rng = np.random.default_rng(int(args.seed))
    if int(args.batch_chunks) < 1 or int(args.chunks_per_epoch) < 1:
        raise SystemExit("--batch-chunks and --chunks-per-epoch must be at least 1.")
    if int(args.validation_chunks) < 1:
        raise SystemExit("--validation-chunks must be at least 1.")
    if int(args.early_stopping_patience) < 0 or int(args.lr_patience) < 0:
        raise SystemExit("Patience values cannot be negative; use 0 to disable.")
    if float(args.gradient_clip_norm) <= 0.0:
        raise SystemExit("--gradient-clip-norm must be greater than 0.")
    source_rate, source = read_audio(Path(args.probe))
    target_rate, target = read_audio(Path(args.target))
    if source.ndim == 2:
        source = source[:, 0]
    if target.ndim == 2:
        target = target[:, 0]
    if source_rate != target_rate:
        raise SystemExit("Probe and target return must have the same capture sample rate.")
    capture_manifest = json_read(Path(args.capture_manifest) if args.capture_manifest else None)
    probe_manifest = dict(capture_manifest.get("probe_manifest", {})) if capture_manifest else {}
    if not probe_manifest and args.probe_manifest:
        probe_manifest = json_read(Path(args.probe_manifest))

    source, target, latency_samples, polarity = align_calibrated_pair(source, target, source_rate)
    response_diagnostics = capture_response_diagnostics(
        source,
        target,
        sample_rate=source_rate,
        probe_manifest=probe_manifest,
    )
    oversample = int(args.oversample_factor)
    if oversample not in {1, 2, 4}:
        raise SystemExit("--oversample-factor must be 1, 2, or 4.")
    model_rate = int(source_rate * oversample)
    if oversample > 1:
        source = resample_audio(source, source_rate, model_rate)
        target = resample_audio(target, target_rate, model_rate)

    input_scale = 1.0
    target_scale = max(0.05, float(np.percentile(np.abs(target), 99.9)))
    source_scaled = np.asarray(source / input_scale, dtype=np.float32)
    target_scaled = np.asarray(target / target_scale, dtype=np.float32)

    levels = int(args.levels)
    channels = int(args.channels)
    if levels < 3 or channels < 2:
        raise SystemExit("Rig model requires at least 3 levels and 2 channels.")
    receptive = tcn_receptive_field(levels)
    loss_samples = int(args.chunk_samples)
    segment_samples = receptive - 1 + loss_samples
    if len(source_scaled) < segment_samples * 3:
        raise SystemExit("Capture is too short for the requested receptive field and chunk size.")

    validation_start_capture = int(probe_manifest.get("validation_start_sample", round(len(source) * 0.84)))
    validation_start = int(round(validation_start_capture * oversample))
    validation_start = int(np.clip(validation_start, segment_samples, len(source_scaled) - segment_samples))
    train_starts = sample_starts(
        source_scaled,
        start=0,
        end=validation_start,
        segment_samples=segment_samples,
        hop=max(128, loss_samples // 3),
    )
    validation_starts = sample_starts(
        source_scaled,
        start=validation_start,
        end=len(source_scaled),
        segment_samples=segment_samples,
        hop=max(128, loss_samples // 2),
    )
    if not len(train_starts) or not len(validation_starts):
        raise SystemExit("Probe does not contain enough active training and validation chunks.")
    if len(validation_starts) > int(args.validation_chunks):
        validation_starts = rng.choice(validation_starts, size=int(args.validation_chunks), replace=False)

    params = init_tcn_params(mx, channels=channels, levels=levels, seed=int(args.seed))
    first, second = tree_zeros(mx, params), tree_zeros(mx, params)

    def loss_fn(candidate_params, x_batch, y_batch):
        return rig_loss(mx, candidate_params, x_batch, y_batch, levels=levels, loss_samples=loss_samples)

    value_and_grad = mx.value_and_grad(loss_fn)
    history = []
    best_params_np = None
    best_epoch = 0
    best_validation = float("inf")
    stale_epochs = 0
    learning_rate = float(args.learning_rate)
    step = 0
    print("Training calibrated causal rig capture...")
    print(
        f"capture_rate={source_rate} model_rate={model_rate} oversample={oversample}x "
        f"levels={levels} channels={channels} receptive={receptive} samples/{1000.0 * receptive / model_rate:.1f}ms"
    )
    print(
        f"Absolute calibration preserved: input_scale={input_scale:.4f} target_scale={target_scale:.6f} "
        f"latency={latency_samples:.3f} samples polarity={polarity:+d}"
    )

    batch_chunks = int(args.batch_chunks)
    chunks_per_epoch = int(args.chunks_per_epoch)
    for epoch in range(1, int(args.epochs) + 1):
        epoch_losses = []
        epoch_grad_norms = []
        remaining = chunks_per_epoch
        while remaining > 0:
            count = min(batch_chunks, remaining)
            chosen = rng.choice(train_starts, size=count, replace=len(train_starts) < count)
            x_np = np.stack([source_scaled[start : start + segment_samples] for start in chosen], axis=0)[..., None]
            y_np = np.stack([target_scaled[start : start + segment_samples] for start in chosen], axis=0)[..., None]
            loss, grads = value_and_grad(params, mx.array(x_np), mx.array(y_np))
            grads, grad_norm = clip_gradients(mx, grads, float(args.gradient_clip_norm))
            step += 1
            params, first, second = adam_update(mx, params, grads, first, second, step, learning_rate)
            mx.eval(loss, *params.values(), *first.values(), *second.values())
            epoch_losses.append(float(loss.item()))
            epoch_grad_norms.append(grad_norm)
            remaining -= count

        validation_losses = []
        for start in validation_starts:
            x_np = source_scaled[start : start + segment_samples].reshape(1, -1, 1)
            y_np = target_scaled[start : start + segment_samples].reshape(1, -1, 1)
            value = loss_fn(params, mx.array(x_np), mx.array(y_np))
            mx.eval(value)
            validation_losses.append(float(value.item()))
        train_loss = float(np.mean(epoch_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": learning_rate,
                "mean_gradient_norm": float(np.mean(epoch_grad_norms)),
            }
        )
        improved = validation_loss < best_validation - float(args.min_delta)
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_params_np = numpy_model_params(params)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if int(args.lr_patience) > 0 and stale_epochs % int(args.lr_patience) == 0:
                learning_rate = max(float(args.min_learning_rate), learning_rate * float(args.lr_decay))
        if epoch == 1 or epoch % int(args.print_every) == 0 or improved or epoch == int(args.epochs):
            marker = " best" if improved else ""
            print(
                f"Epoch {epoch:03d}: train={train_loss:.6f} validation={validation_loss:.6f} "
                f"lr={learning_rate:.7f} grad={np.mean(epoch_grad_norms):.3f}{marker}"
            )
        if int(args.early_stopping_patience) > 0 and stale_epochs >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.")
            break

    if best_params_np is None:
        raise SystemExit("Training did not produce a finite validation checkpoint.")
    params_np = best_params_np

    validation_source = source[validation_start:]
    validation_target = target[validation_start:]
    base_metadata = {
        "model_version": RIG_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "capture_sample_rate_hz": source_rate,
        "sample_rate_hz": model_rate,
        "oversample_factor": oversample,
        "levels": levels,
        "channels": channels,
        "receptive_field_samples": receptive,
        "receptive_field_ms": float(1000.0 * receptive / model_rate),
        "input_scale": input_scale,
        "target_scale": target_scale,
        "application_input_trim_db": 0.0,
        "alignment_latency_samples_at_capture_rate": latency_samples,
        "target_polarity": polarity,
        "calibration_preserved": True,
        "capture_type": str(capture_manifest.get("capture_type", "amp-cab")) if capture_manifest else "amp-cab",
        "capture_type_label": str(capture_manifest.get("capture_type_label", "Amp & Cab"))
        if capture_manifest
        else "Amp & Cab",
        "capture_manifest": capture_manifest,
        "probe_manifest": probe_manifest,
        "measured_response": response_diagnostics,
        "training": {
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "history": history,
            "early_stopping_patience": int(args.early_stopping_patience),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "loss": "waveform + ESR + pre-emphasis + envelope + full-chunk MR-STFT",
        },
    }
    validation_prediction = predict_rig_model(
        validation_source,
        base_metadata,
        params_np,
        chunk_samples=int(args.render_chunk_samples),
    )
    metrics = validation_metrics(validation_source, validation_target, validation_prediction, model_rate)
    accepted, failures = accepted_validation(metrics, args)
    base_metadata["validation"] = {
        **metrics,
        "accepted": accepted,
        "failures": failures,
        "thresholds": {
            "max_esr": float(args.max_validation_esr),
            "min_correlation": float(args.min_validation_correlation),
            "max_spectral_error_db": float(args.max_validation_spectral_error_db),
            "min_amp_movement_db": float(args.min_amp_movement_db),
        },
    }

    requested_model = Path(args.model)
    model_path = requested_model
    if not accepted and not args.allow_failed_validation:
        model_path = requested_model.with_name(f"{requested_model.stem}.rejected{requested_model.suffix}")
    save_rig_model(model_path, base_metadata, params_np)

    output_rate = int(args.render_sample_rate or source_rate)
    render = validation_prediction
    target_for_output = validation_target
    if model_rate != output_rate:
        render = resample_audio(render, model_rate, output_rate)
        target_for_output = resample_audio(target_for_output, model_rate, output_rate)
    output_path = Path(args.output)
    audition_render = render
    audition_peak = float(np.max(np.abs(audition_render)) + 1e-12)
    if audition_peak >= 1.0:
        audition_render = audition_render * (0.98 / audition_peak)
        print("Validation audition exceeded 0 dBFS and was peak-scaled for WAV playback; validation used the unscaled render.")
    write_audio(output_path, output_rate, audition_render)
    if args.comparison_output:
        silence = np.zeros(int(round(0.75 * output_rate)), dtype=np.float64)
        length = min(len(target_for_output), len(audition_render), int(round(float(args.comparison_seconds) * output_rate)))
        comparison = np.concatenate([target_for_output[:length], silence, audition_render[:length]])
        write_audio(Path(args.comparison_output), output_rate, comparison)

    print(f"Best checkpoint: epoch {best_epoch} validation_loss={best_validation:.6f}")
    print(
        f"Validation: accepted={accepted} ESR={metrics['esr']:.3f} corr={metrics['correlation']:.3f} "
        f"spectral={metrics['spectral_error_db']:.2f}dB movement={metrics['model_vs_gain_only_distance_db']:.2f}dB"
    )
    print(f"Model: {model_path}")
    print(f"Validation render: {output_path}")
    print(
        "Measured response: "
        f"frequency_points={response_diagnostics.get('frequency_points_analyzed', 0)} "
        f"cabinet_resonance={response_diagnostics.get('cabinet_resonance_hz')} Hz "
        f"resonance_intensity={response_diagnostics.get('cabinet_resonance_intensity_db')} dB "
        f"compression_slope={response_diagnostics.get('dynamic_compression_slope', 1.0):.3f}"
    )
    if failures:
        print("Validation failures:")
        for failure in failures:
            print(f"  - {failure}")
    if not accepted and not args.allow_failed_validation:
        raise SystemExit("Rig capture was rejected; the requested production model path was not overwritten.")


def run_refine_rig_capture(args) -> None:
    mx = require_mlx()
    rng = np.random.default_rng(int(args.seed))
    if int(args.epochs) < 1 or int(args.batch_chunks) < 1 or int(args.chunks_per_epoch) < 1:
        raise SystemExit("--epochs, --batch-chunks, and --chunks-per-epoch must be at least 1.")
    if int(args.validation_chunks) < 1 or int(args.chunk_samples) < 128:
        raise SystemExit("--validation-chunks must be at least 1 and --chunk-samples at least 128.")
    if int(args.lr_patience) < 0 or int(args.early_stopping_patience) < 0:
        raise SystemExit("Patience values cannot be negative; use 0 to disable.")
    if float(args.gradient_clip_norm) <= 0.0:
        raise SystemExit("--gradient-clip-norm must be greater than 0.")
    base_model_path = Path(args.model)
    requested_model_path = Path(args.output_model)
    if base_model_path.resolve() == requested_model_path.resolve():
        raise SystemExit("Refinement requires a new --output-model path so the accepted base model stays protected.")

    metadata, base_params_np = load_rig_model(base_model_path)
    input_rate, source = read_audio(Path(args.di))
    target_rate, target = read_audio(Path(args.target))
    if source.ndim == 2:
        source = source[:, 0]
    if target.ndim == 2:
        target = target[:, 0]
    if target_rate != input_rate:
        target = resample_audio(target, target_rate, input_rate)
    source, target, latency_samples, polarity = align_calibrated_pair(source, target, input_rate)

    model_rate = int(metadata["sample_rate_hz"])
    source = resample_audio(source, input_rate, model_rate)
    target = resample_audio(target, input_rate, model_rate)
    length = min(len(source), len(target))
    source, target = source[:length], target[:length]
    if float(args.validation_fraction) <= 0.05 or float(args.validation_fraction) >= 0.45:
        raise SystemExit("--validation-fraction must be between 0.05 and 0.45.")

    if args.input_trim_db is None:
        input_trim_db, calibration = calibrate_model_input_trim(
            source,
            target,
            metadata,
            base_params_np,
            chunk_samples=int(args.render_chunk_samples),
        )
    else:
        input_trim_db = float(args.input_trim_db)
        calibration = {
            "method": "manual",
            "selected_trim_db": input_trim_db,
        }
    model_source = source * db_to_linear(input_trim_db)

    levels = int(metadata["levels"])
    receptive = int(metadata["receptive_field_samples"])
    loss_samples = int(args.chunk_samples)
    segment_samples = receptive - 1 + loss_samples
    validation_start = int(round(length * (1.0 - float(args.validation_fraction))))
    validation_start = int(np.clip(validation_start, segment_samples, length - segment_samples))
    train_starts = sample_starts(
        model_source,
        start=0,
        end=validation_start,
        segment_samples=segment_samples,
        hop=max(128, loss_samples // 3),
    )
    validation_starts = sample_starts(
        model_source,
        start=validation_start,
        end=length,
        segment_samples=segment_samples,
        hop=max(128, loss_samples // 2),
    )
    if not len(train_starts) or not len(validation_starts):
        raise SystemExit(
            "Refinement recording needs more active guitar in both its training and held-out ending. "
            "Play hard chords, palm mutes, sustained notes, and strong transients throughout the take."
        )
    if len(validation_starts) > int(args.validation_chunks):
        validation_starts = rng.choice(validation_starts, size=int(args.validation_chunks), replace=False)

    input_scale = float(metadata.get("input_scale", 1.0))
    target_scale = float(metadata.get("target_scale", 1.0))
    source_scaled = np.asarray(model_source / input_scale, dtype=np.float32)
    target_scaled = np.asarray(target / target_scale, dtype=np.float32)
    params = {key: mx.array(value) for key, value in base_params_np.items()}
    first, second = tree_zeros(mx, params), tree_zeros(mx, params)

    def loss_fn(candidate_params, x_batch, y_batch):
        return rig_loss(mx, candidate_params, x_batch, y_batch, levels=levels, loss_samples=loss_samples)

    def validation_loss(candidate_params) -> float:
        values = []
        for start in validation_starts:
            x_np = source_scaled[start : start + segment_samples].reshape(1, -1, 1)
            y_np = target_scaled[start : start + segment_samples].reshape(1, -1, 1)
            value = loss_fn(candidate_params, mx.array(x_np), mx.array(y_np))
            mx.eval(value)
            values.append(float(value.item()))
        return float(np.mean(values))

    value_and_grad = mx.value_and_grad(loss_fn)
    best_params_np = dict(base_params_np)
    best_validation = validation_loss(params)
    best_epoch = 0
    learning_rate = float(args.learning_rate)
    stale_epochs = 0
    step = 0
    history = []

    validation_source = model_source[validation_start:]
    validation_target = target[validation_start:]
    before_prediction = predict_rig_model(
        validation_source,
        metadata,
        base_params_np,
        chunk_samples=int(args.render_chunk_samples),
    )
    before_metrics = validation_metrics(
        validation_source,
        validation_target,
        before_prediction,
        model_rate,
    )

    print("Refining the controlled rig model with real guitar performance...")
    print(
        f"Input calibration={input_trim_db:+.1f} dB | latency={latency_samples:.3f} samples "
        f"polarity={polarity:+d} | starting_validation={best_validation:.6f}"
    )
    print(
        "Play-data focus: hard chords, intermodulation, palm mutes, sustained notes, "
        "pick attack, and volume cleanup."
    )

    for epoch in range(1, int(args.epochs) + 1):
        losses = []
        grad_norms = []
        remaining = int(args.chunks_per_epoch)
        while remaining > 0:
            count = min(int(args.batch_chunks), remaining)
            chosen = rng.choice(train_starts, size=count, replace=len(train_starts) < count)
            x_np = np.stack([source_scaled[start : start + segment_samples] for start in chosen], axis=0)[..., None]
            y_np = np.stack([target_scaled[start : start + segment_samples] for start in chosen], axis=0)[..., None]
            loss, grads = value_and_grad(params, mx.array(x_np), mx.array(y_np))
            grads, grad_norm = clip_gradients(mx, grads, float(args.gradient_clip_norm))
            step += 1
            params, first, second = adam_update(mx, params, grads, first, second, step, learning_rate)
            mx.eval(loss, *params.values(), *first.values(), *second.values())
            losses.append(float(loss.item()))
            grad_norms.append(grad_norm)
            remaining -= count

        candidate_validation = validation_loss(params)
        improved = candidate_validation < best_validation - float(args.min_delta)
        if improved:
            best_validation = candidate_validation
            best_params_np = numpy_model_params(params)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if int(args.lr_patience) > 0 and stale_epochs % int(args.lr_patience) == 0:
                learning_rate = max(float(args.min_learning_rate), learning_rate * float(args.lr_decay))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": candidate_validation,
                "learning_rate": learning_rate,
                "mean_gradient_norm": float(np.mean(grad_norms)),
                "best_checkpoint": improved,
            }
        )
        if epoch == 1 or improved or epoch % int(args.print_every) == 0 or epoch == int(args.epochs):
            marker = " best" if improved else ""
            print(
                f"Refine {epoch:03d}: train={np.mean(losses):.6f} validation={candidate_validation:.6f} "
                f"lr={learning_rate:.7f} grad={np.mean(grad_norms):.3f}{marker}"
            )
        if int(args.early_stopping_patience) > 0 and stale_epochs >= int(args.early_stopping_patience):
            print(f"Refinement stopped at epoch {epoch}; restoring epoch {best_epoch}.")
            break

    after_prediction = predict_rig_model(
        validation_source,
        metadata,
        best_params_np,
        chunk_samples=int(args.render_chunk_samples),
    )
    after_metrics = validation_metrics(
        validation_source,
        validation_target,
        after_prediction,
        model_rate,
    )
    accepted, failures = accepted_validation(after_metrics, args)
    material_improvements = [
        after_metrics["esr"] <= before_metrics["esr"] * (1.0 - float(args.min_esr_improvement)),
        after_metrics["spectral_error_db"] <= before_metrics["spectral_error_db"] - float(args.min_spectral_improvement_db),
        after_metrics["correlation"] >= before_metrics["correlation"] + float(args.min_correlation_improvement),
    ]
    if not any(material_improvements):
        failures.append("held-out refinement did not materially improve ESR, spectrum, or correlation")
    if after_metrics["esr"] > before_metrics["esr"] * 1.05:
        failures.append("held-out ESR regressed by more than 5%")
    if after_metrics["spectral_error_db"] > before_metrics["spectral_error_db"] + 0.50:
        failures.append("held-out spectral error regressed by more than 0.50 dB")
    if after_metrics["correlation"] < before_metrics["correlation"] - 0.02:
        failures.append("held-out correlation regressed by more than 0.02")
    accepted = bool(accepted and not failures)

    refined_metadata = dict(metadata)
    refined_metadata["created_at"] = datetime.now().isoformat(timespec="seconds")
    refined_metadata["application_input_trim_db"] = input_trim_db
    refined_metadata["refinement"] = {
        "method": "real_guitar_dynamics_refinement_1.0",
        "base_model": str(base_model_path),
        "di": str(args.di),
        "target": str(args.target),
        "recording_sample_rate_hz": int(input_rate),
        "alignment_latency_samples": latency_samples,
        "target_polarity": polarity,
        "input_calibration": calibration,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "history": history,
        "before_validation": before_metrics,
        "after_validation": after_metrics,
        "accepted": accepted,
        "failures": failures,
    }

    model_path = requested_model_path
    if not accepted and not args.allow_failed_validation:
        model_path = requested_model_path.with_name(
            f"{requested_model_path.stem}.rejected{requested_model_path.suffix}"
        )
    save_rig_model(model_path, refined_metadata, best_params_np)

    output_rate = int(args.output_sample_rate or input_rate)
    output = resample_audio(after_prediction, model_rate, output_rate)
    before_output = resample_audio(before_prediction, model_rate, output_rate)
    target_output = resample_audio(validation_target, model_rate, output_rate)
    output_peak = float(np.max(np.abs(output)) + 1e-12)
    output_for_wav = output if output_peak < 1.0 else output * (0.98 / output_peak)
    write_audio(Path(args.output), output_rate, output_for_wav)
    if args.comparison_output:
        length = min(
            len(target_output),
            len(before_output),
            len(output),
            int(round(float(args.comparison_seconds) * output_rate)),
        )
        common_peak = max(
            float(np.max(np.abs(target_output[:length]))),
            float(np.max(np.abs(before_output[:length]))),
            float(np.max(np.abs(output[:length]))),
            1e-12,
        )
        common_scale = min(1.0, 0.98 / common_peak)
        silence = np.zeros(int(round(0.75 * output_rate)), dtype=np.float64)
        comparison = np.concatenate(
            [
                target_output[:length] * common_scale,
                silence,
                before_output[:length] * common_scale,
                silence,
                output[:length] * common_scale,
            ]
        )
        write_audio(Path(args.comparison_output), output_rate, comparison)

    print(f"Best refinement checkpoint: epoch {best_epoch} validation_loss={best_validation:.6f}")
    print(
        "Held-out metrics: "
        f"ESR {before_metrics['esr']:.3f}->{after_metrics['esr']:.3f} | "
        f"corr {before_metrics['correlation']:.3f}->{after_metrics['correlation']:.3f} | "
        f"spectral {before_metrics['spectral_error_db']:.2f}->{after_metrics['spectral_error_db']:.2f} dB"
    )
    print(f"Refinement accepted: {accepted}")
    print(f"Model: {model_path}")
    print(f"Refined validation render: {args.output}")
    if failures:
        print("Refinement failures:")
        for failure in failures:
            print(f"  - {failure}")
    if not accepted and not args.allow_failed_validation:
        raise SystemExit("Refinement was rejected; the accepted base model and requested output path remain protected.")


def run_apply_rig_capture(args) -> None:
    input_rate, audio = read_audio(Path(args.input))
    if audio.ndim == 2:
        audio = audio[:, 0]
    metadata, params = load_rig_model(Path(args.model))
    model_rate = int(metadata["sample_rate_hz"])
    model_input = resample_audio(remove_dc(audio), input_rate, model_rate)
    stored_trim_db = 0.0 if args.ignore_model_input_calibration else float(
        metadata.get("application_input_trim_db", 0.0)
    )
    effective_trim_db = stored_trim_db + float(args.input_trim_db)
    model_input *= db_to_linear(effective_trim_db)
    prediction = predict_rig_model(
        model_input,
        metadata,
        params,
        chunk_samples=int(args.chunk_samples),
    )
    output_rate = int(args.output_sample_rate or input_rate)
    output = resample_audio(prediction, model_rate, output_rate)
    if float(args.output_trim_db):
        output *= db_to_linear(float(args.output_trim_db))
    cabinet_variant_metadata = None
    virtual_studio_metadata = None
    virtual_studio_enabled = bool(
        args.virtual_mic_b
        or args.virtual_room_preset != "off"
        or float(args.virtual_speaker_overload) > 0.0
        or abs(float(args.virtual_variphi_ms)) > 1e-9
        or bool(args.virtual_invert_mic_b)
        or abs(float(args.virtual_mic_a_level_db)) > 1e-9
        or abs(float(args.virtual_mic_b_level_db)) > 1e-9
        or abs(float(args.virtual_mic_a_pan)) > 1e-9
        or abs(float(args.virtual_mic_b_pan)) > 1e-9
    )
    if virtual_studio_enabled:
        from virtual_studio_workflow import apply_virtual_studio_audio

        output, virtual_studio_metadata = apply_virtual_studio_audio(
            output,
            output_rate,
            mic_a_profile=Path(args.cabinet_variant) if args.cabinet_variant else None,
            mic_b_profile=Path(args.virtual_mic_b) if args.virtual_mic_b else None,
            mic_a_variant_mix=float(args.cabinet_mix),
            mic_morph=float(args.virtual_mic_morph),
            mic_a_level_db=float(args.virtual_mic_a_level_db),
            mic_b_level_db=float(args.virtual_mic_b_level_db),
            mic_a_pan=float(args.virtual_mic_a_pan),
            mic_b_pan=float(args.virtual_mic_b_pan),
            variphi_ms=float(args.virtual_variphi_ms),
            invert_mic_b=bool(args.virtual_invert_mic_b),
            room_preset=str(args.virtual_room_preset),
            distance=float(args.virtual_distance),
            room_mix=float(args.virtual_room_mix),
            overload=float(args.virtual_speaker_overload),
            low_cut_hz=float(args.cabinet_low_cut_hz),
            high_cut_hz=float(args.cabinet_high_cut_hz),
        )
        if args.cabinet_variant:
            cabinet_variant_metadata = virtual_studio_metadata["mic_a"]
    elif args.cabinet_variant:
        from cabinet_variant_workflow import apply_cabinet_variant_audio

        output, cabinet_variant_metadata = apply_cabinet_variant_audio(
            output,
            output_rate,
            Path(args.cabinet_variant),
            mix=float(args.cabinet_mix),
            low_cut_hz=float(args.cabinet_low_cut_hz),
            high_cut_hz=float(args.cabinet_high_cut_hz),
        )
    if float(np.max(np.abs(output))) >= 1.0:
        if args.limiter == "off":
            raise SystemExit("Rig model output clips. Reduce --input-trim-db or --output-trim-db.")
        output = np.tanh(output) / np.tanh(1.0)
        output *= 0.98
    write_audio(Path(args.output), output_rate, output)
    print(f"Applied calibrated rig model: {args.model}")
    print(f"Input: {args.input}")
    print(
        f"Model input calibration: stored={stored_trim_db:+.1f} dB "
        f"additional={float(args.input_trim_db):+.1f} dB effective={effective_trim_db:+.1f} dB"
    )
    if cabinet_variant_metadata is not None:
        variant = dict(cabinet_variant_metadata.get("variant", {}))
        print(
            f"Cabinet variant: {cabinet_variant_metadata.get('name', args.cabinet_variant)} | "
            f"{variant.get('cabinet', '')} | {variant.get('microphone', '')} | "
            f"{variant.get('mic_position', '')} | mix={float(args.cabinet_mix):.2f}"
        )
    if virtual_studio_metadata is not None:
        controls = virtual_studio_metadata["controls"]
        mic_b = virtual_studio_metadata.get("mic_b")
        print(
            f"Virtual studio: mic_B={mic_b.get('name', args.virtual_mic_b) if mic_b else 'off'} | "
            f"morph={controls['mic_morph']:.2f} | Variphi={controls['variphi_ms']:+.3f} ms | "
            f"room={controls['room_preset']}/{controls['effective_room_mix']:.2f} | "
            f"overload={controls['speaker_overload']:.2f}"
        )
        print("Virtual power amp: off (the captured rig already includes the Peavey power amp)")
    print(f"Output: {args.output} | {output_rate} Hz | peak={peak_dbfs(output):.2f} dBFS")
