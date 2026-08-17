#!/usr/bin/env python3
"""Measured cabinet/microphone variants for controlled rig captures."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from rig_capture_workflow import (
    align_calibrated_pair,
    json_read,
    peak_dbfs,
    read_audio,
    remove_dc,
    resample_audio,
    rms,
    spectral_error,
    write_audio,
)


CABINET_VARIANT_VERSION = "measured_cabinet_variant_1.0"


def mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float64)
    return array[:, 0] if array.ndim == 2 else array


def correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    length = min(len(reference), len(candidate))
    if length < 2:
        return 0.0
    reference = reference[:length]
    candidate = candidate[:length]
    if float(np.std(reference)) < 1e-10 or float(np.std(candidate)) < 1e-10:
        return 0.0
    return float(np.corrcoef(reference, candidate)[0, 1])


def level_match(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    gain = rms(reference) / max(rms(candidate), 1e-12)
    return np.asarray(candidate * gain, dtype=np.float64)


def analysis_section(probe_manifest: dict, length: int) -> tuple[int, int, str]:
    sections = list(probe_manifest.get("sections", []))
    sweeps = [
        section
        for section in sections
        if str(section.get("label", "")).startswith("log_sweep_") and section.get("peak_dbfs") is not None
    ]
    if sweeps:
        selected = min(sweeps, key=lambda section: float(section.get("peak_dbfs", 0.0)))
        start = max(0, int(selected.get("start_sample", 0)))
        end = min(length, int(selected.get("end_sample", length)))
        if end - start >= 8192:
            return start, end, str(selected.get("label", "low_level_sweep"))
    end = min(length, 262144)
    return 0, end, "capture_start_fallback"


def validation_section(probe_manifest: dict, length: int) -> tuple[int, int, str]:
    for section in probe_manifest.get("sections", []):
        if str(section.get("role", "")) == "validation":
            start = max(0, int(section.get("start_sample", 0)))
            end = min(length, int(section.get("end_sample", length)))
            if end - start >= 2048:
                return start, end, str(section.get("label", "validation"))
    start = int(round(length * 0.80))
    return start, length, "capture_tail_fallback"


def minimum_phase_impulse(magnitude: np.ndarray, fft_size: int, fir_length: int) -> np.ndarray:
    magnitude = np.maximum(np.asarray(magnitude, dtype=np.float64), 1e-6)
    log_half = np.log(magnitude)
    log_full = np.concatenate([log_half, log_half[-2:0:-1]])
    cepstrum = np.fft.ifft(log_full).real
    minimum_cepstrum = np.zeros(fft_size, dtype=np.float64)
    minimum_cepstrum[0] = cepstrum[0]
    minimum_cepstrum[1 : fft_size // 2] = 2.0 * cepstrum[1 : fft_size // 2]
    minimum_cepstrum[fft_size // 2] = cepstrum[fft_size // 2]
    spectrum = np.exp(np.fft.fft(minimum_cepstrum))
    impulse = np.fft.ifft(spectrum).real[:fir_length]
    fade_count = max(8, fir_length // 5)
    impulse[-fade_count:] *= np.linspace(1.0, 0.0, fade_count)
    return impulse


def extract_cabinet_variant(
    probe: np.ndarray,
    reference: np.ndarray,
    variant: np.ndarray,
    sample_rate: int,
    probe_manifest: dict,
    fft_size: int,
    fir_length: int,
    smoothing_bins: int,
    max_correction_db: float,
    level_mode: str,
) -> tuple[np.ndarray, dict]:
    length = min(len(probe), len(reference), len(variant))
    start, end, section_label = analysis_section(probe_manifest, length)
    source_part = remove_dc(probe[start:end])
    reference_part = remove_dc(reference[start:end])
    variant_part = remove_dc(variant[start:end])
    if len(source_part) < 8192:
        raise SystemExit("Cabinet variant extraction needs at least 8192 aligned probe samples.")
    if fft_size < 16384 or fft_size & (fft_size - 1):
        raise SystemExit("--fft-size must be a power of two and at least 16384.")
    required_fft_size = 1 << int(np.ceil(np.log2(len(source_part))))
    fft_size = max(int(fft_size), int(required_fft_size))
    if fir_length < 256 or fir_length > fft_size // 2:
        raise SystemExit("--fir-length must be between 256 and half of --fft-size.")

    window_length = len(source_part)
    window = np.hanning(window_length)
    source_fft = np.fft.rfft(source_part[:window_length] * window, n=fft_size)
    reference_fft = np.fft.rfft(reference_part[:window_length] * window, n=fft_size)
    variant_fft = np.fft.rfft(variant_part[:window_length] * window, n=fft_size)
    source_power = np.square(np.abs(source_fft))
    regularizer = max(float(np.max(source_power)) * 1e-7, 1e-18)
    reference_transfer = reference_fft * np.conj(source_fft) / (source_power + regularizer)
    variant_transfer = variant_fft * np.conj(source_fft) / (source_power + regularizer)
    transfer_floor = max(float(np.max(np.abs(reference_transfer))) * 1e-5, 1e-10)
    raw_ratio = (np.abs(variant_transfer) + transfer_floor) / (np.abs(reference_transfer) + transfer_floor)
    correction_db = 20.0 * np.log10(np.maximum(raw_ratio, 1e-8))

    smoothing_bins = max(1, int(smoothing_bins))
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1
    kernel = np.ones(smoothing_bins, dtype=np.float64) / smoothing_bins
    correction_db = np.convolve(correction_db, kernel, mode="same")
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    audible = (frequencies >= 70.0) & (frequencies <= min(12000.0, sample_rate * 0.45))
    raw_broadband_gain_db = float(np.median(correction_db[audible])) if np.any(audible) else 0.0
    if level_mode == "tone-only":
        correction_db -= raw_broadband_gain_db
    elif level_mode != "preserve":
        raise SystemExit("--level-mode must be tone-only or preserve.")
    correction_db = np.clip(correction_db, -float(max_correction_db), float(max_correction_db))
    correction_magnitude = np.power(10.0, correction_db / 20.0)
    impulse = minimum_phase_impulse(correction_magnitude, fft_size=fft_size, fir_length=fir_length)

    realized = np.abs(np.fft.rfft(impulse, n=fft_size)) + 1e-12
    realized_db = 20.0 * np.log10(realized)
    realized_center_db = float(np.median(realized_db[audible])) if np.any(audible) else 0.0
    desired_center_db = float(np.median(correction_db[audible])) if np.any(audible) else 0.0
    impulse *= np.power(10.0, (desired_center_db - realized_center_db) / 20.0)

    peak_index = int(np.argmax(np.abs(impulse)))
    details = {
        "analysis_section": section_label,
        "fft_size": int(fft_size),
        "frequency_points": int(len(frequencies)),
        "fir_length_samples": int(fir_length),
        "fir_length_ms": float(1000.0 * fir_length / sample_rate),
        "impulse_peak_sample": peak_index,
        "raw_broadband_gain_db": raw_broadband_gain_db,
        "level_mode": level_mode,
        "maximum_correction_db": float(np.max(np.abs(correction_db))),
        "correction_rms_db": float(np.sqrt(np.mean(np.square(correction_db[audible])))) if np.any(audible) else 0.0,
    }
    return impulse.astype(np.float64), details


def load_cabinet_variant(path: Path) -> tuple[dict, np.ndarray]:
    if not path.exists():
        raise SystemExit(f"Cabinet variant profile not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("profile_version") != CABINET_VARIANT_VERSION:
            raise SystemExit(f"Unsupported cabinet variant version: {metadata.get('profile_version')}")
        impulse = archive["impulse_response"].astype(np.float64)
    return metadata, impulse


def save_cabinet_variant(path: Path, metadata: dict, impulse: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=json.dumps(metadata, indent=2),
        impulse_response=np.asarray(impulse, dtype=np.float32),
    )


def filter_variant_output(audio: np.ndarray, sample_rate: int, low_cut_hz: float, high_cut_hz: float) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float64)
    if low_cut_hz > 0.0:
        if low_cut_hz >= sample_rate * 0.45:
            raise SystemExit("--low-cut-hz must be below Nyquist.")
        output = sosfilt(butter(2, low_cut_hz, btype="highpass", fs=sample_rate, output="sos"), output)
    if high_cut_hz > 0.0:
        if high_cut_hz >= sample_rate * 0.49:
            raise SystemExit("--high-cut-hz must be below Nyquist.")
        output = sosfilt(butter(2, high_cut_hz, btype="lowpass", fs=sample_rate, output="sos"), output)
    return output


def apply_cabinet_variant_audio(
    audio: np.ndarray,
    sample_rate: int,
    profile_path: Path,
    mix: float = 1.0,
    low_cut_hz: float = 0.0,
    high_cut_hz: float = 0.0,
) -> tuple[np.ndarray, dict]:
    metadata, impulse = load_cabinet_variant(profile_path)
    profile_rate = int(metadata["sample_rate_hz"])
    if profile_rate != sample_rate:
        impulse = resample_audio(impulse, profile_rate, sample_rate)
    dry = np.asarray(audio, dtype=np.float64)
    wet = fftconvolve(dry, impulse, mode="full")[: len(dry)]
    wet = filter_variant_output(wet, sample_rate, float(low_cut_hz), float(high_cut_hz))
    amount = float(np.clip(mix, 0.0, 1.0))
    output = ((1.0 - amount) * dry) + (amount * wet)
    return output, metadata


def run_build_cabinet_variant(args) -> None:
    probe_rate, probe = read_audio(Path(args.probe))
    reference_rate, reference = read_audio(Path(args.reference_target))
    variant_rate, variant = read_audio(Path(args.variant_target))
    if not (probe_rate == reference_rate == variant_rate):
        raise SystemExit("Probe, reference return, and variant return must use the same sample rate.")
    probe, reference, variant = mono(probe), mono(reference), mono(variant)
    probe_manifest = json_read(Path(args.probe_manifest) if args.probe_manifest else None)
    if probe_manifest.get("probe_manifest"):
        probe_manifest = dict(probe_manifest["probe_manifest"])
    probe_reference, reference, reference_latency, reference_polarity = align_calibrated_pair(
        probe, reference, probe_rate
    )
    probe_variant, variant, variant_latency, variant_polarity = align_calibrated_pair(probe, variant, probe_rate)
    length = min(len(probe_reference), len(probe_variant), len(reference), len(variant))
    probe_aligned = probe_reference[:length]
    reference = reference[:length]
    variant = variant[:length]

    impulse, extraction = extract_cabinet_variant(
        probe_aligned,
        reference,
        variant,
        sample_rate=probe_rate,
        probe_manifest=probe_manifest,
        fft_size=int(args.fft_size),
        fir_length=int(args.fir_length),
        smoothing_bins=int(args.smoothing_bins),
        max_correction_db=float(args.max_correction_db),
        level_mode=str(args.level_mode),
    )
    corrected = fftconvolve(reference, impulse, mode="full")[:length]
    validation_start, validation_end, validation_label = validation_section(probe_manifest, length)
    reference_validation = reference[validation_start:validation_end]
    variant_validation = variant[validation_start:validation_end]
    corrected_validation = corrected[validation_start:validation_end]
    if args.level_mode == "tone-only":
        reference_for_metrics = level_match(reference_validation, variant_validation)
        corrected_for_metrics = level_match(corrected_validation, variant_validation)
    else:
        reference_for_metrics = reference_validation
        corrected_for_metrics = corrected_validation
    baseline_spectral = spectral_error(variant_validation, reference_for_metrics, probe_rate)
    corrected_spectral = spectral_error(variant_validation, corrected_for_metrics, probe_rate)
    improvement_db = float(baseline_spectral - corrected_spectral)
    metrics = {
        "validation_section": validation_label,
        "baseline_spectral_error_db": baseline_spectral,
        "corrected_spectral_error_db": corrected_spectral,
        "spectral_improvement_db": improvement_db,
        "baseline_correlation": correlation(variant_validation, reference_for_metrics),
        "corrected_correlation": correlation(variant_validation, corrected_for_metrics),
    }
    failures = []
    if improvement_db < float(args.min_spectral_improvement_db):
        failures.append(
            f"spectral improvement {improvement_db:.2f} dB < {float(args.min_spectral_improvement_db):.2f} dB"
        )
    if metrics["corrected_correlation"] < metrics["baseline_correlation"] - 0.03:
        failures.append("corrected cabinet response reduced held-out correlation by more than 0.03")
    accepted = not failures
    metadata = {
        "profile_version": CABINET_VARIANT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": str(args.name),
        "sample_rate_hz": int(probe_rate),
        "reference": {
            "target": str(args.reference_target),
            "cabinet": str(args.reference_cabinet),
            "microphone": str(args.reference_microphone),
            "mic_position": str(args.reference_mic_position),
            "latency_samples": reference_latency,
            "polarity": reference_polarity,
        },
        "variant": {
            "target": str(args.variant_target),
            "cabinet": str(args.variant_cabinet),
            "microphone": str(args.variant_microphone),
            "mic_position": str(args.variant_mic_position),
            "mic_axis": str(args.variant_mic_axis),
            "latency_samples": variant_latency,
            "polarity": variant_polarity,
        },
        "extraction": extraction,
        "validation": {**metrics, "accepted": accepted, "failures": failures},
    }
    requested_path = Path(args.profile)
    profile_path = requested_path
    if not accepted and not args.allow_failed_validation:
        profile_path = requested_path.with_name(f"{requested_path.stem}.rejected{requested_path.suffix}")
    save_cabinet_variant(profile_path, metadata, impulse)

    if args.comparison_output:
        seconds = int(round(float(args.comparison_seconds) * probe_rate))
        count = min(seconds, len(reference_validation), len(variant_validation), len(corrected_validation))
        segments = [reference_validation[:count], variant_validation[:count], corrected_validation[:count]]
        common_peak = max(max(float(np.max(np.abs(item))), 1e-12) for item in segments)
        scale = min(1.0, 0.98 / common_peak)
        silence = np.zeros(int(round(0.75 * probe_rate)), dtype=np.float64)
        comparison = np.concatenate(
            [segments[0] * scale, silence, segments[1] * scale, silence, segments[2] * scale]
        )
        write_audio(Path(args.comparison_output), probe_rate, comparison)

    print(
        f"Cabinet variant validation: accepted={accepted} spectral={baseline_spectral:.2f}->"
        f"{corrected_spectral:.2f} dB improvement={improvement_db:.2f} dB "
        f"corr={metrics['baseline_correlation']:.3f}->{metrics['corrected_correlation']:.3f}"
    )
    print(
        f"Measured correction: points={extraction['frequency_points']} FIR={extraction['fir_length_samples']} "
        f"samples/{extraction['fir_length_ms']:.1f} ms raw_gain={extraction['raw_broadband_gain_db']:+.2f} dB"
    )
    print(f"Cabinet variant profile: {profile_path}")
    if args.comparison_output:
        print(f"Comparison order: reference cabinet/mic, recorded variant, synthesized variant: {args.comparison_output}")
    if failures:
        print("Cabinet variant failures:")
        for failure in failures:
            print(f"  - {failure}")
    if not accepted and not args.allow_failed_validation:
        raise SystemExit("Cabinet variant was rejected; the requested production profile was not written.")


def run_apply_cabinet_variant(args) -> None:
    sample_rate, audio = read_audio(Path(args.input))
    audio = mono(audio)
    output, metadata = apply_cabinet_variant_audio(
        audio,
        sample_rate,
        Path(args.profile),
        mix=float(args.mix),
        low_cut_hz=float(args.low_cut_hz),
        high_cut_hz=float(args.high_cut_hz),
    )
    output *= np.power(10.0, float(args.output_trim_db) / 20.0)
    peak = float(np.max(np.abs(output)) + 1e-12)
    if peak >= 1.0:
        if args.limiter == "off":
            raise SystemExit("Cabinet variant output clips. Reduce --output-trim-db or --mix.")
        output = np.tanh(output) / np.tanh(1.0)
        output *= 0.98
    write_audio(Path(args.output), sample_rate, output)
    print(f"Applied cabinet/mic variant: {metadata.get('name', args.profile)}")
    print(
        f"Variant: {metadata.get('variant', {}).get('cabinet', '')} | "
        f"{metadata.get('variant', {}).get('microphone', '')} | "
        f"{metadata.get('variant', {}).get('mic_position', '')}"
    )
    print(f"Output: {args.output} | mix={float(args.mix):.2f} | peak={peak_dbfs(output):.2f} dBFS")


def run_build_separated_cabinet(args) -> None:
    preamp_manifest_path = Path(args.preamp_capture_manifest)
    amp_cab_manifest_path = Path(args.amp_cab_capture_manifest)
    preamp = json_read(preamp_manifest_path)
    amp_cab = json_read(amp_cab_manifest_path)
    if not preamp or not amp_cab:
        raise SystemExit("Both separated-cabinet capture manifests must exist and contain JSON objects.")
    if str(preamp.get("capture_type", "")) != "amp-preamp":
        raise SystemExit("The reference capture must use --capture-type amp-preamp with a safe line-level return.")
    if str(amp_cab.get("capture_type", "")) != "amp-cab":
        raise SystemExit("The cabinet capture must use --capture-type amp-cab with the recorded microphone return.")
    if int(preamp.get("sample_rate_hz", 0)) != int(amp_cab.get("sample_rate_hz", 0)):
        raise SystemExit("Preamp and amp/cab captures must use the same sample rate.")

    preamp_probe = dict(preamp.get("probe_manifest", {}))
    amp_cab_probe = dict(amp_cab.get("probe_manifest", {}))
    probe_keys = ("probe_version", "sample_rate_hz", "seed", "duration_s")
    mismatched_probe = [key for key in probe_keys if preamp_probe.get(key) != amp_cab_probe.get(key)]
    if mismatched_probe:
        raise SystemExit(
            "Separated captures must use the identical calibrated probe; mismatched fields: "
            + ", ".join(mismatched_probe)
        )

    preamp_rig = dict(preamp.get("rig", {}))
    amp_cab_rig = dict(amp_cab.get("rig", {}))
    fixed_fields = ("pedal", "amp", "amp_settings")
    mismatched_rig = [key for key in fixed_fields if str(preamp_rig.get(key, "")) != str(amp_cab_rig.get(key, ""))]
    if mismatched_rig:
        raise SystemExit(
            "Amp, pedal, and controls must remain fixed between preamp and cabinet captures; mismatched fields: "
            + ", ".join(mismatched_rig)
        )
    preamp_trim = float(dict(preamp.get("routing", {})).get("send_trim_db", 0.0))
    amp_cab_trim = float(dict(amp_cab.get("routing", {})).get("send_trim_db", 0.0))
    if abs(preamp_trim - amp_cab_trim) > 1e-6:
        raise SystemExit("Separated captures must use the same reamp send trim.")
    if not bool(preamp.get("capture_valid", False)) or not bool(amp_cab.get("capture_valid", False)):
        raise SystemExit("Both capture manifests must pass level, clipping, and stream validation.")

    delegated = SimpleNamespace(
        probe=Path(str(preamp["probe_input_wav"])),
        reference_target=Path(str(preamp["target_return_wav"])),
        variant_target=Path(str(amp_cab["target_return_wav"])),
        probe_manifest=preamp_manifest_path,
        profile=Path(args.profile),
        name=str(args.name),
        reference_cabinet="cabinet bypassed / approved preamp line return",
        reference_microphone="line return",
        reference_mic_position="post-preamp line level",
        variant_cabinet=str(amp_cab_rig.get("cabinet", "unlabeled cabinet")),
        variant_microphone=str(amp_cab_rig.get("microphone", "unlabeled microphone")),
        variant_mic_position=str(amp_cab_rig.get("mic_position", "unlabeled position")),
        variant_mic_axis=str(args.mic_axis),
        fft_size=int(args.fft_size),
        fir_length=int(args.fir_length),
        smoothing_bins=int(args.smoothing_bins),
        max_correction_db=float(args.max_correction_db),
        level_mode=str(args.level_mode),
        min_spectral_improvement_db=float(args.min_spectral_improvement_db),
        comparison_output=Path(args.comparison_output) if args.comparison_output else None,
        comparison_seconds=float(args.comparison_seconds),
        allow_failed_validation=bool(args.allow_failed_validation),
    )
    run_build_cabinet_variant(delegated)
