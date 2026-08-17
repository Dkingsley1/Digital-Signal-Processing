#!/usr/bin/env python3
"""PyTorch/NAM research worker kept outside the production MLX interpreter."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def ensure_tone_only_paths(paths: list[Path]) -> None:
    for path in paths:
        if "schwab_trading_bot" in str(path).lower():
            raise SystemExit("Refusing a Schwab project path in the isolated tone-modeling worker.")


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    import soundfile as sf

    if not path.exists():
        raise SystemExit(f"Audio file not found: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return int(sample_rate), np.nan_to_num(np.asarray(audio, dtype=np.float32))


def write_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    peak = float(np.max(np.abs(clean)) + 1e-12)
    if peak > 1.0:
        clean = clean / peak * 0.98
    sf.write(path, clean, sample_rate, subtype="FLOAT")


def limited_audio(audio: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    peak = float(np.max(np.abs(clean)) + 1e-12)
    return clean * (peak_limit / peak) if peak > peak_limit else clean


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(audio, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def align(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, int]:
    from scipy.signal import correlate

    length = min(len(reference), len(candidate), int(sample_rate * 12.0))
    if length < 512:
        raise SystemExit("Audio is too short for aligned research metrics.")
    decimation = max(1, sample_rate // 12000)
    ref_probe = reference[:length:decimation]
    cand_probe = candidate[:length:decimation]
    correlation = correlate(cand_probe, ref_probe, mode="full", method="fft")
    lags = np.arange(-len(ref_probe) + 1, len(cand_probe))
    limit = max(1, int(0.25 * sample_rate / decimation))
    allowed = np.abs(lags) <= limit
    lag = int(lags[allowed][np.argmax(np.abs(correlation[allowed]))] * decimation)
    if lag > 0:
        candidate = candidate[lag:]
    elif lag < 0:
        reference = reference[-lag:]
    final = min(len(reference), len(candidate))
    return reference[:final], candidate[:final], lag


def spectral_error(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    length = min(len(reference), len(candidate))
    fft_size = min(16384, 2 ** int(math.floor(math.log2(max(1024, length)))))
    hop = fft_size // 4
    errors = []
    window = np.hanning(fft_size).astype(np.float32)
    starts = np.arange(0, max(1, length - fft_size + 1), hop, dtype=np.int64)
    if len(starts) > 96:
        starts = starts[np.linspace(0, len(starts) - 1, 96, dtype=np.int64)]
    for start in starts:
        ref_mag = np.abs(np.fft.rfft(reference[start : start + fft_size] * window)) + 1e-7
        cand_mag = np.abs(np.fft.rfft(candidate[start : start + fft_size] * window)) + 1e-7
        errors.append(float(np.mean(np.square(20.0 * np.log10(cand_mag / ref_mag)))))
    return float(math.sqrt(np.mean(errors))) if errors else float("inf")


def distributed_excerpt(audio: np.ndarray, total_samples: int, segments: int = 5) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32)
    if len(values) <= total_samples:
        return values
    segment_length = max(1, total_samples // max(1, segments))
    starts = np.linspace(0, len(values) - segment_length, segments, dtype=np.int64)
    return np.concatenate([values[start : start + segment_length] for start in starts])


def multiresolution_stft(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    import auraloss
    import torch

    limit = min(len(reference), len(candidate))
    excerpt_samples = min(limit, int(sample_rate * 10.0))
    ref_excerpt = distributed_excerpt(reference[:limit], excerpt_samples)
    candidate_excerpt = distributed_excerpt(candidate[:limit], excerpt_samples)
    ref = torch.from_numpy(ref_excerpt).reshape(1, 1, -1)
    cand = torch.from_numpy(candidate_excerpt).reshape(1, 1, -1)
    losses = build_perceptual_losses(auraloss, sample_rate, torch.device("cpu"), limit)
    with torch.no_grad():
        linear = losses["linear_stft"](cand, ref)
        mel = mel_perceptual_mrstft(cand, ref, losses)
        return float((0.5 * linear + 0.5 * mel).item())


def signal_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(audio, dtype=np.float64))) + 1e-12))


def listening_sections(target: np.ndarray, sample_rate: int, source: np.ndarray | None = None) -> list[dict]:
    window = max(1024, int(round(sample_rate * 1.0)))
    hop = max(512, window // 2)
    if len(target) <= window:
        starts = np.asarray([0], dtype=np.int64)
        window = len(target)
    else:
        starts = np.arange(0, len(target) - window + 1, hop, dtype=np.int64)
    features = []
    for start in starts:
        segment = np.asarray(target[int(start) : int(start) + window], dtype=np.float32)
        source_segment = (
            np.asarray(source[int(start) : int(start) + window], dtype=np.float32)
            if source is not None
            else segment
        )
        rms_value = signal_rms(segment)
        source_rms = signal_rms(source_segment)
        transient_ratio = signal_rms(np.diff(segment)) / max(rms_value, 1e-7)
        crest = float(np.max(np.abs(segment)) / max(rms_value, 1e-7))
        features.append(
            {
                "start_sample": int(start),
                "samples": int(len(segment)),
                "target_rms": rms_value,
                "source_rms": source_rms,
                "target_rms_dbfs": float(20.0 * np.log10(rms_value + 1e-12)),
                "target_transient_ratio": transient_ratio,
                "target_crest_factor": crest,
            }
        )
    if not features:
        return []
    peak_rms = max(item["target_rms"] for item in features)
    peak_source_rms = max(item["source_rms"] for item in features)
    active = [
        item
        for item in features
        if item["target_rms"] >= peak_rms * 0.05 and item["source_rms"] >= peak_source_rms * 0.08
    ]
    if not active:
        active = features
    by_level = sorted(active, key=lambda item: item["target_rms"])
    medium_level = float(np.median([item["target_rms"] for item in active]))
    selections = {
        "quiet": by_level[0],
        "medium": min(active, key=lambda item: abs(item["target_rms"] - medium_level)),
        "loud": by_level[-1],
        "transient": max(active, key=lambda item: item["target_transient_ratio"]),
        "sustained": min(
            [item for item in active if item["target_rms"] >= medium_level] or active,
            key=lambda item: item["target_transient_ratio"],
        ),
    }
    return [{"label": label, **selection} for label, selection in selections.items()]


def listening_section_metrics(
    target: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    sections: list[dict],
    max_spectral_error_db: float,
    min_correlation: float,
    max_level_error_db: float,
) -> list[dict]:
    results = []
    for section in sections:
        start = int(section["start_sample"])
        end = min(len(target), len(candidate), start + int(section["samples"]))
        target_segment = target[start:end]
        candidate_segment = candidate[start:end]
        target_rms = signal_rms(target_segment)
        candidate_rms = signal_rms(candidate_segment)
        level_error = float(20.0 * np.log10((candidate_rms + 1e-12) / (target_rms + 1e-12)))
        correlation = (
            float(np.corrcoef(candidate_segment, target_segment)[0, 1])
            if np.std(candidate_segment) > 1e-8 and np.std(target_segment) > 1e-8
            else 0.0
        )
        envelope_relative = float(
            np.mean(np.abs(np.abs(candidate_segment) - np.abs(target_segment)))
            / (np.mean(np.abs(target_segment)) + 1e-7)
        )
        target_difference = np.diff(target_segment)
        candidate_difference = np.diff(candidate_segment)
        transient_relative = float(
            np.mean(np.abs(candidate_difference - target_difference))
            / (np.mean(np.abs(target_difference)) + 1e-7)
        )
        section_spectral_error = spectral_error(target_segment, candidate_segment, sample_rate)
        passes = bool(
            section_spectral_error <= max_spectral_error_db + 4.0
            and correlation >= min_correlation - 0.20
            and abs(level_error) <= max_level_error_db
            and envelope_relative <= 1.0
            and transient_relative <= 1.25
        )
        results.append(
            {
                **section,
                "spectral_error_db": section_spectral_error,
                "correlation": correlation,
                "level_error_db": level_error,
                "envelope_relative_error": envelope_relative,
                "transient_relative_error": transient_relative,
                "passes": passes,
            }
        )
    return results


def prepare_metric_audio(
    di: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    di, target, di_lag = align(di, target, sample_rate)
    target, candidate, candidate_lag = align(target, candidate, sample_rate)
    if candidate_lag < 0:
        di = di[-candidate_lag:]
    length = min(len(di), len(target), len(candidate))
    return di[:length], target[:length], candidate[:length], di_lag, candidate_lag


def candidate_metrics(
    di: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    min_improvement_db: float,
    min_movement_db: float,
    max_listening_spectral_error_db: float = 14.0,
    min_listening_correlation: float = 0.50,
    max_listening_level_error_db: float = 2.0,
    min_listening_section_pass_rate: float = 0.60,
) -> dict:
    di = np.nan_to_num(np.asarray(di, dtype=np.float32))
    target = np.nan_to_num(np.asarray(target, dtype=np.float32))
    candidate = np.nan_to_num(np.asarray(candidate, dtype=np.float32))
    di, target, candidate, di_lag, candidate_lag = prepare_metric_audio(di, target, candidate, sample_rate)
    gain = float(np.dot(di, target) / (np.dot(di, di) + 1e-12))
    gain_only = di * gain
    candidate_gain = float(np.dot(candidate, target) / (np.dot(candidate, candidate) + 1e-12))
    tone_candidate = candidate * candidate_gain
    candidate_error = spectral_error(target, tone_candidate, sample_rate)
    gain_error = spectral_error(target, gain_only, sample_rate)
    movement = spectral_error(gain_only, tone_candidate, sample_rate)
    improvement = gain_error - candidate_error
    error = candidate - target
    esr = float(np.mean(np.square(error)) / (np.mean(np.square(target)) + 1e-12))
    correlation = float(np.corrcoef(candidate, target)[0, 1]) if np.std(candidate) > 1e-8 else 0.0
    transient = float(np.mean(np.abs(np.diff(tone_candidate) - np.diff(target))))
    envelope = float(np.mean(np.abs(np.abs(tone_candidate) - np.abs(target))))
    level_correction_db = float(20.0 * np.log10(abs(candidate_gain) + 1e-12))
    section_definitions = listening_sections(target, sample_rate, source=di)
    section_results = listening_section_metrics(
        target,
        tone_candidate,
        sample_rate,
        section_definitions,
        max_spectral_error_db=max_listening_spectral_error_db,
        min_correlation=min_listening_correlation,
        max_level_error_db=max_listening_level_error_db,
    )
    section_pass_rate = float(np.mean([item["passes"] for item in section_results])) if section_results else 0.0
    amp_tone_guard_passed = bool(improvement >= min_improvement_db and movement >= min_movement_db)
    listening_failures = []
    if candidate_error > max_listening_spectral_error_db:
        listening_failures.append("aggregate spectral error")
    if correlation < min_listening_correlation:
        listening_failures.append("aggregate correlation")
    if abs(level_correction_db) > max_listening_level_error_db:
        listening_failures.append("raw output level")
    if candidate_gain <= 0.0:
        listening_failures.append("output polarity")
    if section_pass_rate < min_listening_section_pass_rate:
        listening_failures.append("listening section pass rate")
    listening_promotion_ready = bool(amp_tone_guard_passed and not listening_failures)
    mrstft_loss = multiresolution_stft(target, tone_candidate, sample_rate)
    spectral_score = float(np.clip(1.0 - candidate_error / 30.0, 0.0, 1.0))
    correlation_score = float(np.clip(correlation, 0.0, 1.0))
    perceptual_score = float(1.0 / (1.0 + max(0.0, mrstft_loss)))
    audible_match_score = float(
        100.0
        * (
            0.30 * spectral_score
            + 0.25 * correlation_score
            + 0.20 * section_pass_rate
            + 0.15 * perceptual_score
            + 0.10 * float(amp_tone_guard_passed)
        )
    )
    return {
        "esr": esr,
        "correlation": correlation,
        "spectral_error_db": candidate_error,
        "gain_only_spectral_error_db": gain_error,
        "spectral_improvement_over_gain_only_db": improvement,
        "movement_from_gain_only_db": movement,
        "multi_resolution_stft_loss": mrstft_loss,
        "candidate_level_match_gain": candidate_gain,
        "candidate_level_correction_db": level_correction_db,
        "transient_l1": transient,
        "envelope_l1": envelope,
        "candidate_alignment_lag_samples": candidate_lag,
        "di_target_alignment_lag_samples": di_lag,
        "amp_tone_guard_passed": amp_tone_guard_passed,
        "listening_promotion_ready": listening_promotion_ready,
        "audible_match_score_0_100": audible_match_score,
        "listening_section_pass_rate": section_pass_rate,
        "listening_sections": section_results,
        "listening_failures": listening_failures,
        "guard_thresholds": {
            "minimum_spectral_improvement_db": min_improvement_db,
            "minimum_movement_from_gain_only_db": min_movement_db,
            "maximum_listening_spectral_error_db": max_listening_spectral_error_db,
            "minimum_listening_correlation": min_listening_correlation,
            "maximum_listening_level_error_db": max_listening_level_error_db,
            "minimum_listening_section_pass_rate": min_listening_section_pass_rate,
        },
    }


def write_dry_listening_auditions(
    output_dir: Path,
    candidate_name: str,
    di: np.ndarray,
    target: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    metrics: dict,
    seconds: float,
) -> list[dict]:
    _, aligned_target, aligned_candidate, _, _ = prepare_metric_audio(di, target, candidate, sample_rate)
    tone_candidate = aligned_candidate * float(metrics["candidate_level_match_gain"])
    clip_samples = max(1, int(round(seconds * sample_rate)))
    silence = np.zeros(max(1, int(round(0.35 * sample_rate))), dtype=np.float32)
    safe_name = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in candidate_name)
    candidate_dir = output_dir / safe_name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for section in metrics.get("listening_sections", []):
        center = int(section["start_sample"]) + int(section["samples"]) // 2
        start = max(0, min(len(aligned_target) - clip_samples, center - clip_samples // 2))
        end = min(len(aligned_target), len(tone_candidate), start + clip_samples)
        target_clip = aligned_target[start:end]
        candidate_clip = tone_candidate[start:end]
        if len(target_clip) < clip_samples:
            padding = clip_samples - len(target_clip)
            target_clip = np.pad(target_clip, (0, padding))
            candidate_clip = np.pad(candidate_clip, (0, padding))
        combined = np.concatenate([target_clip, silence, candidate_clip])
        output = candidate_dir / f"AB_{section['label']}_A_real_then_B_model_{seconds:g}s_each.wav"
        write_audio(output, sample_rate, combined)
        written.append(
            {
                "label": section["label"],
                "output": str(output),
                "source_start_seconds": float(start / sample_rate),
                "A": "recorded amp/cab/microphone target",
                "B": "candidate model with one global level correction",
            }
        )
    manifest = {
        "format": "tone_capture_dry_listening_auditions_1.0",
        "candidate": candidate_name,
        "sample_rate_hz": sample_rate,
        "seconds_per_A_or_B": seconds,
        "level_correction_db": metrics.get("candidate_level_correction_db"),
        "render_policy": "strict dry close-mic endpoint; no room, reflection, extra cabinet, or section-by-section normalization",
        "auditions": written,
    }
    (candidate_dir / "audition_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written


def run_check(_: argparse.Namespace) -> None:
    import auraloss  # noqa: F401
    import nam  # noqa: F401
    import torch
    import torchaudio

    payload = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "auraloss": package_version("auraloss"),
        "neural_amp_modeler": package_version("neural-amp-modeler"),
        "nablafx": package_version("nablafx"),
        "nablafx_route": "recipes and explicit submodules; optional eager evaluation import disabled",
        "nablafx_dependency_note": "rational-activations is intentionally installed --no-deps per NablAFx guidance",
        "mps_available": bool(torch.backends.mps.is_available()),
        "runtime_device": "mps" if torch.backends.mps.is_available() else "cpu",
    }
    print(json.dumps(payload, indent=2))


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Candidate must use NAME=PATH format.")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Candidate must use NAME=PATH format.")
    return name.strip(), Path(raw_path).expanduser().resolve()


def run_metrics(args: argparse.Namespace) -> None:
    ensure_tone_only_paths([args.di, args.target, args.output, *[path for _, path in args.candidate]])
    di_rate, di = read_audio(args.di)
    target_rate, target = read_audio(args.target)
    target = resample(target, target_rate, di_rate)
    results = {}
    auditions = {}
    for name, path in args.candidate:
        candidate_rate, candidate = read_audio(path)
        candidate = resample(candidate, candidate_rate, di_rate)
        results[name] = candidate_metrics(
            di,
            target,
            candidate,
            di_rate,
            min_improvement_db=args.min_improvement_db,
            min_movement_db=args.min_movement_db,
            max_listening_spectral_error_db=args.max_listening_spectral_error_db,
            min_listening_correlation=args.min_listening_correlation,
            max_listening_level_error_db=args.max_listening_level_error_db,
            min_listening_section_pass_rate=args.min_listening_section_pass_rate,
        )
        if args.audition_dir:
            auditions[name] = write_dry_listening_auditions(
                args.audition_dir,
                name,
                di,
                target,
                candidate,
                di_rate,
                results[name],
                seconds=args.audition_seconds,
            )
    ranking = sorted(
        results,
        key=lambda name: (
            not results[name]["listening_promotion_ready"],
            -results[name]["audible_match_score_0_100"],
            results[name]["multi_resolution_stft_loss"],
            results[name]["spectral_error_db"],
        ),
    )
    payload = {
        "sample_rate_hz": di_rate,
        "di": str(args.di),
        "target": str(args.target),
        "ranking": ranking,
        "candidates": results,
        "dry_listening_auditions": auditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def run_nam_config(args: argparse.Namespace) -> None:
    import importlib.resources

    ensure_tone_only_paths([args.capture_manifest, args.output_dir])
    capture = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    files = capture["files"]
    validation_start = int(capture["split"]["validation_start_sample"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_pairs = list(files.get("nam_validation_pairs", []))
    if validation_pairs:
        validation_config = [
            {
                "x_path": str(pair["input"]),
                "y_path": str(pair["target"]),
                "ny": int(args.window_samples),
                "require_input_pre_silence": False,
            }
            for pair in validation_pairs
        ]
    else:
        validation_config = {
            "ny": int(args.window_samples),
            "start_samples": validation_start,
            "require_input_pre_silence": False,
        }
    data_config = {
        "train": {"ny": int(args.window_samples), "stop_samples": validation_start},
        "validation": validation_config,
        "common": {
            "x_path": str(files["aligned_input"]),
            "y_path": str(files["aligned_target"]),
            "delay": 0,
            "allow_unequal_lengths": False,
        },
        "joint": [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ],
    }
    resource = importlib.resources.files("nam.train._resources").joinpath("config_model_packed.json")
    with resource.open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    learning_config = {
        "train_dataloader": {
            "batch_size": int(args.batch_size),
            "shuffle": True,
            "pin_memory": False,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {
            "batch_size": int(args.batch_size),
            "num_workers": 0,
        },
        # NAM 0.13 pins a Lightning release whose MPS detector rejects this
        # newer Torch build. The custom PyTorch reference still uses MPS.
        "trainer": {
            "max_epochs": int(args.epochs),
            "accelerator": "cpu",
            "devices": 1,
            "num_sanity_val_steps": 0,
        },
    }
    if int(args.train_batches_per_epoch) > 0:
        learning_config["trainer"]["limit_train_batches"] = int(args.train_batches_per_epoch)
    if int(args.validation_batches_per_epoch) > 0:
        learning_config["trainer"]["limit_val_batches"] = int(args.validation_batches_per_epoch)
    paths = {
        "data": args.output_dir / "nam_data.json",
        "model": args.output_dir / "nam_model.json",
        "learning": args.output_dir / "nam_learning.json",
    }
    for name, payload in (("data", data_config), ("model", model_config), ("learning", learning_config)):
        paths[name].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


def left_pad_1d(x, amount: int):
    import torch.nn.functional as functional

    return functional.pad(x, (amount, 0))


def build_tcn(channels: int, levels: int, input_channels: int = 1):
    import torch

    class ResidualBlock(torch.nn.Module):
        def __init__(self, width: int, dilation: int):
            super().__init__()
            self.dilation = dilation
            self.filter = torch.nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
            self.gate = torch.nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
            self.mix = torch.nn.Conv1d(width, width, 1)

        def forward(self, x):
            padding = 2 * self.dilation
            padded = left_pad_1d(x, padding)
            active = torch.tanh(self.filter(padded)) * torch.sigmoid(self.gate(padded))
            return x + 0.25 * torch.tanh(self.mix(active))

    class CausalTCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Conv1d(input_channels, channels, 1)
            self.blocks = torch.nn.ModuleList([ResidualBlock(channels, 2**level) for level in range(levels)])
            self.output = torch.nn.Conv1d(channels, 1, 1)

        def forward(self, x):
            hidden = self.input(x)
            for block in self.blocks:
                hidden = block(hidden)
            return self.output(torch.tanh(hidden))

    return CausalTCN(), 1 + 2 * sum(2**level for level in range(levels))


def build_tcn_fullness(channels: int, levels: int, stacks: int = 2, input_channels: int = 1):
    import torch

    class FullnessBlock(torch.nn.Module):
        def __init__(self, width: int, dilation: int):
            super().__init__()
            self.dilation = dilation
            self.filter = torch.nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
            self.gate = torch.nn.Conv1d(width, width, 3, dilation=dilation, groups=width)
            self.residual = torch.nn.Conv1d(width, width, 1)
            self.skip = torch.nn.Conv1d(width, width, 1)

        def forward(self, x):
            padded = left_pad_1d(x, 2 * self.dilation)
            active = torch.tanh(self.filter(padded)) * torch.sigmoid(self.gate(padded))
            return x + 0.20 * self.residual(active), self.skip(active)

    class FullnessTCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Conv1d(input_channels, channels, 1)
            dilations = [2**level for _ in range(stacks) for level in range(levels)]
            self.blocks = torch.nn.ModuleList([FullnessBlock(channels, dilation) for dilation in dilations])
            self.output = torch.nn.Sequential(
                torch.nn.LeakyReLU(0.1),
                torch.nn.Conv1d(channels, channels, 1),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Conv1d(channels, 1, 1),
            )

        def forward(self, x):
            hidden = self.input(x)
            skip_sum = None
            for block in self.blocks:
                hidden, skip = block(hidden)
                skip_sum = skip if skip_sum is None else skip_sum + skip
            return self.output(skip_sum / math.sqrt(max(1, len(self.blocks))))

    receptive = 1 + 2 * stacks * sum(2**level for level in range(levels))
    return FullnessTCN(), receptive


def build_lstm(hidden_size: int, input_channels: int = 1):
    import torch

    class RecurrentAmp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_channels, hidden_size, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, x):
            sequence = x.transpose(1, 2)
            hidden, _ = self.lstm(sequence)
            return self.output(hidden).transpose(1, 2)

        def forward_stream(self, x, state=None):
            sequence = x.transpose(1, 2)
            hidden, next_state = self.lstm(sequence, state)
            return self.output(hidden).transpose(1, 2), next_state

    return RecurrentAmp(), 4096


def build_gru(hidden_size: int, input_channels: int = 1):
    import torch

    class RecurrentAmp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(input_channels, hidden_size, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, 1)

        def forward(self, x):
            sequence = x.transpose(1, 2)
            hidden, _ = self.gru(sequence)
            return self.output(hidden).transpose(1, 2)

        def forward_stream(self, x, state=None):
            sequence = x.transpose(1, 2)
            hidden, next_state = self.gru(sequence, state)
            return self.output(hidden).transpose(1, 2), next_state

    return RecurrentAmp(), 4096


def conditioned_tensor(audio: np.ndarray, device, condition: np.ndarray | None = None):
    import torch

    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(device).reshape(1, 1, -1)
    if condition is None or len(condition) == 0:
        return tensor
    values = torch.from_numpy(np.asarray(condition, dtype=np.float32)).to(device).reshape(1, -1, 1)
    return torch.cat([tensor, values.expand(-1, -1, tensor.shape[-1])], dim=1)


def render_model(
    model,
    audio: np.ndarray,
    device,
    context: int,
    chunk_samples: int = 65536,
    condition: np.ndarray | None = None,
) -> np.ndarray:
    import torch

    result = np.zeros(len(audio), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "forward_stream"):
            state = None
            if context > 0:
                warmup = conditioned_tensor(np.zeros(context, dtype=np.float32), device, condition=condition)
                _, state = model.forward_stream(warmup, state)
            for start in range(0, len(audio), chunk_samples):
                end = min(len(audio), start + chunk_samples)
                tensor = conditioned_tensor(audio[start:end], device, condition=condition)
                prediction, state = model.forward_stream(tensor, state)
                result[start:end] = prediction.detach().cpu().numpy().reshape(-1)
            return result
        for start in range(0, len(audio), chunk_samples):
            end = min(len(audio), start + chunk_samples)
            history_start = max(0, start - context)
            segment = audio[history_start:end]
            if start < context:
                segment = np.pad(segment, (context - start, 0))
            tensor = conditioned_tensor(segment, device, condition=condition)
            prediction = model(tensor).detach().cpu().numpy().reshape(-1)
            result[start:end] = prediction[-(end - start) :]
    return result


def build_perceptual_losses(auraloss, sample_rate: int, device, chunk_samples: int) -> dict:
    import torch
    import torchaudio

    usable_sizes = [size for size in ([512, 2048, 8192] if sample_rate >= 88200 else [256, 1024, 4096]) if size <= chunk_samples]
    if len(usable_sizes) < 2:
        usable_sizes = [max(64, chunk_samples // 4), max(128, chunk_samples)]
    common = {
        "fft_sizes": usable_sizes,
        "hop_sizes": [max(1, size // 4) for size in usable_sizes],
        "win_lengths": usable_sizes,
    }
    linear = auraloss.freq.MultiResolutionSTFTLoss(**common).to(device)
    mel_banks = []
    windows = []
    for fft_size in usable_sizes:
        mel_bins = min(96, max(24, int(fft_size) // 16))
        bank = torchaudio.functional.melscale_fbanks(
            n_freqs=(fft_size // 2) + 1,
            f_min=35.0,
            f_max=min(16000.0, sample_rate * 0.49),
            n_mels=mel_bins,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        mel_banks.append(bank.to(device))
        windows.append(torch.hann_window(fft_size, device=device))
    return {
        "linear_stft": linear,
        "mel_fft_sizes": usable_sizes,
        "mel_banks": mel_banks,
        "mel_windows": windows,
    }


def mel_perceptual_mrstft(prediction, target, losses: dict):
    import torch

    terms = []
    pred_audio = prediction.squeeze(1)
    target_audio = target.squeeze(1)
    for fft_size, bank, window in zip(
        losses["mel_fft_sizes"],
        losses["mel_banks"],
        losses["mel_windows"],
    ):
        hop = max(1, int(fft_size) // 4)
        pred_stft = torch.stft(
            pred_audio,
            n_fft=int(fft_size),
            hop_length=hop,
            win_length=int(fft_size),
            window=window,
            return_complex=True,
        )
        target_stft = torch.stft(
            target_audio,
            n_fft=int(fft_size),
            hop_length=hop,
            win_length=int(fft_size),
            window=window,
            return_complex=True,
        )
        pred_mel = torch.matmul(pred_stft.abs().transpose(1, 2), bank).clamp_min(1e-7)
        target_mel = torch.matmul(target_stft.abs().transpose(1, 2), bank).clamp_min(1e-7)
        log_error = torch.mean(torch.abs(torch.log(pred_mel) - torch.log(target_mel)))
        convergence = torch.linalg.vector_norm(pred_mel - target_mel) / (
            torch.linalg.vector_norm(target_mel) + 1e-7
        )
        terms.append(log_error + 0.25 * convergence)
    return sum(terms) / len(terms)


def multiband_log_energy_loss(prediction, target, sample_rate: int):
    import torch

    fft_size = min(4096, int(prediction.shape[-1]))
    fft_size = 2 ** int(math.floor(math.log2(max(256, fft_size))))
    hop = max(1, fft_size // 4)
    window = torch.hann_window(fft_size, device=prediction.device)
    pred_stft = torch.stft(
        prediction.squeeze(1),
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        return_complex=True,
    )
    target_stft = torch.stft(
        target.squeeze(1),
        n_fft=fft_size,
        hop_length=hop,
        win_length=fft_size,
        window=window,
        return_complex=True,
    )
    frequencies = torch.fft.rfftfreq(fft_size, d=1.0 / sample_rate).to(prediction.device)
    bands = ((55.0, 180.0), (180.0, 500.0), (500.0, 1600.0), (1600.0, 4000.0), (4000.0, 9000.0), (9000.0, min(16000.0, sample_rate * 0.49)))
    pred_power = pred_stft.abs().square().mean(dim=-1)
    target_power = target_stft.abs().square().mean(dim=-1)
    terms = []
    for low_hz, high_hz in bands:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        if bool(torch.any(mask).item()):
            pred_band = pred_power[:, mask].mean(dim=-1).clamp_min(1e-9)
            target_band = target_power[:, mask].mean(dim=-1).clamp_min(1e-9)
            terms.append(torch.mean(torch.abs(torch.log(pred_band) - torch.log(target_band))))
    return sum(terms) / max(1, len(terms))


def amp_perceptual_loss(prediction, target, losses: dict, sample_rate: int, profile: str = "balanced-v1"):
    import torch
    import torch.nn.functional as functional

    error = prediction - target
    target_energy = torch.mean(target.square()) + 1e-7
    esr = torch.mean(error.square()) / target_energy
    waveform = functional.smooth_l1_loss(prediction, target, beta=0.04)
    pre_prediction = prediction[:, :, 1:] - 0.97 * prediction[:, :, :-1]
    pre_target = target[:, :, 1:] - 0.97 * target[:, :, :-1]
    pre_esr = torch.mean((pre_prediction - pre_target).square()) / (torch.mean(pre_target.square()) + 1e-7)
    pred_difference = torch.diff(prediction)
    target_difference = torch.diff(target)
    transient = torch.mean(torch.abs(pred_difference - target_difference))
    transient_esr = torch.mean((pred_difference - target_difference).square()) / (
        torch.mean(target_difference.square()) + 1e-7
    )
    envelope_terms = []
    log_envelope_terms = []
    for milliseconds in (2.0, 18.0, 60.0):
        kernel = max(3, int(round(sample_rate * milliseconds / 1000.0)))
        kernel = min(kernel, max(3, prediction.shape[-1] // 3))
        pred_envelope = functional.avg_pool1d(torch.abs(prediction), kernel, stride=1, padding=kernel // 2)
        target_envelope = functional.avg_pool1d(torch.abs(target), kernel, stride=1, padding=kernel // 2)
        common = min(pred_envelope.shape[-1], target_envelope.shape[-1])
        envelope_terms.append(torch.mean(torch.abs(pred_envelope[:, :, :common] - target_envelope[:, :, :common])))
        pred_rms = torch.sqrt(
            functional.avg_pool1d(prediction.square(), kernel, stride=1, padding=kernel // 2) + 1e-9
        )
        target_rms = torch.sqrt(
            functional.avg_pool1d(target.square(), kernel, stride=1, padding=kernel // 2) + 1e-9
        )
        common_rms = min(pred_rms.shape[-1], target_rms.shape[-1])
        log_envelope_terms.append(
            torch.mean(torch.abs(torch.log(pred_rms[:, :, :common_rms]) - torch.log(target_rms[:, :, :common_rms])))
        )
    envelope = sum(envelope_terms) / len(envelope_terms)
    log_envelope = sum(log_envelope_terms) / len(log_envelope_terms)
    dc = torch.abs(torch.mean(prediction) - torch.mean(target))
    linear_spectral = losses["linear_stft"](prediction, target)
    mel_spectral = mel_perceptual_mrstft(prediction, target, losses)
    band_energy = multiband_log_energy_loss(prediction, target, sample_rate)
    pred_rms = torch.sqrt(torch.mean(prediction.square()) + 1e-9)
    target_rms = torch.sqrt(torch.mean(target.square()) + 1e-9)
    level = torch.abs(torch.log(pred_rms) - torch.log(target_rms))
    pred_crest = torch.amax(torch.abs(prediction)) / (pred_rms + 1e-7)
    target_crest = torch.amax(torch.abs(target)) / (target_rms + 1e-7)
    crest = torch.abs(torch.log(pred_crest + 1e-7) - torch.log(target_crest + 1e-7))
    if profile == "fullness-v2":
        total = (
            0.24 * torch.log1p(esr)
            + 0.10 * torch.log1p(pre_esr)
            + 0.11 * torch.log1p(linear_spectral)
            + 0.11 * torch.log1p(mel_spectral)
            + 0.16 * torch.log1p(band_energy)
            + 0.12 * torch.log1p(log_envelope)
            + 0.07 * torch.log1p(transient_esr)
            + 0.04 * torch.log1p(level)
            + 0.03 * torch.log1p(crest)
            + 0.01 * waveform
            + 0.01 * dc
        )
    else:
        total = (
            0.30 * esr
            + 0.14 * pre_esr
            + 0.14 * linear_spectral
            + 0.14 * mel_spectral
            + 0.10 * envelope
            + 0.08 * transient
            + 0.06 * waveform
            + 0.04 * dc
        )
    return total, {
        "esr": esr,
        "preemphasized_esr": pre_esr,
        "linear_mrstft": linear_spectral,
        "mel_perceptual_mrstft": mel_spectral,
        "multiband_log_energy": band_energy,
        "envelope": envelope,
        "log_rms_envelope": log_envelope,
        "transient": transient,
        "transient_esr": transient_esr,
        "level_log_error": level,
        "crest_log_error": crest,
        "waveform_huber": waveform,
        "dc": dc,
    }


def align_training_pair(source: np.ndarray, target: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    from scipy.signal import correlate

    probe_length = min(len(source), len(target), int(sample_rate * 12.0))
    decimation = max(1, sample_rate // 12000)
    source_probe = source[:probe_length:decimation]
    target_probe = target[:probe_length:decimation]
    values = correlate(target_probe, source_probe, mode="full", method="fft")
    lags = np.arange(-len(source_probe) + 1, len(target_probe))
    limit = max(1, int(0.10 * sample_rate / decimation))
    allowed_indices = np.flatnonzero(np.abs(lags) <= limit)
    local_index = int(np.argmax(np.abs(values[allowed_indices])))
    peak_index = int(allowed_indices[local_index])
    lag_decimated = float(lags[peak_index])
    polarity = -1 if float(values[peak_index]) < 0.0 else 1
    fraction = 0.0
    if 0 < peak_index < len(values) - 1:
        left, center, right = [float(abs(values[index])) for index in (peak_index - 1, peak_index, peak_index + 1)]
        denominator = left - (2.0 * center) + right
        if abs(denominator) > 1e-20:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    lag = (lag_decimated + fraction) * decimation
    integer_lag = int(np.floor(lag))
    if integer_lag > 0:
        target_aligned = target[integer_lag:]
        source_aligned = source[: len(target_aligned)]
    elif integer_lag < 0:
        source_aligned = source[-integer_lag:]
        target_aligned = target[: len(source_aligned)]
    else:
        length = min(len(source), len(target))
        source_aligned = source[:length]
        target_aligned = target[:length]
    length = min(len(source_aligned), len(target_aligned))
    source_aligned = np.asarray(source_aligned[:length], dtype=np.float32)
    target_aligned = np.asarray(target_aligned[:length] * polarity, dtype=np.float32)
    residual = lag - integer_lag
    if abs(residual) > 1e-4 and length > 8:
        positions = np.arange(length, dtype=np.float64) + residual
        target_aligned = np.interp(
            positions,
            np.arange(length, dtype=np.float64),
            target_aligned,
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    return source_aligned, target_aligned, float(lag), int(polarity)


CONDITION_FIELDS = ("guitar", "tuning", "pickup", "pickup_mode", "guitar_volume", "guitar_tone")


def build_condition_schema(entries: list[dict]) -> dict:
    schema = {"rig_fingerprint": sorted({str(entry.get("rig_fingerprint", "unlabeled")) for entry in entries})}
    for field in CONDITION_FIELDS:
        schema[field] = sorted({str(dict(entry.get("conditions", {})).get(field, "unlabeled")) for entry in entries})
    return schema


def condition_vector(entry: dict, schema: dict) -> np.ndarray:
    values = []
    for field, categories in schema.items():
        selected = (
            str(entry.get("rig_fingerprint", "unlabeled"))
            if field == "rig_fingerprint"
            else str(dict(entry.get("conditions", {})).get(field, "unlabeled"))
        )
        values.extend(1.0 if category == selected else 0.0 for category in categories)
    return np.asarray(values, dtype=np.float32)


def load_conditioned_entry(entry: dict, sample_rate: int, condition_schema: dict) -> dict:
    source_rate, source = read_audio(Path(entry["di"]))
    target_rate, target = read_audio(Path(entry["amp_target"]))
    source = resample(source, source_rate, sample_rate)
    target = resample(target, target_rate, sample_rate)
    source, target, lag, polarity = align_training_pair(source, target, sample_rate)
    return {
        "take_name": str(entry.get("take_name", Path(entry["di"]).stem)),
        "source": np.asarray(source, dtype=np.float32),
        "target": np.asarray(target, dtype=np.float32),
        "condition": condition_vector(entry, condition_schema),
        "entry": entry,
        "lag_samples": lag,
        "polarity": polarity,
    }


def internal_validation_pairs(training_pairs: list[dict], focus_rigs: set[str], maximum: int) -> list[dict]:
    if maximum <= 0 or len(training_pairs) <= maximum:
        return list(training_pairs)
    focused = [
        pair for pair in training_pairs if str(pair["entry"].get("rig_fingerprint", "")) in focus_rigs
    ]
    focused_ids = {id(pair) for pair in focused}
    remaining = [pair for pair in training_pairs if id(pair) not in focused_ids]
    slots = max(0, maximum - len(focused))
    if slots and remaining:
        indices = np.linspace(0, len(remaining) - 1, min(slots, len(remaining)), dtype=np.int64)
        focused.extend(remaining[int(index)] for index in indices)
    return focused[:maximum]


def evaluate_internal_validation(
    model,
    pairs: list[dict],
    device,
    receptive: int,
    chunk_samples: int,
    target_scale: float,
    perceptual_losses: dict,
    sample_rate: int,
    loss_profile: str,
) -> tuple[float, dict[str, float]]:
    import torch

    losses = []
    component_totals: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            validation_start = int(pair["internal_validation_start"])
            maximum_start = min(len(pair["source"]), len(pair["target"])) - chunk_samples
            start = validation_start + max(0, (maximum_start - validation_start) // 2)
            segment = pair["source"][start - receptive : start + chunk_samples]
            x = conditioned_tensor(segment, device, condition=pair["condition"])
            target = pair["target"][start : start + chunk_samples] / target_scale
            y = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device).reshape(1, 1, -1)
            prediction = model(x)[:, :, -chunk_samples:]
            loss, components = amp_perceptual_loss(
                prediction,
                y,
                perceptual_losses,
                sample_rate,
                profile=loss_profile,
            )
            losses.append(float(loss.detach().cpu().item()))
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.detach().cpu().item())
    model.train()
    return float(np.mean(losses)), {
        name: value / max(1, len(losses)) for name, value in component_totals.items()
    }


def run_train_reference(args: argparse.Namespace) -> None:
    import auraloss
    import torch

    ensure_tone_only_paths([args.capture_manifest, args.model, args.output, args.metrics_output])
    if min(args.epochs, args.steps_per_epoch, args.chunk_samples, args.print_every) < 1:
        raise SystemExit("Epochs, steps, chunk samples, and print interval must all be at least 1.")
    if args.architecture in {"tcn", "tcn-v2"} and min(args.channels, args.levels, args.tcn_stacks) < 1:
        raise SystemExit("TCN channels and levels must be at least 1.")
    if args.architecture in {"lstm", "gru"} and args.hidden_size < 1:
        raise SystemExit("Recurrent hidden size must be at least 1.")
    capture = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    files = capture["files"]
    train_input_rate, train_input = read_audio(Path(files["train_input"]))
    train_target_rate, train_target = read_audio(Path(files["train_target"]))
    validation_input_rate, validation_input = read_audio(Path(files["validation_input"]))
    validation_target_rate, validation_target = read_audio(Path(files["validation_target"]))
    if len({train_input_rate, train_target_rate, validation_input_rate, validation_target_rate}) != 1:
        raise SystemExit("Prepared research audio files must share one sample rate.")
    sample_rate = train_input_rate
    if args.architecture == "tcn":
        model, receptive = build_tcn(args.channels, args.levels)
    elif args.architecture == "tcn-v2":
        model, receptive = build_tcn_fullness(args.channels, args.levels, stacks=args.tcn_stacks)
    elif args.architecture == "gru":
        model, receptive = build_gru(args.hidden_size)
    else:
        model, receptive = build_lstm(args.hidden_size)
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    perceptual_losses = build_perceptual_losses(auraloss, sample_rate, device, int(args.chunk_samples))
    rng = np.random.default_rng(args.seed)
    chunk = int(args.chunk_samples)
    if len(train_input) < chunk + receptive or len(train_target) < chunk + receptive:
        raise SystemExit("Prepared training audio is too short for the requested chunk/receptive field.")
    target_scale = max(0.05, float(np.percentile(np.abs(train_target), 99.9)))
    x_audio = np.asarray(train_input, dtype=np.float32)
    y_audio = np.asarray(train_target / target_scale, dtype=np.float32)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for _ in range(args.steps_per_epoch):
            start = int(rng.integers(receptive, len(x_audio) - chunk))
            x = torch.from_numpy(x_audio[start - receptive : start + chunk]).to(device).reshape(1, 1, -1)
            y = torch.from_numpy(y_audio[start : start + chunk]).to(device).reshape(1, 1, -1)
            prediction = model(x)[:, :, -chunk:]
            loss, _ = amp_perceptual_loss(
                prediction,
                y,
                perceptual_losses,
                sample_rate,
                profile=args.loss_profile,
            )
            if not bool(torch.isfinite(loss).item()):
                raise SystemExit("Nonfinite PyTorch loss detected; candidate was not saved or accepted.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(np.mean(losses))
        history.append(epoch_loss)
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(f"Epoch {epoch:03d}: train_loss={epoch_loss:.6f}")
    prediction = limited_audio(render_model(model, validation_input, device, receptive - 1) * target_scale)
    write_audio(args.output, sample_rate, prediction)
    metrics = candidate_metrics(
        validation_input,
        validation_target,
        prediction,
        sample_rate,
        min_improvement_db=args.min_improvement_db,
        min_movement_db=args.min_movement_db,
        max_listening_spectral_error_db=args.max_listening_spectral_error_db,
        min_listening_correlation=args.min_listening_correlation,
        max_listening_level_error_db=args.max_listening_level_error_db,
        min_listening_section_pass_rate=args.min_listening_section_pass_rate,
    )
    accepted = bool(metrics["listening_promotion_ready"])
    saved_model = args.model
    if not accepted and not args.allow_failed_validation:
        saved_model = args.model.with_name(f"{args.model.stem}.rejected{args.model.suffix}")
    saved_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "tone_capture_torch_reference_1.0",
            "architecture": args.architecture,
            "channels": args.channels,
            "levels": args.levels,
            "tcn_stacks": args.tcn_stacks,
            "hidden_size": args.hidden_size,
            "input_channels": 1,
            "receptive_field_samples": receptive,
            "sample_rate_hz": sample_rate,
            "target_scale": target_scale,
            "capture_manifest": str(args.capture_manifest),
            "accepted": accepted,
            "loss_profile": args.loss_profile,
            "state_dict": model.state_dict(),
        },
        saved_model,
    )
    payload = {
        "model": str(saved_model),
        "render": str(args.output),
        "architecture": args.architecture,
        "device": str(device),
        "accepted": accepted,
        "training_loss": history,
        "validation": metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not accepted and not args.allow_failed_validation:
        print("REJECTED: candidate did not beat the DI-gain-only amp-tone guard.")


def run_train_conditioned_reference(args: argparse.Namespace) -> None:
    import auraloss
    import torch

    ensure_tone_only_paths([args.dataset_manifest, args.model, args.output, args.metrics_output])
    if min(args.epochs, args.steps_per_epoch, args.chunk_samples, args.print_every) < 1:
        raise SystemExit("Epochs, steps, chunk samples, and print interval must all be at least 1.")
    if args.architecture in {"tcn", "tcn-v2"} and min(args.channels, args.levels, args.tcn_stacks) < 1:
        raise SystemExit("TCN channels, levels, and stacks must all be at least 1.")
    if args.checkpoint_every < 1 or args.internal_validation_takes < 1:
        raise SystemExit("Checkpoint interval and internal validation take count must be at least 1.")
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if dataset.get("format") != "tone_capture_conditioned_dataset_1.0":
        raise SystemExit(f"Unsupported conditioned dataset: {args.dataset_manifest}")
    entries = [entry for entry in dataset.get("entries", []) if float(entry.get("sampling_weight", 1.0)) > 0.0]
    if len(entries) < 2:
        raise SystemExit("Conditioned training requires at least two complete recording pairs.")

    holdout_names = set(args.holdout_take or [])
    if not holdout_names:
        group_counts = {}
        for entry in entries:
            group_counts.setdefault(str(entry.get("rig_fingerprint", "")), []).append(entry)
        eligible = [group for group in group_counts.values() if len(group) >= 2]
        if not eligible:
            raise SystemExit("No rig group has enough takes for a whole-take holdout.")
        selected_group = max(eligible, key=len)
        holdout_names.add(sorted(selected_group, key=lambda item: str(item.get("take_name", "")))[-1]["take_name"])
    known_names = {str(entry.get("take_name", "")) for entry in entries}
    missing_holdouts = holdout_names - known_names
    if missing_holdouts:
        raise SystemExit(f"Unknown holdout takes: {', '.join(sorted(missing_holdouts))}")
    train_entries = [entry for entry in entries if str(entry.get("take_name", "")) not in holdout_names]
    validation_entries = [entry for entry in entries if str(entry.get("take_name", "")) in holdout_names]
    if not train_entries or not validation_entries:
        raise SystemExit("Whole-take split must leave at least one training and one validation take.")

    condition_schema = build_condition_schema(entries)
    condition_dim = sum(len(values) for values in condition_schema.values())
    sample_rate = int(args.sample_rate)
    print(
        f"Loading conditioned dataset: train_takes={len(train_entries)} "
        f"whole_take_holdouts={len(validation_entries)} sample_rate={sample_rate} condition_features={condition_dim}"
    )
    training_pairs = []
    for index, entry in enumerate(train_entries, start=1):
        pair = load_conditioned_entry(entry, sample_rate, condition_schema)
        training_pairs.append(pair)
        print(
            f"  train {index:02d}/{len(train_entries)} {pair['take_name']} "
            f"seconds={len(pair['source']) / sample_rate:.1f} lag={pair['lag_samples']:.2f} polarity={pair['polarity']:+d}"
        )
    validation_pairs = []
    for entry in validation_entries:
        pair = load_conditioned_entry(entry, sample_rate, condition_schema)
        validation_pairs.append(pair)
        print(
            f"  HOLDOUT {pair['take_name']} seconds={len(pair['source']) / sample_rate:.1f} "
            f"lag={pair['lag_samples']:.2f} polarity={pair['polarity']:+d}"
        )

    input_channels = 1 + condition_dim
    if args.architecture == "tcn":
        model, receptive = build_tcn(args.channels, args.levels, input_channels=input_channels)
    elif args.architecture == "tcn-v2":
        model, receptive = build_tcn_fullness(
            args.channels,
            args.levels,
            stacks=args.tcn_stacks,
            input_channels=input_channels,
        )
    elif args.architecture == "gru":
        model, receptive = build_gru(args.hidden_size, input_channels=input_channels)
    else:
        model, receptive = build_lstm(args.hidden_size, input_channels=input_channels)
    validation_fraction = float(np.clip(args.training_validation_fraction, 0.0, 0.40))
    for pair in training_pairs:
        total_samples = min(len(pair["source"]), len(pair["target"]))
        tail_samples = max(int(args.chunk_samples), int(round(total_samples * validation_fraction)))
        pair["internal_validation_start"] = total_samples - tail_samples
        pair["optimization_end"] = pair["internal_validation_start"]
    usable_pairs = [
        pair
        for pair in training_pairs
        if int(pair["optimization_end"]) - int(args.chunk_samples) > receptive
    ]
    if not usable_pairs:
        raise SystemExit("No training take is long enough for the requested chunk and receptive field.")
    focus_rigs = {str(entry.get("rig_fingerprint", "")) for entry in validation_entries}
    focus_mask = np.asarray(
        [str(pair["entry"].get("rig_fingerprint", "")) in focus_rigs for pair in usable_pairs],
        dtype=bool,
    )
    focus_fraction = float(np.clip(args.focus_rig_fraction, 0.0, 1.0))
    pair_sampling_weights = np.zeros(len(usable_pairs), dtype=np.float64)
    if np.any(focus_mask) and np.any(~focus_mask):
        pair_sampling_weights[focus_mask] = focus_fraction / int(np.sum(focus_mask))
        pair_sampling_weights[~focus_mask] = (1.0 - focus_fraction) / int(np.sum(~focus_mask))
    else:
        pair_sampling_weights[:] = 1.0 / len(usable_pairs)
    pair_sampling_weights /= float(np.sum(pair_sampling_weights) + 1e-12)
    checkpoint_pairs = internal_validation_pairs(
        usable_pairs,
        focus_rigs,
        maximum=int(args.internal_validation_takes),
    )
    print(
        "Take sampling: all recordings active; "
        f"holdout-rig training share={float(np.sum(pair_sampling_weights[focus_mask])) * 100.0:.1f}% "
        f"other-rig regularization share={float(np.sum(pair_sampling_weights[~focus_mask])) * 100.0:.1f}%"
    )
    focus_pair_count = int(np.sum(focus_mask))
    if focus_pair_count < 3:
        print(
            f"DATA LIMIT: only {focus_pair_count} approved training take(s) match the whole-take holdout's "
            "exact pedal/amp/cab/microphone fingerprint. Capture at least three repeated exact-rig takes "
            "before expecting a stable high-accuracy promotion."
        )
    print(
        f"Checkpoint selection: {validation_fraction * 100.0:.1f}% tail excluded from optimization "
        f"for {len(checkpoint_pairs)} training takes; whole-take holdout remains separate"
    )
    print(
        f"Architecture={args.architecture} channels={args.channels} levels={args.levels} "
        f"stacks={args.tcn_stacks if args.architecture == 'tcn-v2' else 1} "
        f"receptive={receptive} samples/{1000.0 * receptive / sample_rate:.1f} ms loss={args.loss_profile}"
    )

    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    perceptual_losses = build_perceptual_losses(auraloss, sample_rate, device, int(args.chunk_samples))
    target_levels = [float(np.percentile(np.abs(pair["target"]), 99.9)) for pair in usable_pairs]
    target_scale = max(0.05, float(np.median(target_levels)))
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        component_totals: dict[str, float] = {}
        for _ in range(args.steps_per_epoch):
            pair = usable_pairs[int(rng.choice(len(usable_pairs), p=pair_sampling_weights))]
            maximum_start = int(pair["optimization_end"]) - int(args.chunk_samples)
            start = int(rng.integers(receptive, maximum_start))
            segment = pair["source"][start - receptive : start + int(args.chunk_samples)]
            x = conditioned_tensor(segment, device, condition=pair["condition"])
            target = pair["target"][start : start + int(args.chunk_samples)] / target_scale
            y = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device).reshape(1, 1, -1)
            prediction = model(x)[:, :, -int(args.chunk_samples) :]
            loss, components = amp_perceptual_loss(
                prediction,
                y,
                perceptual_losses,
                sample_rate,
                profile=args.loss_profile,
            )
            if not bool(torch.isfinite(loss).item()):
                raise SystemExit("Nonfinite conditioned loss detected; candidate was not saved.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value.detach().cpu().item())
        epoch_loss = float(np.mean(losses))
        component_means = {name: value / max(1, len(losses)) for name, value in component_totals.items()}
        validation_loss = None
        validation_components = {}
        if epoch == 1 or epoch == args.epochs or epoch % int(args.checkpoint_every) == 0:
            validation_loss, validation_components = evaluate_internal_validation(
                model,
                checkpoint_pairs,
                device,
                receptive,
                int(args.chunk_samples),
                target_scale,
                perceptual_losses,
                sample_rate,
                args.loss_profile,
            )
            if validation_loss < best_loss - float(args.min_delta):
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss,
                "training_components": component_means,
                "internal_validation_loss": validation_loss,
                "internal_validation_components": validation_components,
            }
        )
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(
                f"Epoch {epoch:03d}: train={epoch_loss:.6f} "
                f"internal_val={validation_loss if validation_loss is not None else float('nan'):.6f} "
                f"esr={component_means.get('esr', 0.0):.5f} "
                f"mel={component_means.get('mel_perceptual_mrstft', 0.0):.5f} "
                f"band={component_means.get('multiband_log_energy', 0.0):.5f} "
                f"envelope={component_means.get('log_rms_envelope', 0.0):.5f}"
            )
        if (
            validation_loss is not None
            and args.early_stopping_patience > 0
            and stale_epochs >= args.early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch}; restoring best training epoch.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    validation_results = []
    output_paths = []
    audition_reports = {}
    model.eval()
    for index, pair in enumerate(validation_pairs):
        prediction = limited_audio(
            render_model(
                model,
                pair["source"],
                device,
                receptive - 1,
                chunk_samples=int(args.render_chunk_samples),
                condition=pair["condition"],
            )
            * target_scale
        )
        output_path = args.output
        if index > 0:
            output_path = args.output.with_name(f"{args.output.stem}_{index + 1:02d}{args.output.suffix}")
        write_audio(output_path, sample_rate, prediction)
        metrics = candidate_metrics(
            pair["source"],
            pair["target"],
            prediction,
            sample_rate,
            min_improvement_db=args.min_improvement_db,
            min_movement_db=args.min_movement_db,
            max_listening_spectral_error_db=args.max_listening_spectral_error_db,
            min_listening_correlation=args.min_listening_correlation,
            max_listening_level_error_db=args.max_listening_level_error_db,
            min_listening_section_pass_rate=args.min_listening_section_pass_rate,
        )
        validation_results.append({"take_name": pair["take_name"], "output": str(output_path), **metrics})
        if args.audition_dir:
            audition_reports[pair["take_name"]] = write_dry_listening_auditions(
                args.audition_dir,
                output_path.stem,
                pair["source"],
                pair["target"],
                prediction,
                sample_rate,
                metrics,
                seconds=args.audition_seconds,
            )
        output_paths.append(output_path)
    accepted = bool(validation_results and all(item["listening_promotion_ready"] for item in validation_results))
    saved_model = args.model
    if not accepted and not args.allow_failed_validation:
        saved_model = args.model.with_name(f"{args.model.stem}.rejected{args.model.suffix}")
    saved_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "tone_capture_torch_conditioned_reference_2.0",
            "architecture": args.architecture,
            "channels": args.channels,
            "levels": args.levels,
            "tcn_stacks": args.tcn_stacks,
            "hidden_size": args.hidden_size,
            "input_channels": input_channels,
            "receptive_field_samples": receptive,
            "sample_rate_hz": sample_rate,
            "target_scale": target_scale,
            "capture_endpoint": "full_amp_cab_close_mic",
            "render_policy": "raw dry model output; no room, post cabinet, or per-section normalization",
            "dataset_manifest": str(args.dataset_manifest),
            "frozen_manifest": str(args.frozen_manifest) if args.frozen_manifest else None,
            "condition_schema": condition_schema,
            "holdout_takes": sorted(holdout_names),
            "loss_profile": args.loss_profile,
            "training_validation_fraction": validation_fraction,
            "accepted": accepted,
            "state_dict": model.state_dict(),
        },
        saved_model,
    )
    aggregate = {
        "mean_spectral_error_db": float(np.mean([item["spectral_error_db"] for item in validation_results])),
        "mean_mrstft_loss": float(np.mean([item["multi_resolution_stft_loss"] for item in validation_results])),
        "mean_correlation": float(np.mean([item["correlation"] for item in validation_results])),
        "amp_tone_guard_pass_rate": float(
            np.mean([item["amp_tone_guard_passed"] for item in validation_results])
        ),
        "listening_promotion_pass_rate": float(
            np.mean([item["listening_promotion_ready"] for item in validation_results])
        ),
        "mean_audible_match_score_0_100": float(
            np.mean([item["audible_match_score_0_100"] for item in validation_results])
        ),
    }
    payload = {
        "model": str(saved_model),
        "renders": [str(path) for path in output_paths],
        "architecture": args.architecture,
        "device": str(device),
        "accepted": accepted,
        "training_take_count": len(training_pairs),
        "sampling_policy": {
            "all_recordings_active": True,
            "focus_rigs": sorted(focus_rigs),
            "exact_rig_training_take_count": focus_pair_count,
            "focus_rig_fraction": focus_fraction,
            "optimization_excludes_training_tail_fraction": validation_fraction,
            "internal_validation_takes": [pair["take_name"] for pair in checkpoint_pairs],
        },
        "whole_take_holdouts": sorted(holdout_names),
        "condition_schema": condition_schema,
        "losses": [
            "ESR",
            "preemphasized_ESR",
            "linear_multi_resolution_STFT",
            "mel_perceptual_multi_resolution_STFT",
            "guitar_band_log_energy",
            "multiscale_envelope",
            "multiscale_log_RMS_envelope",
            "transient",
            "transient_ESR",
            "level_and_crest",
            "waveform_Huber",
            "DC",
        ],
        "loss_profile": args.loss_profile,
        "capture_endpoint": "full_amp_cab_close_mic",
        "render_policy": {
            "dry_close_mic": True,
            "additional_room_stage": False,
            "additional_cabinet_stage": False,
            "audition_level_matching": "one global correction shared by every listening section",
        },
        "promotion_policy": (
            "whole-take amp-tone guard plus quiet/medium/loud/transient/sustained listening gates; "
            "training loss alone cannot promote a model"
        ),
        "best_internal_validation_loss": best_loss,
        "training_history": history,
        "aggregate_validation": aggregate,
        "validation": validation_results,
        "dry_listening_auditions": audition_reports,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "training_history"}, indent=2))
    if not accepted and not args.allow_failed_validation:
        print("REJECTED: candidate failed the whole-take amp-tone or listening-promotion guard.")


def run_apply_reference(args: argparse.Namespace) -> None:
    import torch

    ensure_tone_only_paths([args.input, args.model, args.output])
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in {
        "tone_capture_torch_reference_1.0",
        "tone_capture_torch_conditioned_reference_2.0",
    }:
        raise SystemExit(f"Unsupported PyTorch tone model: {args.model}")
    if not checkpoint.get("accepted", False) and not args.allow_rejected:
        raise SystemExit("Refusing to apply a rejected PyTorch reference model.")
    architecture = str(checkpoint["architecture"])
    if architecture == "tcn":
        model, receptive = build_tcn(
            int(checkpoint["channels"]),
            int(checkpoint["levels"]),
            input_channels=int(checkpoint.get("input_channels", 1)),
        )
    elif architecture == "tcn-v2":
        model, receptive = build_tcn_fullness(
            int(checkpoint["channels"]),
            int(checkpoint["levels"]),
            stacks=int(checkpoint.get("tcn_stacks", 2)),
            input_channels=int(checkpoint.get("input_channels", 1)),
        )
    elif architecture == "lstm":
        model, receptive = build_lstm(
            int(checkpoint["hidden_size"]),
            input_channels=int(checkpoint.get("input_channels", 1)),
        )
    elif architecture == "gru":
        model, receptive = build_gru(
            int(checkpoint["hidden_size"]),
            input_channels=int(checkpoint.get("input_channels", 1)),
        )
    else:
        raise SystemExit(f"Unsupported PyTorch architecture: {architecture}")
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    model.to(device)
    source_rate, source = read_audio(args.input)
    model_rate = int(checkpoint["sample_rate_hz"])
    source = resample(source, source_rate, model_rate)
    source = source * float(10.0 ** (args.input_trim_db / 20.0))
    condition = None
    condition_schema = dict(checkpoint.get("condition_schema", {}))
    if condition_schema:
        condition_entry = {
            "rig_fingerprint": str(args.rig_fingerprint or ""),
            "conditions": {
                "guitar": str(args.guitar or ""),
                "tuning": str(args.tuning or ""),
                "pickup": str(args.pickup or ""),
                "pickup_mode": str(args.pickup_mode or ""),
                "guitar_volume": str(args.guitar_volume or ""),
                "guitar_tone": str(args.guitar_tone or ""),
            },
        }
        missing = []
        for field, categories in condition_schema.items():
            selected = (
                condition_entry["rig_fingerprint"]
                if field == "rig_fingerprint"
                else condition_entry["conditions"].get(field, "")
            )
            if selected not in categories:
                missing.append(f"{field}={selected or '<required>'}")
        if missing:
            raise SystemExit(
                "Conditioned model needs known labels: " + ", ".join(missing) + ". "
                "Use the exact values stored in the model's condition schema."
            )
        condition = condition_vector(condition_entry, condition_schema)
    output = render_model(model, source, device, receptive - 1, condition=condition) * float(checkpoint["target_scale"])
    output *= float(10.0 ** (args.output_trim_db / 20.0))
    output = limited_audio(output)
    write_audio(args.output, model_rate, output)
    print(f"Wrote PyTorch reference render: {args.output}")
    print(f"Architecture={architecture} sample_rate={model_rate} device={device}")


def run_apply_nam_reference(args: argparse.Namespace) -> None:
    import torch
    from nam.models import init_from_nam

    ensure_tone_only_paths([args.input, args.model, args.output])
    if not args.model.exists():
        raise SystemExit(f"NAM model not found: {args.model}")
    payload = json.loads(args.model.read_text(encoding="utf-8"))
    if payload.get("architecture") == "SlimmableContainer":
        submodels = list(dict(payload.get("config", {})).get("submodels", []))
        if not submodels:
            raise SystemExit("NAM slimmable container has no submodels.")
        selected = max(submodels, key=lambda item: float(item.get("max_value", 0.0)))
        payload = dict(selected.get("model", {}))
        print(f"Selected highest-quality NAM A2 submodel: max_value={float(selected.get('max_value', 0.0)):.3f}")
    model = init_from_nam(payload)
    model.eval()
    model_rate = int(round(float(payload.get("sample_rate") or model.sample_rate or 48000)))
    source_rate, source = read_audio(args.input)
    source = resample(source, source_rate, model_rate)
    source *= float(10.0 ** (args.input_trim_db / 20.0))
    context = max(0, int(model.receptive_field) - 1)
    output = np.zeros(len(source), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(source), int(args.render_chunk_samples)):
            end = min(len(source), start + int(args.render_chunk_samples))
            history_start = max(0, start - context)
            segment = source[history_start:end]
            if start < context:
                segment = np.pad(segment, (context - start, 0))
            tensor = torch.from_numpy(np.asarray(segment, dtype=np.float32)).reshape(1, -1)
            prediction = model(tensor).detach().cpu().numpy().reshape(-1)
            output[start:end] = prediction[-(end - start) :]
    output *= float(10.0 ** (args.output_trim_db / 20.0))
    write_audio(args.output, model_rate, limited_audio(output))
    print(f"Wrote NAM reference render: {args.output}")
    print(
        f"Architecture={payload.get('architecture', 'unknown')} sample_rate={model_rate} "
        f"receptive_field={int(model.receptive_field)} device=cpu"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check")
    check.set_defaults(func=run_check)

    metrics = subparsers.add_parser("metrics")
    metrics.add_argument("--di", type=Path, required=True)
    metrics.add_argument("--target", type=Path, required=True)
    metrics.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--min-improvement-db", type=float, default=1.25)
    metrics.add_argument("--min-movement-db", type=float, default=0.75)
    metrics.add_argument("--max-listening-spectral-error-db", type=float, default=14.0)
    metrics.add_argument("--min-listening-correlation", type=float, default=0.50)
    metrics.add_argument("--max-listening-level-error-db", type=float, default=2.0)
    metrics.add_argument("--min-listening-section-pass-rate", type=float, default=0.60)
    metrics.add_argument("--audition-dir", type=Path, default=None)
    metrics.add_argument("--audition-seconds", type=float, default=10.0)
    metrics.set_defaults(func=run_metrics)

    nam_config = subparsers.add_parser("nam-config")
    nam_config.add_argument("--capture-manifest", type=Path, required=True)
    nam_config.add_argument("--output-dir", type=Path, required=True)
    nam_config.add_argument("--epochs", type=int, default=100)
    nam_config.add_argument("--batch-size", type=int, default=16)
    nam_config.add_argument("--window-samples", type=int, default=8192)
    nam_config.add_argument("--train-batches-per-epoch", type=int, default=0)
    nam_config.add_argument("--validation-batches-per-epoch", type=int, default=0)
    nam_config.set_defaults(func=run_nam_config)

    train = subparsers.add_parser("train-reference")
    train.add_argument("--capture-manifest", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--metrics-output", type=Path, required=True)
    train.add_argument("--architecture", choices=["tcn", "tcn-v2", "lstm", "gru"], default="tcn")
    train.add_argument("--channels", type=int, default=24)
    train.add_argument("--levels", type=int, default=9)
    train.add_argument("--tcn-stacks", type=int, default=2)
    train.add_argument("--hidden-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--steps-per-epoch", type=int, default=64)
    train.add_argument("--chunk-samples", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=0.0005)
    train.add_argument("--loss-profile", choices=["balanced-v1", "fullness-v2"], default="balanced-v1")
    train.add_argument("--print-every", type=int, default=5)
    train.add_argument("--seed", type=int, default=6505)
    train.add_argument("--cpu", action="store_true")
    train.add_argument("--min-improvement-db", type=float, default=1.25)
    train.add_argument("--min-movement-db", type=float, default=0.75)
    train.add_argument("--max-listening-spectral-error-db", type=float, default=14.0)
    train.add_argument("--min-listening-correlation", type=float, default=0.50)
    train.add_argument("--max-listening-level-error-db", type=float, default=2.0)
    train.add_argument("--min-listening-section-pass-rate", type=float, default=0.60)
    train.add_argument("--allow-failed-validation", action="store_true")
    train.set_defaults(func=run_train_reference)

    conditioned = subparsers.add_parser("train-conditioned-reference")
    conditioned.add_argument("--dataset-manifest", type=Path, required=True)
    conditioned.add_argument("--holdout-take", action="append", default=[])
    conditioned.add_argument("--model", type=Path, required=True)
    conditioned.add_argument("--output", type=Path, required=True)
    conditioned.add_argument("--metrics-output", type=Path, required=True)
    conditioned.add_argument("--architecture", choices=["tcn", "tcn-v2", "lstm", "gru"], default="tcn")
    conditioned.add_argument("--sample-rate", type=int, default=96000)
    conditioned.add_argument("--channels", type=int, default=24)
    conditioned.add_argument("--levels", type=int, default=12)
    conditioned.add_argument("--tcn-stacks", type=int, default=2)
    conditioned.add_argument("--hidden-size", type=int, default=48)
    conditioned.add_argument("--epochs", type=int, default=30)
    conditioned.add_argument("--steps-per-epoch", type=int, default=64)
    conditioned.add_argument("--chunk-samples", type=int, default=8192)
    conditioned.add_argument("--render-chunk-samples", type=int, default=65536)
    conditioned.add_argument("--learning-rate", type=float, default=0.0004)
    conditioned.add_argument("--loss-profile", choices=["balanced-v1", "fullness-v2"], default="balanced-v1")
    conditioned.add_argument("--focus-rig-fraction", type=float, default=0.70)
    conditioned.add_argument("--training-validation-fraction", type=float, default=0.10)
    conditioned.add_argument("--internal-validation-takes", type=int, default=8)
    conditioned.add_argument("--checkpoint-every", type=int, default=1)
    conditioned.add_argument("--frozen-manifest", type=Path, default=None)
    conditioned.add_argument("--audition-dir", type=Path, default=None)
    conditioned.add_argument("--audition-seconds", type=float, default=10.0)
    conditioned.add_argument("--print-every", type=int, default=5)
    conditioned.add_argument("--early-stopping-patience", type=int, default=10)
    conditioned.add_argument("--min-delta", type=float, default=1e-5)
    conditioned.add_argument("--seed", type=int, default=6505)
    conditioned.add_argument("--cpu", action="store_true")
    conditioned.add_argument("--min-improvement-db", type=float, default=1.25)
    conditioned.add_argument("--min-movement-db", type=float, default=0.75)
    conditioned.add_argument("--max-listening-spectral-error-db", type=float, default=14.0)
    conditioned.add_argument("--min-listening-correlation", type=float, default=0.50)
    conditioned.add_argument("--max-listening-level-error-db", type=float, default=2.0)
    conditioned.add_argument("--min-listening-section-pass-rate", type=float, default=0.60)
    conditioned.add_argument("--allow-failed-validation", action="store_true")
    conditioned.set_defaults(func=run_train_conditioned_reference)

    apply_reference = subparsers.add_parser("apply-reference")
    apply_reference.add_argument("--input", type=Path, required=True)
    apply_reference.add_argument("--model", type=Path, required=True)
    apply_reference.add_argument("--output", type=Path, required=True)
    apply_reference.add_argument("--input-trim-db", type=float, default=0.0)
    apply_reference.add_argument("--output-trim-db", type=float, default=0.0)
    apply_reference.add_argument("--cpu", action="store_true")
    apply_reference.add_argument("--allow-rejected", action="store_true")
    apply_reference.add_argument("--rig-fingerprint", default="")
    apply_reference.add_argument("--guitar", default="")
    apply_reference.add_argument("--tuning", default="")
    apply_reference.add_argument("--pickup", default="")
    apply_reference.add_argument("--pickup-mode", default="")
    apply_reference.add_argument("--guitar-volume", default="")
    apply_reference.add_argument("--guitar-tone", default="")
    apply_reference.set_defaults(func=run_apply_reference)

    apply_nam = subparsers.add_parser("apply-nam-reference")
    apply_nam.add_argument("--input", type=Path, required=True)
    apply_nam.add_argument("--model", type=Path, required=True)
    apply_nam.add_argument("--output", type=Path, required=True)
    apply_nam.add_argument("--input-trim-db", type=float, default=0.0)
    apply_nam.add_argument("--output-trim-db", type=float, default=0.0)
    apply_nam.add_argument("--render-chunk-samples", type=int, default=65536)
    apply_nam.set_defaults(func=run_apply_nam_reference)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
