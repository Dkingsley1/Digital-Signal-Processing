#!/usr/bin/env python3
"""Isolated PyTorch/NAM research routing for the production MLX tone system."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from rig_capture_workflow import align_calibrated_pair, json_write, read_audio, resample_audio
from rig_identity import rig_fingerprint, rig_identity_from_manifest


PROJECT_DIR = Path(__file__).resolve().parent
RESEARCH_LAUNCHER = PROJECT_DIR / "scripts/research_python.sh"
RESEARCH_WORKER = PROJECT_DIR / "scripts/research_audio_worker.py"
RESEARCH_MOUNT = Path("/Volumes/ToneCaptureResearch")
RESEARCH_IMAGE = Path("/Volumes/VIDEO/ToneCaptureResearch/ToneCaptureResearch.sparsebundle")
RESEARCH_WORKING_CAP_BYTES = 5 * 1024**3
SCHWAB_MARKER = "schwab_trading_bot"
MODEL_INFLUENCES = {
    "Neural Amp Modeler": "aligned input/output export and an independent reference benchmark",
    "NablAFx": "causal TCN/LSTM architecture recipes, perceptual losses, and parameter-conditioning schema",
    "Proteus": "knob/setting condition labels for repeat captures across a controlled rig",
    "AIDA-X": "compact recurrent LSTM reference candidate",
    "TONEX": "prepared excitation plus real-guitar pairs, latency/polarity calibration, and held-out comparison",
    "CODEX": "model exchange boundaries and modular amp/cabinet/post-capture routing",
    "Quad Cortex": "controlled multilevel probe, exact-rig capture, and blind validation",
    "Kemper": "real-guitar refinement and response-detail validation",
    "Fender Tone Master": "separate accepted cabinet/microphone response stage",
    "Two notes": "dual measured-mic, phase, room, and speaker post-capture stage",
}


def resolve(path: Path, base: Path | None = None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base is not None and (base / path).exists():
        return (base / path).resolve()
    return (PROJECT_DIR / path).resolve()


def reject_schwab_paths(paths: list[Path]) -> None:
    for path in paths:
        if SCHWAB_MARKER in str(path).lower():
            raise SystemExit("Refusing a Schwab project path: tone-modeling research must remain fully separate.")


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"Manifest not found: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read manifest {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Manifest must contain a JSON object: {resolved}")
    return payload


def mono(audio: np.ndarray) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    return np.mean(values, axis=1) if values.ndim == 2 else values


def distributed_audio_pairs(
    source: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    total_seconds: float = 10.0,
    segments: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    length = min(len(source), len(target))
    total_samples = min(length, int(round(total_seconds * sample_rate)))
    segment_samples = max(8192, total_samples // max(1, segments))
    segment_samples = min(segment_samples, length)
    starts = np.linspace(0, max(0, length - segment_samples), segments, dtype=np.int64)
    return [
        (
            np.asarray(source[start : start + segment_samples], dtype=np.float64),
            np.asarray(target[start : start + segment_samples], dtype=np.float64),
        )
        for start in starts
    ]


def write_research_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    clean = np.nan_to_num(np.asarray(audio, dtype=np.float64))
    peak = float(np.max(np.abs(clean)) + 1e-12)
    if peak > 1.0:
        raise SystemExit(f"Refusing clipped research audio: {path}")
    sf.write(path, clean, sample_rate, subtype="PCM_24")


def hardware_fields(manifest: dict) -> dict:
    return rig_identity_from_manifest(manifest)


def source_conditions(manifest: dict) -> dict:
    metadata = dict(manifest.get("take_metadata", {}))
    return {
        "guitar": str(metadata.get("guitar") or "unlabeled"),
        "tuning": str(metadata.get("tuning") or "unlabeled"),
        "pickup": str(metadata.get("pickup") or "unlabeled"),
        "pickup_mode": str(metadata.get("pickup_mode") or "unlabeled"),
        "guitar_volume": str(metadata.get("guitar_volume") or "unlabeled"),
        "guitar_tone": str(metadata.get("guitar_tone") or "unlabeled"),
        "performance": str(metadata.get("performance") or "unlabeled"),
    }


def run_research(args: list[str]) -> subprocess.CompletedProcess[str]:
    reject_schwab_paths([RESEARCH_LAUNCHER, RESEARCH_WORKER, *[Path(item) for item in args if "/" in item]])
    if not RESEARCH_LAUNCHER.exists():
        raise SystemExit(f"Research launcher not found: {RESEARCH_LAUNCHER}")
    command = [str(RESEARCH_LAUNCHER), str(RESEARCH_WORKER), *args]
    try:
        return subprocess.run(command, cwd=PROJECT_DIR, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Isolated research command failed with exit code {exc.returncode}.") from exc


def run_research_stack_check(_: object) -> None:
    if RESEARCH_MOUNT.exists():
        stats = os.statvfs(RESEARCH_MOUNT)
        used = (stats.f_blocks - stats.f_bfree) * stats.f_frsize
        print(
            f"Capped research environment: {used / 1024**3:.2f} / "
            f"{RESEARCH_WORKING_CAP_BYTES / 1024**3:.2f} GiB working use"
        )
    else:
        print(f"Research image will auto-mount from: {RESEARCH_IMAGE}")
    print("Isolation: /Volumes/ToneCaptureResearch only; Schwab paths are rejected.")
    run_research(["check"])


def run_prepare_research_capture(args) -> None:
    di_path = resolve(Path(args.di))
    target_path = resolve(Path(args.target))
    source_manifest_path = resolve(Path(args.manifest)) if args.manifest else None
    probe_manifest_path = resolve(Path(args.probe_manifest)) if args.probe_manifest else None
    output_dir = resolve(Path(args.output_dir)) / str(args.name)
    if not 0.05 <= float(args.validation_fraction) <= 0.50:
        raise SystemExit("--validation-fraction must be between 0.05 and 0.50.")
    if int(args.nam_epochs) < 0:
        raise SystemExit("--nam-epochs cannot be negative.")
    reject_schwab_paths([di_path, target_path, output_dir, *[p for p in [source_manifest_path, probe_manifest_path] if p]])
    source_rate, source = read_audio(di_path)
    target_rate, target = read_audio(target_path)
    source, target = mono(source), mono(target)
    if target_rate != source_rate:
        target = resample_audio(target, target_rate, source_rate)
    source, target, latency_samples, polarity = align_calibrated_pair(source, target, source_rate)
    if len(source) < int(source_rate * 3.0):
        raise SystemExit("A research capture needs at least three seconds of aligned audio.")
    source_manifest = load_json(source_manifest_path) if source_manifest_path else {}
    probe_manifest = load_json(probe_manifest_path) if probe_manifest_path else {}
    validation_start = int(round(len(source) * (1.0 - float(args.validation_fraction))))
    manifest_validation_start = int(probe_manifest.get("validation_start_sample", 0))
    if manifest_validation_start and manifest_validation_start < len(source):
        validation_start = manifest_validation_start
    validation_start = int(np.clip(validation_start, source_rate, len(source) - source_rate))

    files = {
        "aligned_input": output_dir / "aligned_input.wav",
        "aligned_target": output_dir / "aligned_amp_target.wav",
        "train_input": output_dir / "train_input.wav",
        "train_target": output_dir / "train_amp_target.wav",
        "validation_input": output_dir / "validation_input.wav",
        "validation_target": output_dir / "validation_amp_target.wav",
    }
    write_research_audio(files["aligned_input"], source_rate, source)
    write_research_audio(files["aligned_target"], source_rate, target)
    write_research_audio(files["train_input"], source_rate, source[:validation_start])
    write_research_audio(files["train_target"], source_rate, target[:validation_start])
    write_research_audio(files["validation_input"], source_rate, source[validation_start:])
    write_research_audio(files["validation_target"], source_rate, target[validation_start:])

    rig = hardware_fields(source_manifest)
    if not rig["sample_rate_hz"]:
        rig["sample_rate_hz"] = source_rate
    rig_id = rig_fingerprint(rig)
    capture_manifest_path = output_dir / "research_capture.json"
    payload = {
        "format": "tone_capture_research_pair_1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": str(args.name),
        "sample_rate_hz": source_rate,
        "source_files": {
            "di": str(di_path),
            "amp_target": str(target_path),
            "hardware_manifest": str(source_manifest_path) if source_manifest_path else None,
            "probe_manifest": str(probe_manifest_path) if probe_manifest_path else None,
        },
        "files": {name: str(path.resolve()) for name, path in files.items()},
        "alignment": {
            "latency_samples": float(latency_samples),
            "latency_ms": float(1000.0 * latency_samples / source_rate),
            "polarity": int(polarity),
        },
        "split": {
            "validation_start_sample": validation_start,
            "training_samples": validation_start,
            "validation_samples": len(source) - validation_start,
        },
        "rig_identity": rig,
        "rig_fingerprint": rig_id,
        "source_conditions": source_conditions(source_manifest),
        "data_policy": {
            "training_target": "recorded amplifier/cabinet/microphone waveform",
            "di_role": "input only; DI gain is never a training target",
            "take_sampling": "balanced when combined through build-conditioned-dataset",
            "acceptance": "held-out target improvement plus DI-gain-only movement guard",
        },
        "modeler_influences": MODEL_INFLUENCES,
    }
    json_write(capture_manifest_path, payload)

    nam_dir = output_dir / "nam"
    run_research(
        [
            "nam-config",
            "--capture-manifest", str(capture_manifest_path),
            "--output-dir", str(nam_dir),
            "--epochs", str(args.nam_epochs),
        ]
    )
    nam_command = shlex.join(
        [
            "scripts/research_python.sh",
            "--nam-full",
            str(nam_dir / "nam_data.json"),
            str(nam_dir / "nam_model.json"),
            str(nam_dir / "nam_learning.json"),
            str(nam_dir / "runs"),
            "--no-show",
            "--no-plots",
        ]
    )
    (output_dir / "NAM_REFERENCE_COMMAND.txt").write_text(
        "NAM is an independent reference trainer; its output is never auto-promoted.\n\n"
        f"Prepared input: {files['aligned_input']}\n"
        f"Prepared output: {files['aligned_target']}\n\n"
        f"Run the generated NAM 0.13 full-trainer configs:\n{nam_command}\n",
        encoding="utf-8",
    )
    json_write(
        output_dir / "nablafx_recipe.json",
        {
            "framework": "NablAFx-inspired benchmark recipe",
            "architectures": ["causal_tcn", "lstm"],
            "losses": ["ESR", "multi_resolution_STFT", "transient", "envelope"],
            "conditioning_schema": {**rig, **source_conditions(source_manifest)},
            "gray_box_modules": ["captured_amp_model", "accepted_cabinet_response", "virtual_studio"],
            "note": "NablAFx optional Frechet/CLAP evaluation is not loaded automatically to control disk use.",
        },
    )
    print(f"Prepared isolated research capture: {capture_manifest_path}")
    print(f"Aligned at {source_rate} Hz; latency={latency_samples:.2f} samples polarity={polarity:+d}")
    print(f"Rig fingerprint: {rig_id}")
    print("The DI is input only. Every training loss targets the recorded amp/cab/mic waveform.")


def run_prepare_conditioned_nam_reference(args) -> None:
    dataset_path = resolve(Path(args.dataset_manifest))
    output_dir = resolve(Path(args.output_dir))
    reject_schwab_paths([dataset_path, output_dir])
    dataset = load_json(dataset_path)
    if dataset.get("format") != "tone_capture_conditioned_dataset_1.0":
        raise SystemExit(f"Unsupported conditioned dataset: {dataset_path}")
    entries = list(dataset.get("entries", []))
    holdout = next(
        (entry for entry in entries if str(entry.get("take_name", "")) == str(args.holdout_take)),
        None,
    )
    if holdout is None:
        raise SystemExit(f"Holdout take not found: {args.holdout_take}")
    holdout_rig = str(holdout.get("rig_fingerprint", ""))
    train_entries = [
        entry
        for entry in entries
        if str(entry.get("rig_fingerprint", "")) == holdout_rig
        and str(entry.get("take_name", "")) != str(args.holdout_take)
    ]
    if not train_entries:
        raise SystemExit("NAM A2 preparation needs another recording from the exact holdout rig.")

    sample_rate = int(args.sample_rate)
    max_samples = int(round(float(args.max_pair_seconds) * sample_rate))
    silence = np.zeros(int(round(0.35 * sample_rate)), dtype=np.float64)
    train_sources = []
    train_targets = []
    pair_reports = []
    for entry in train_entries:
        di_path = resolve(Path(str(entry["di"])))
        target_path = resolve(Path(str(entry["amp_target"])))
        reject_schwab_paths([di_path, target_path])
        di_rate, source = read_audio(di_path)
        target_rate, target = read_audio(target_path)
        source, target = mono(source), mono(target)
        source = resample_audio(source, di_rate, sample_rate)
        target = resample_audio(target, target_rate, sample_rate)
        source, target, lag, polarity = align_calibrated_pair(source, target, sample_rate)
        length = min(len(source), len(target), max_samples)
        train_sources.extend([source[:length], silence])
        train_targets.extend([target[:length], silence])
        pair_reports.append(
            {
                "take_name": str(entry.get("take_name", "")),
                "di": str(di_path),
                "target": str(target_path),
                "samples": int(length),
                "latency_samples": float(lag),
                "polarity": int(polarity),
            }
        )

    holdout_di = resolve(Path(str(holdout["di"])))
    holdout_target = resolve(Path(str(holdout["amp_target"])))
    reject_schwab_paths([holdout_di, holdout_target])
    di_rate, validation_source = read_audio(holdout_di)
    target_rate, validation_target = read_audio(holdout_target)
    validation_source = resample_audio(mono(validation_source), di_rate, sample_rate)
    validation_target = resample_audio(mono(validation_target), target_rate, sample_rate)
    validation_source, validation_target, validation_lag, validation_polarity = align_calibrated_pair(
        validation_source,
        validation_target,
        sample_rate,
    )
    validation_length = min(len(validation_source), len(validation_target), max_samples)
    validation_source = validation_source[:validation_length]
    validation_target = validation_target[:validation_length]
    training_source = np.concatenate(train_sources)
    training_target = np.concatenate(train_targets)
    boundary_silence = np.zeros(sample_rate, dtype=np.float64)
    validation_start = len(training_source) + len(boundary_silence)
    aligned_source = np.concatenate([training_source, boundary_silence, validation_source])
    aligned_target = np.concatenate([training_target, boundary_silence, validation_target])

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "aligned_input": output_dir / "aligned_input.wav",
        "aligned_target": output_dir / "aligned_amp_target.wav",
        "train_input": output_dir / "train_input.wav",
        "train_target": output_dir / "train_amp_target.wav",
        "validation_input": output_dir / "validation_input.wav",
        "validation_target": output_dir / "validation_amp_target.wav",
    }
    write_research_audio(files["aligned_input"], sample_rate, aligned_source)
    write_research_audio(files["aligned_target"], sample_rate, aligned_target)
    write_research_audio(files["train_input"], sample_rate, training_source)
    write_research_audio(files["train_target"], sample_rate, training_target)
    write_research_audio(files["validation_input"], sample_rate, validation_source)
    write_research_audio(files["validation_target"], sample_rate, validation_target)
    nam_validation_pairs = []
    for index, (excerpt_source, excerpt_target) in enumerate(
        distributed_audio_pairs(validation_source, validation_target, sample_rate),
        start=1,
    ):
        excerpt_input = output_dir / f"nam_validation_input_{index:02d}.wav"
        excerpt_target_path = output_dir / f"nam_validation_target_{index:02d}.wav"
        write_research_audio(excerpt_input, sample_rate, excerpt_source)
        write_research_audio(excerpt_target_path, sample_rate, excerpt_target)
        nam_validation_pairs.append(
            {"input": str(excerpt_input.resolve()), "target": str(excerpt_target_path.resolve())}
        )
    capture_manifest = output_dir / "research_capture.json"
    payload = {
        "format": "tone_capture_conditioned_nam_a2_1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": str(args.name),
        "sample_rate_hz": sample_rate,
        "rig_fingerprint": holdout_rig,
        "training_pairs": pair_reports,
        "whole_take_holdout": {
            "take_name": str(args.holdout_take),
            "di": str(holdout_di),
            "target": str(holdout_target),
            "samples": int(validation_length),
            "latency_samples": float(validation_lag),
            "polarity": int(validation_polarity),
        },
        "files": {
            **{name: str(path.resolve()) for name, path in files.items()},
            "nam_validation_pairs": nam_validation_pairs,
        },
        "split": {
            "validation_start_sample": int(validation_start),
            "training_samples": int(len(training_source)),
            "validation_samples": int(validation_length),
            "policy": "whole recording excluded from NAM optimization",
        },
        "data_policy": {
            "training_target": "recorded amplifier/cabinet/microphone waveform",
            "di_role": "input only",
            "rig_policy": "exact fingerprint only",
        },
    }
    json_write(capture_manifest, payload)
    nam_dir = output_dir / "nam_a2"
    run_research(
        [
            "nam-config",
            "--capture-manifest", str(capture_manifest),
            "--output-dir", str(nam_dir),
            "--epochs", str(args.nam_epochs),
            "--batch-size", str(args.nam_batch_size),
            "--window-samples", str(args.nam_window_samples),
            "--train-batches-per-epoch", str(args.nam_train_batches_per_epoch),
            "--validation-batches-per-epoch", str(args.nam_validation_batches_per_epoch),
        ]
    )
    command = [
        str(RESEARCH_LAUNCHER),
        "--nam-full",
        str(nam_dir / "nam_data.json"),
        str(nam_dir / "nam_model.json"),
        str(nam_dir / "nam_learning.json"),
        str(nam_dir / "runs"),
        "--no-show",
        "--no-plots",
    ]
    (output_dir / "NAM_A2_TRAIN_COMMAND.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    print(
        f"Prepared NAM A2 exact-rig dataset: train_takes={len(train_entries)} "
        f"holdout={args.holdout_take} sample_rate={sample_rate}"
    )
    print(f"Capture manifest: {capture_manifest}")
    print(f"NAM A2 configs: {nam_dir}")
    if args.start_training:
        print("Starting NAM A2 full reference training...")
        try:
            subprocess.run(command, cwd=PROJECT_DIR, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"NAM A2 training failed with exit code {exc.returncode}.") from exc


def run_train_torch_reference(args) -> None:
    capture_manifest = resolve(Path(args.capture_manifest))
    model = resolve(Path(args.model))
    output = resolve(Path(args.output))
    metrics_output = resolve(Path(args.metrics_output))
    reject_schwab_paths([capture_manifest, model, output, metrics_output])
    command = [
        "train-reference",
        "--capture-manifest", str(capture_manifest),
        "--model", str(model),
        "--output", str(output),
        "--metrics-output", str(metrics_output),
        "--architecture", str(args.architecture),
        "--channels", str(args.channels),
        "--levels", str(args.levels),
        "--tcn-stacks", str(args.tcn_stacks),
        "--hidden-size", str(args.hidden_size),
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--chunk-samples", str(args.chunk_samples),
        "--learning-rate", str(args.learning_rate),
        "--loss-profile", str(args.loss_profile),
        "--print-every", str(args.print_every),
        "--seed", str(args.seed),
        "--min-improvement-db", str(args.min_improvement_db),
        "--min-movement-db", str(args.min_movement_db),
        "--max-listening-spectral-error-db", str(args.max_listening_spectral_error_db),
        "--min-listening-correlation", str(args.min_listening_correlation),
        "--max-listening-level-error-db", str(args.max_listening_level_error_db),
        "--min-listening-section-pass-rate", str(args.min_listening_section_pass_rate),
    ]
    if args.cpu:
        command.append("--cpu")
    if args.allow_failed_validation:
        command.append("--allow-failed-validation")
    run_research(command)


def run_train_conditioned_torch_reference(args) -> None:
    dataset_manifest = resolve(Path(args.dataset_manifest))
    frozen_manifest = resolve(Path(args.frozen_manifest)) if args.frozen_manifest else None
    model = resolve(Path(args.model))
    output = resolve(Path(args.output))
    metrics_output = resolve(Path(args.metrics_output))
    reject_schwab_paths([dataset_manifest, model, output, metrics_output, *([frozen_manifest] if frozen_manifest else [])])
    if frozen_manifest:
        verified = verify_frozen_dataset(frozen_manifest, expected_dataset=dataset_manifest)
        print(
            f"Verified frozen dataset before research launch: "
            f"pairs={verified['pair_count']} assets={verified['asset_count']}"
        )
    command = [
        "train-conditioned-reference",
        "--dataset-manifest", str(dataset_manifest),
        "--model", str(model),
        "--output", str(output),
        "--metrics-output", str(metrics_output),
        "--architecture", str(args.architecture),
        "--sample-rate", str(args.sample_rate),
        "--channels", str(args.channels),
        "--levels", str(args.levels),
        "--tcn-stacks", str(args.tcn_stacks),
        "--hidden-size", str(args.hidden_size),
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--chunk-samples", str(args.chunk_samples),
        "--render-chunk-samples", str(args.render_chunk_samples),
        "--learning-rate", str(args.learning_rate),
        "--loss-profile", str(args.loss_profile),
        "--focus-rig-fraction", str(args.focus_rig_fraction),
        "--training-validation-fraction", str(args.training_validation_fraction),
        "--internal-validation-takes", str(args.internal_validation_takes),
        "--checkpoint-every", str(args.checkpoint_every),
        "--print-every", str(args.print_every),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--min-delta", str(args.min_delta),
        "--seed", str(args.seed),
        "--min-improvement-db", str(args.min_improvement_db),
        "--min-movement-db", str(args.min_movement_db),
        "--max-listening-spectral-error-db", str(args.max_listening_spectral_error_db),
        "--min-listening-correlation", str(args.min_listening_correlation),
        "--max-listening-level-error-db", str(args.max_listening_level_error_db),
        "--min-listening-section-pass-rate", str(args.min_listening_section_pass_rate),
    ]
    if frozen_manifest:
        command.extend(["--frozen-manifest", str(frozen_manifest)])
    if args.audition_dir:
        audition_dir = resolve(Path(args.audition_dir))
        reject_schwab_paths([audition_dir])
        command.extend(["--audition-dir", str(audition_dir), "--audition-seconds", str(args.audition_seconds)])
    for take_name in args.holdout_take:
        command.extend(["--holdout-take", str(take_name)])
    if args.cpu:
        command.append("--cpu")
    if args.allow_failed_validation:
        command.append("--allow-failed-validation")
    run_research(command)


def run_apply_torch_reference(args) -> None:
    input_path = resolve(Path(args.input))
    model = resolve(Path(args.model))
    output = resolve(Path(args.output))
    reject_schwab_paths([input_path, model, output])
    command = [
        "apply-reference",
        "--input", str(input_path),
        "--model", str(model),
        "--output", str(output),
        "--input-trim-db", str(args.input_trim_db),
        "--output-trim-db", str(args.output_trim_db),
    ]
    if args.cpu:
        command.append("--cpu")
    if args.allow_rejected:
        command.append("--allow-rejected")
    for option, value in (
        ("--rig-fingerprint", args.rig_fingerprint),
        ("--guitar", args.guitar),
        ("--tuning", args.tuning),
        ("--pickup", args.pickup),
        ("--pickup-mode", args.pickup_mode),
        ("--guitar-volume", args.guitar_volume),
        ("--guitar-tone", args.guitar_tone),
    ):
        if str(value or ""):
            command.extend([option, str(value)])
    run_research(command)


def run_apply_nam_reference(args) -> None:
    input_path = resolve(Path(args.input))
    model = resolve(Path(args.model))
    output = resolve(Path(args.output))
    reject_schwab_paths([input_path, model, output])
    run_research(
        [
            "apply-nam-reference",
            "--input", str(input_path),
            "--model", str(model),
            "--output", str(output),
            "--input-trim-db", str(args.input_trim_db),
            "--output-trim-db", str(args.output_trim_db),
            "--render-chunk-samples", str(args.render_chunk_samples),
        ]
    )


def run_hybrid_model_compare(args) -> None:
    di = resolve(Path(args.di))
    target = resolve(Path(args.target))
    output = resolve(Path(args.output))
    candidate_values = []
    candidate_paths = []
    for value in args.candidate:
        if "=" not in value:
            raise SystemExit("Each --candidate must use NAME=PATH format.")
        name, raw_path = value.split("=", 1)
        path = resolve(Path(raw_path))
        candidate_paths.append(path)
        candidate_values.append(f"{name}={path}")
    reject_schwab_paths([di, target, output, *candidate_paths])
    command = [
        "metrics",
        "--di", str(di),
        "--target", str(target),
        "--output", str(output),
        "--min-improvement-db", str(args.min_improvement_db),
        "--min-movement-db", str(args.min_movement_db),
        "--max-listening-spectral-error-db", str(args.max_listening_spectral_error_db),
        "--min-listening-correlation", str(args.min_listening_correlation),
        "--max-listening-level-error-db", str(args.max_listening_level_error_db),
        "--min-listening-section-pass-rate", str(args.min_listening_section_pass_rate),
    ]
    if args.audition_dir:
        audition_dir = resolve(Path(args.audition_dir))
        reject_schwab_paths([audition_dir])
        command.extend(["--audition-dir", str(audition_dir), "--audition-seconds", str(args.audition_seconds)])
    for value in candidate_values:
        command.extend(["--candidate", value])
    run_research(command)
    print(f"Wrote hybrid MLX/PyTorch/NAM comparison: {output}")


def dataset_take_paths(dataset_path: Path, take: dict) -> tuple[Path, Path, Path | None]:
    base = dataset_path.parent
    di = resolve(Path(str(take.get("clean_di_wav", ""))), base=base)
    target = resolve(Path(str(take.get("amp_mic_target_wav", ""))), base=base)
    manifest_value = str(take.get("hardware_manifest", ""))
    manifest = resolve(Path(manifest_value), base=base) if manifest_value else None
    return di, target, manifest


def run_build_conditioned_dataset(args) -> None:
    dataset_path = resolve(Path(args.dataset))
    output = resolve(Path(args.output))
    reject_schwab_paths([dataset_path, output])
    dataset = load_json(dataset_path)
    entries = []
    missing = []
    seen_di: set[Path] = set()

    def add_take(take: dict, index: int) -> None:
        if not isinstance(take, dict):
            return
        cleanup_status = dict(take.get("cleanup_status", {}))
        if bool(take.get("inactive_for_training", False) or take.get("archived", False)):
            return
        if str(cleanup_status.get("action", "")) in {"archived", "deleted"}:
            return
        di, target, manifest_path = dataset_take_paths(dataset_path, take)
        if di in seen_di:
            return
        if not di.exists() or not target.exists():
            missing.append(str(take.get("take_name") or f"take_{index}"))
            return
        if di.name.startswith("level_test"):
            return
        seen_di.add(di)
        manifest = load_json(manifest_path) if manifest_path and manifest_path.exists() else take
        rig = hardware_fields(manifest)
        if not rig["sample_rate_hz"]:
            rig["sample_rate_hz"] = int(dict(take.get("audio_interface", {})).get("sample_rate_hz") or 0)
        conditions = source_conditions(manifest)
        entries.append(
            {
                "take_name": str(take.get("take_name") or di.stem),
                "di": str(di),
                "amp_target": str(target),
                "hardware_manifest": str(manifest_path) if manifest_path else None,
                "rig_fingerprint": rig_fingerprint(rig),
                "rig_identity": rig,
                "conditions": conditions,
                "sampling_weight": 1.0,
                "target_role": "recorded_amp_cab_microphone",
            }
        )

    for index, take in enumerate(dataset.get("takes", []), start=1):
        add_take(take, index)

    discovered_count = 0
    for di in sorted((PROJECT_DIR / "recordings").glob("*_clean_di.wav")):
        if di.name.startswith("level_test") or di.resolve() in seen_di:
            continue
        target = di.with_name(di.name.replace("_clean_di.wav", "_amp_mic_target.wav"))
        manifest = di.with_name(di.name.replace("_clean_di.wav", "_hardware_manifest.json"))
        if not target.exists():
            missing.append(di.name.replace("_clean_di.wav", ""))
            continue
        add_take(
            {
                "take_name": di.name.replace("_clean_di.wav", ""),
                "clean_di_wav": str(di),
                "amp_mic_target_wav": str(target),
                "hardware_manifest": str(manifest) if manifest.exists() else "",
            },
            len(entries) + 1,
        )
        discovered_count += 1
    if not entries:
        raise SystemExit("No complete DI/amp-target pairs were found in the dataset.")
    group_counts = Counter(entry["rig_fingerprint"] for entry in entries)
    if len(group_counts) > 1 and not args.allow_mixed_rigs:
        details = ", ".join(f"{name}:{count}" for name, count in group_counts.most_common())
        raise SystemExit(
            "Dataset contains multiple exact rigs. Re-record with one fixed rig, select one group, or use "
            f"--allow-mixed-rigs for explicitly conditioned research. Groups: {details}"
        )
    condition_values = {}
    for key in source_conditions({}).keys():
        condition_values[key] = sorted({entry["conditions"][key] for entry in entries})
    payload = {
        "format": "tone_capture_conditioned_dataset_1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(dataset_path),
        "pair_count": len(entries),
        "pairs_discovered_outside_dataset": discovered_count,
        "missing_pair_count": len(missing),
        "missing_takes": missing,
        "rig_groups": dict(group_counts),
        "mixed_rigs_enabled": bool(args.allow_mixed_rigs),
        "condition_values": condition_values,
        "sampling_policy": "balanced per take so long recordings cannot dominate",
        "training_policy": "DI is input only; every loss and guard targets the recorded amp/cab/mic channel",
        "entries": entries,
        "modeler_influences": MODEL_INFLUENCES,
    }
    json_write(output, payload)
    print(f"Wrote conditioned all-recordings dataset: {output}")
    print(f"Complete pairs: {len(entries)} | exact-rig groups: {len(group_counts)} | missing pairs: {len(missing)}")
    print("Every take has equal sampling weight; no take is converted into a DI-gain training target.")


def file_sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_dataset_assets(dataset_manifest: Path, dataset: dict) -> list[dict]:
    assets = []
    for entry in dataset.get("entries", []):
        take_name = str(entry.get("take_name", "unnamed_take"))
        for role, key in (("clean_di", "di"), ("amp_target", "amp_target"), ("hardware_manifest", "hardware_manifest")):
            raw_path = entry.get(key)
            if not raw_path:
                if role == "hardware_manifest":
                    continue
                raise SystemExit(f"Frozen dataset entry is missing {key}: {take_name}")
            path = resolve(Path(str(raw_path)), base=dataset_manifest.parent)
            if not path.exists():
                raise SystemExit(f"Cannot freeze missing {role} for {take_name}: {path}")
            assets.append(
                {
                    "take_name": take_name,
                    "role": role,
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": file_sha256(path),
                }
            )
    return assets


def verify_frozen_dataset(frozen_manifest: Path, expected_dataset: Path | None = None) -> dict:
    frozen = load_json(frozen_manifest)
    if frozen.get("format") != "tone_capture_frozen_dataset_1.0":
        raise SystemExit(f"Unsupported frozen dataset manifest: {frozen_manifest}")
    source_manifest = resolve(Path(str(frozen.get("source_manifest", ""))), base=frozen_manifest.parent)
    if expected_dataset is not None and source_manifest != expected_dataset.resolve():
        raise SystemExit(
            f"Frozen dataset source mismatch: expected {expected_dataset.resolve()}, found {source_manifest}"
        )
    if not source_manifest.exists():
        raise SystemExit(f"Frozen source manifest is missing: {source_manifest}")
    current_manifest_hash = file_sha256(source_manifest)
    if current_manifest_hash != str(frozen.get("source_manifest_sha256", "")):
        raise SystemExit("Frozen conditioned dataset changed after it was approved; rebuild the freeze manifest deliberately.")

    verified = 0
    for asset in frozen.get("assets", []):
        path = Path(str(asset.get("path", ""))).expanduser().resolve()
        reject_schwab_paths([path])
        if not path.exists():
            raise SystemExit(f"Frozen recording asset is missing: {path}")
        if path.stat().st_size != int(asset.get("size_bytes", -1)):
            raise SystemExit(f"Frozen recording asset size changed: {path}")
        if file_sha256(path) != str(asset.get("sha256", "")):
            raise SystemExit(f"Frozen recording asset content changed: {path}")
        verified += 1
    if verified != int(frozen.get("asset_count", -1)):
        raise SystemExit("Frozen dataset asset count does not match its manifest.")
    return {"pair_count": int(frozen.get("pair_count", 0)), "asset_count": verified}


def run_freeze_conditioned_dataset(args) -> None:
    dataset_manifest = resolve(Path(args.dataset_manifest))
    output = resolve(Path(args.output))
    reject_schwab_paths([dataset_manifest, output])
    dataset = load_json(dataset_manifest)
    if dataset.get("format") != "tone_capture_conditioned_dataset_1.0":
        raise SystemExit(f"Unsupported conditioned dataset: {dataset_manifest}")
    entries = list(dataset.get("entries", []))
    if not entries or len(entries) != int(dataset.get("pair_count", -1)):
        raise SystemExit("Conditioned dataset pair count is incomplete; rebuild it before freezing.")
    if int(dataset.get("missing_pair_count", 0)) != 0:
        raise SystemExit("Refusing to freeze a conditioned dataset with missing recording pairs.")

    print(f"Hashing {len(entries)} approved DI/amp pairs without copying audio...")
    assets = frozen_dataset_assets(dataset_manifest, dataset)
    canonical_assets = json.dumps(assets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        "format": "tone_capture_frozen_dataset_1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(dataset_manifest),
        "source_manifest_sha256": file_sha256(dataset_manifest),
        "pair_count": len(entries),
        "asset_count": len(assets),
        "asset_bytes": int(sum(item["size_bytes"] for item in assets)),
        "asset_set_sha256": hashlib.sha256(canonical_assets).hexdigest(),
        "policy": {
            "audio_copied": False,
            "training_target": "recorded amplifier/cabinet/microphone waveform",
            "verification": "source manifest, byte size, and SHA-256 must match before accuracy-lane training",
        },
        "assets": assets,
    }
    json_write(output, payload)
    verified = verify_frozen_dataset(output, expected_dataset=dataset_manifest)
    print(
        f"Wrote frozen modeling dataset: {output}\n"
        f"Pairs: {verified['pair_count']} | assets: {verified['asset_count']} | copied audio: 0 bytes"
    )
