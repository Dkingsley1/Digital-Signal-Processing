#!/usr/bin/env python3
"""Portable performance rigs inspired by practical hardware modeler workflows."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from cabinet_variant_workflow import apply_cabinet_variant_audio, filter_variant_output, load_cabinet_variant
from modeler_runtime import (
    MODELER_RUNTIME_VERSION,
    apply_dynamic_speaker,
    apply_speaker_impedance_response,
    blend_model_paths,
    destination_plan,
    load_snapshots,
    speaker_curve_config,
    validate_input_impedance,
)
from rig_capture_workflow import (
    db_to_linear,
    load_rig_model,
    peak_dbfs,
    predict_rig_model,
    read_audio,
    remove_dc,
    resample_audio,
    rms,
    write_audio,
)


PERFORMANCE_RIG_VERSION = "performance_rig_preset_1.1"
SUPPORTED_PERFORMANCE_RIG_VERSIONS = {"performance_rig_preset_1.0", PERFORMANCE_RIG_VERSION}
HEADRUSH_REFERENCE = "https://www.headrushfx.com/products/prime/"
UNIVERSAL_AUDIO_REFERENCE = "https://www.uaudio.com/products/ox-amp-top-box"
FRACTAL_REFERENCE = "https://www.fractalaudio.com/downloads/manuals/fas-guides/Fractal-Audio-Blocks-Guide.pdf"
BOSS_REFERENCE = "https://www.boss.info/us/products/gt-1000/support/"
LINE6_REFERENCE = "https://line6.com/support/manuals/helix"


def _portable_path(target: Path | None, preset_path: Path) -> str | None:
    if target is None:
        return None
    return os.path.relpath(target.resolve(), start=preset_path.resolve().parent)


def _resolve_path(value: str | None, preset_path: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (preset_path.resolve().parent / path).resolve()


def _accepted_model(path: Path) -> tuple[dict, dict]:
    metadata, params = load_rig_model(path)
    validation = dict(metadata.get("validation", {}))
    if not bool(validation.get("accepted", False)):
        raise SystemExit(f"Performance rigs require an accepted controlled model: {path}")
    return metadata, params


def _validate_gain(name: str, value: float, minimum: float = -24.0, maximum: float = 24.0) -> float:
    number = float(value)
    if not minimum <= number <= maximum:
        raise SystemExit(f"{name} must be between {minimum:g} and {maximum:g} dB.")
    return number


def _apply_input_gate(audio: np.ndarray, sample_rate: int, threshold_dbfs: float) -> np.ndarray:
    if threshold_dbfs <= -100.0:
        return np.asarray(audio, dtype=np.float64)
    if not -100.0 <= threshold_dbfs <= 0.0:
        raise SystemExit("Gate threshold must be between -100 and 0 dBFS.")
    source = np.asarray(audio, dtype=np.float64)
    detector = sosfilt(
        butter(1, 24.0, btype="lowpass", fs=sample_rate, output="sos"),
        np.square(source),
    )
    envelope = np.sqrt(np.maximum(detector, 0.0))
    threshold = db_to_linear(threshold_dbfs)
    fully_open = threshold * db_to_linear(12.0)
    gain = np.clip((envelope - threshold) / max(fully_open - threshold, 1e-12), 0.0, 1.0)
    gain = sosfilt(butter(1, 35.0, btype="lowpass", fs=sample_rate, output="sos"), gain)
    return source * np.clip(gain, 0.0, 1.0)


def _apply_three_band_tone(
    audio: np.ndarray,
    sample_rate: int,
    bass_db: float,
    middle_db: float,
    treble_db: float,
) -> np.ndarray:
    source = np.asarray(audio, dtype=np.float64)
    bass_db = _validate_gain("Bass", bass_db, -12.0, 12.0)
    middle_db = _validate_gain("Middle", middle_db, -12.0, 12.0)
    treble_db = _validate_gain("Treble", treble_db, -12.0, 12.0)
    if max(abs(bass_db), abs(middle_db), abs(treble_db)) < 1e-9:
        return source
    low = sosfilt(butter(2, 250.0, btype="lowpass", fs=sample_rate, output="sos"), source)
    high = sosfilt(butter(2, 2500.0, btype="highpass", fs=sample_rate, output="sos"), source)
    middle = source - low - high
    return (
        low * db_to_linear(bass_db)
        + middle * db_to_linear(middle_db)
        + high * db_to_linear(treble_db)
    )


def _render_model(
    audio: np.ndarray,
    input_rate: int,
    output_rate: int,
    model_path: Path,
    additional_input_trim_db: float,
    chunk_samples: int,
) -> tuple[np.ndarray, dict]:
    metadata, params = _accepted_model(model_path)
    model_rate = int(metadata["sample_rate_hz"])
    model_input = resample_audio(remove_dc(audio), input_rate, model_rate)
    stored_trim_db = float(metadata.get("application_input_trim_db", 0.0))
    effective_trim_db = stored_trim_db + float(additional_input_trim_db)
    model_input *= db_to_linear(effective_trim_db)
    prediction = predict_rig_model(model_input, metadata, params, chunk_samples=int(chunk_samples))
    return resample_audio(prediction, model_rate, output_rate), {
        "path": str(model_path),
        "stored_input_trim_db": stored_trim_db,
        "additional_input_trim_db": float(additional_input_trim_db),
        "effective_input_trim_db": effective_trim_db,
        "capture_type": str(metadata.get("capture_type", "amp-cab")),
    }


def _constant_level_morph(primary: np.ndarray, secondary: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    length = min(len(primary), len(secondary))
    first = np.asarray(primary[:length], dtype=np.float64)
    second = np.asarray(secondary[:length], dtype=np.float64)
    blended = ((1.0 - amount) * first) + (amount * second)
    expected_rms = ((1.0 - amount) * rms(first)) + (amount * rms(second))
    if expected_rms > 1e-12 and rms(blended) > 1e-12:
        blended *= expected_rms / rms(blended)
    return blended


def _validate_snapshot_values(
    snapshots: dict[str, dict],
    *,
    capture_type: str,
    has_full_cabinet: bool,
    has_secondary: bool,
    has_second_mic: bool,
) -> None:
    numeric_ranges = {
        "model_morph": (0.0, 1.0),
        "primary_level_db": (-24.0, 12.0),
        "secondary_level_db": (-24.0, 12.0),
        "input_trim_db": (-24.0, 24.0),
        "gate_threshold_dbfs": (-100.0, 0.0),
        "bass_db": (-12.0, 12.0),
        "middle_db": (-12.0, 12.0),
        "treble_db": (-12.0, 12.0),
        "speaker_drive": (0.0, 1.0),
        "cone_cry": (0.0, 1.0),
        "mic_position_morph": (0.0, 1.0),
        "output_trim_db": (-24.0, 24.0),
    }
    for name, settings in snapshots.items():
        for key, (minimum, maximum) in numeric_ranges.items():
            if key in settings:
                value = float(settings[key])
                if not minimum <= value <= maximum:
                    raise SystemExit(
                        f"Snapshot {name!r} setting {key} must be between {minimum:g} and {maximum:g}."
                    )
        if float(settings.get("model_morph", 0.0)) > 0.0 and not has_secondary:
            raise SystemExit(f"Snapshot {name!r} requests model morphing without a secondary model.")
        if float(settings.get("mic_position_morph", 0.0)) > 0.0 and not has_second_mic:
            raise SystemExit(f"Snapshot {name!r} requests mic-position morphing without a second measured profile.")
        if capture_type != "amp-preamp" and (
            float(settings.get("speaker_drive", 0.0)) > 0.0
            or float(settings.get("cone_cry", 0.0)) > 0.0
        ):
            raise SystemExit(f"Snapshot {name!r} cannot add a speaker stage to a {capture_type} capture.")
        if "destination" in settings:
            destination_plan(capture_type, str(settings["destination"]), has_full_cabinet=has_full_cabinet)


def _apply_cabinet_ir(
    audio: np.ndarray,
    sample_rate: int,
    ir_path: Path,
    ir_samples: int,
    mix: float,
    low_cut_hz: float,
    high_cut_hz: float,
) -> np.ndarray:
    ir_rate, impulse = read_audio(ir_path)
    if impulse.ndim == 2:
        impulse = np.mean(impulse, axis=1)
    if ir_rate != sample_rate:
        impulse = resample_audio(impulse, ir_rate, sample_rate)
    impulse = np.asarray(impulse[: int(ir_samples)], dtype=np.float64)
    if len(impulse) < 32 or float(np.max(np.abs(impulse))) < 1e-9:
        raise SystemExit(f"Cabinet IR is empty or too short: {ir_path}")
    dry = np.asarray(audio, dtype=np.float64)
    wet = fftconvolve(dry, impulse, mode="full")[: len(dry)]
    wet = filter_variant_output(wet, sample_rate, low_cut_hz, high_cut_hz)
    amount = float(np.clip(mix, 0.0, 1.0))
    return ((1.0 - amount) * dry) + (amount * wet)


def run_build_performance_rig(args) -> None:
    preset_path = Path(args.preset)
    primary_path = Path(args.model)
    primary_metadata, _ = _accepted_model(primary_path)
    secondary_path = Path(args.secondary_model) if args.secondary_model else None
    secondary_metadata = None
    if secondary_path is not None:
        secondary_metadata, _ = _accepted_model(secondary_path)
        if str(secondary_metadata.get("capture_type", "amp-cab")) != str(
            primary_metadata.get("capture_type", "amp-cab")
        ):
            raise SystemExit("Primary and secondary models must use the same capture type for a performance-rig morph.")
    morph = float(args.model_morph)
    if not 0.0 <= morph <= 1.0:
        raise SystemExit("--model-morph must be between zero and one.")
    if secondary_path is None and morph > 0.0:
        raise SystemExit("--model-morph requires --secondary-model.")

    path_mode = str(args.model_path_mode)
    if path_mode not in {"morph", "parallel"}:
        raise SystemExit("--model-path-mode must be morph or parallel.")
    primary_level_db = _validate_gain("Primary path level", args.primary_level_db, -24.0, 12.0)
    secondary_level_db = _validate_gain("Secondary path level", args.secondary_level_db, -24.0, 12.0)

    cabinet_variant = Path(args.cabinet_variant) if args.cabinet_variant else None
    cabinet_variant_b = Path(args.cabinet_variant_b) if args.cabinet_variant_b else None
    cabinet_ir = Path(args.cabinet_ir) if args.cabinet_ir else None
    if (cabinet_variant or cabinet_variant_b) and cabinet_ir:
        raise SystemExit("Choose measured cabinet variants or --cabinet-ir, not both.")
    if cabinet_variant_b and not cabinet_variant:
        raise SystemExit("--cabinet-variant-b requires --cabinet-variant as the first measured endpoint.")
    mic_position_morph = float(args.mic_position_morph)
    if not 0.0 <= mic_position_morph <= 1.0:
        raise SystemExit("--mic-position-morph must be between zero and one.")
    if mic_position_morph > 0.0 and not cabinet_variant_b:
        raise SystemExit("--mic-position-morph requires --cabinet-variant-b.")
    capture_type = str(primary_metadata.get("capture_type", "amp-cab"))
    if cabinet_ir and capture_type == "amp-cab":
        raise SystemExit(
            "A full cabinet IR cannot follow an Amp & Cab capture because that stacks two cabinets. "
            "Use an Amp / Pre-Amp capture or a measured --cabinet-variant correction."
        )
    if cabinet_variant:
        variant_metadata, _ = load_cabinet_variant(cabinet_variant)
        if not bool(dict(variant_metadata.get("validation", {})).get("accepted", False)):
            raise SystemExit(f"Cabinet variant has not passed validation: {cabinet_variant}")
    if cabinet_variant_b:
        variant_metadata_b, _ = load_cabinet_variant(cabinet_variant_b)
        if not bool(dict(variant_metadata_b.get("validation", {})).get("accepted", False)):
            raise SystemExit(f"Cabinet variant has not passed validation: {cabinet_variant_b}")
    if cabinet_ir and not cabinet_ir.exists():
        raise SystemExit(f"Cabinet IR not found: {cabinet_ir}")

    speaker = speaker_curve_config(
        str(args.speaker_impedance_curve),
        resonance_hz=float(args.speaker_resonance_hz),
        resonance_db=float(args.speaker_resonance_db),
        resonance_q=float(args.speaker_resonance_q),
        presence_hz=float(args.speaker_presence_hz),
        presence_db=float(args.speaker_presence_db),
        presence_q=float(args.speaker_presence_q),
    )
    speaker.update(
        {
            "dynamic_drive": float(args.dynamic_speaker_drive),
            "cone_cry": float(args.cone_cry),
            "cone_cry_hz": float(args.cone_cry_hz),
            "reference_level_dbfs": float(args.speaker_reference_level_dbfs),
            "implementation_note": (
                "Measured-model post stage with oversampled level dependence; it is not a physical reactive load."
            ),
        }
    )
    if not 0.0 <= speaker["dynamic_drive"] <= 1.0:
        raise SystemExit("--dynamic-speaker-drive must be between zero and one.")
    if not 0.0 <= speaker["cone_cry"] <= 1.0:
        raise SystemExit("--cone-cry must be between zero and one.")
    if not 700.0 <= speaker["cone_cry_hz"] <= 8000.0:
        raise SystemExit("--cone-cry-hz must be between 700 and 8000.")
    if not -60.0 <= speaker["reference_level_dbfs"] <= -3.0:
        raise SystemExit("--speaker-reference-level-dbfs must be between -60 and -3.")
    speaker_active = (
        str(speaker["name"]) != "flat"
        or speaker["dynamic_drive"] > 0.0
        or speaker["cone_cry"] > 0.0
    )
    if speaker_active and capture_type != "amp-preamp":
        raise SystemExit(
            "Speaker impedance and dynamic-speaker stages are only valid after an Amp / Pre-Amp capture. "
            "An Amp & Cab capture already contains its speaker response."
        )

    capture_manifest = dict(primary_metadata.get("capture_manifest") or {})
    measured_input_impedance = capture_manifest.get("input_impedance_kohm")
    requested_input_impedance = getattr(args, "input_impedance_kohm", None)
    input_impedance_kohm = float(
        requested_input_impedance
        if requested_input_impedance is not None
        else measured_input_impedance if measured_input_impedance is not None else 1000.0
    )
    validate_input_impedance(input_impedance_kohm, None, allow_mismatch=False)
    if measured_input_impedance is not None:
        validate_input_impedance(
            float(measured_input_impedance),
            input_impedance_kohm,
            allow_mismatch=False,
        )
    output_plan = destination_plan(
        capture_type,
        str(args.destination),
        has_full_cabinet=bool(cabinet_ir),
    )
    snapshots = load_snapshots(Path(args.snapshots_json) if args.snapshots_json else None)
    _validate_snapshot_values(
        snapshots,
        capture_type=capture_type,
        has_full_cabinet=bool(cabinet_ir),
        has_secondary=secondary_path is not None,
        has_second_mic=cabinet_variant_b is not None,
    )

    preset = {
        "preset_version": PERFORMANCE_RIG_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "name": str(args.name),
        "notes": str(args.notes),
        "workflow_inspiration": {
            "name": "Independent multi-modeler-inspired performance rig",
            "references": {
                "HeadRush": HEADRUSH_REFERENCE,
                "Universal Audio OX": UNIVERSAL_AUDIO_REFERENCE,
                "Fractal Audio": FRACTAL_REFERENCE,
                "BOSS AIRD": BOSS_REFERENCE,
                "Line 6 Helix": LINE6_REFERENCE,
            },
            "implemented_ideas": [
                "portable block preset",
                "accepted clone audition and refinement contract",
                "optional accepted-model morph or parallel graph",
                "oversampled level-dependent speaker behavior",
                "cabinet-linked resonance approximation and measured mic-position morphing",
                "destination-aware cabinet bypass and routing validation",
                "capture input-impedance validation and named snapshots",
                "separate cabinet IR routing, tone controls, cuts, normalization choice, and output protection",
            ],
            "compatibility_note": "Independent local format; no proprietary algorithms or preset formats are used.",
        },
        "capture_type": capture_type,
        "runtime_version": MODELER_RUNTIME_VERSION,
        "models": {
            "primary": _portable_path(primary_path, preset_path),
            "secondary": _portable_path(secondary_path, preset_path),
            "morph": morph,
            "path_mode": path_mode,
            "primary_level_db": primary_level_db,
            "secondary_level_db": secondary_level_db,
            "morph_law": "linear RMS-compensated morph or normalized equal-power parallel blend",
        },
        "input": {
            "trim_db": _validate_gain("Input trim", args.input_trim_db),
            "gate_threshold_dbfs": float(args.gate_threshold_dbfs),
            "impedance_kohm": input_impedance_kohm,
            "impedance_note": (
                "Metadata and mismatch guard only; physical pickup loading is set by interface hardware before A/D conversion."
            ),
        },
        "clone_tone": {
            "bass_db": _validate_gain("Bass", args.bass_db, -12.0, 12.0),
            "middle_db": _validate_gain("Middle", args.middle_db, -12.0, 12.0),
            "treble_db": _validate_gain("Treble", args.treble_db, -12.0, 12.0),
        },
        "cabinet": {
            "variant_profile": _portable_path(cabinet_variant, preset_path),
            "variant_profile_b": _portable_path(cabinet_variant_b, preset_path),
            "mic_position_morph": mic_position_morph,
            "ir_wav": _portable_path(cabinet_ir, preset_path),
            "ir_samples": int(args.cabinet_ir_samples),
            "mix": float(args.cabinet_mix),
            "low_cut_hz": float(args.cabinet_low_cut_hz),
            "high_cut_hz": float(args.cabinet_high_cut_hz),
        },
        "speaker": speaker,
        "output": {
            "sample_rate_hz": int(args.output_sample_rate),
            "destination": str(args.destination),
            "destination_label": output_plan["label"],
            "trim_db": _validate_gain("Output trim", args.output_trim_db),
            "normalize": str(args.normalize),
            "normalize_peak_dbfs": float(args.normalize_peak_dbfs),
            "limiter": str(args.limiter),
        },
        "snapshots": snapshots,
    }
    if not -100.0 <= float(args.gate_threshold_dbfs) <= 0.0:
        raise SystemExit("--gate-threshold-dbfs must be between -100 and 0.")
    if not 0.0 <= float(args.cabinet_mix) <= 1.0:
        raise SystemExit("--cabinet-mix must be between zero and one.")
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    preset_path.write_text(json.dumps(preset, indent=2), encoding="utf-8")
    print(f"Wrote performance rig: {preset_path}")
    print(
        f"Capture type: {capture_type} | graph={path_mode} | secondary={'on' if secondary_path else 'off'} | "
        f"model balance={morph:.2f} | destination={args.destination} | snapshots={len(snapshots)}"
    )


def run_apply_performance_rig(args) -> None:
    preset_path = Path(args.preset)
    if not preset_path.exists():
        raise SystemExit(f"Performance rig not found: {preset_path}")
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    if preset.get("preset_version") not in SUPPORTED_PERFORMANCE_RIG_VERSIONS:
        raise SystemExit(f"Unsupported performance rig version: {preset.get('preset_version')}")

    capture_type = str(preset.get("capture_type", "amp-cab"))
    snapshots = dict(preset.get("snapshots", {}))
    snapshot_name = getattr(args, "snapshot", None)
    snapshot = {}
    if snapshot_name:
        if snapshot_name not in snapshots:
            available = ", ".join(sorted(snapshots)) or "none"
            raise SystemExit(f"Snapshot {snapshot_name!r} not found. Available snapshots: {available}")
        snapshot = dict(snapshots[snapshot_name])

    input_rate, audio = read_audio(Path(args.input))
    if audio.ndim == 2:
        audio = audio[:, 0]
    output_config = dict(preset["output"])
    output_rate = int(args.output_sample_rate or output_config.get("sample_rate_hz") or input_rate)
    input_config = dict(preset["input"])
    captured_impedance = float(input_config.get("impedance_kohm", 1000.0))
    validate_input_impedance(
        captured_impedance,
        getattr(args, "source_input_impedance_kohm", None),
        allow_mismatch=bool(getattr(args, "allow_input_impedance_mismatch", False)),
    )
    gate_threshold = float(snapshot.get("gate_threshold_dbfs", input_config.get("gate_threshold_dbfs", -100.0)))
    gated = _apply_input_gate(remove_dc(audio), input_rate, gate_threshold)
    configured_input_trim = float(snapshot.get("input_trim_db", input_config.get("trim_db", 0.0)))
    additional_trim = configured_input_trim + float(args.input_trim_db)

    model_config = dict(preset["models"])
    primary_path = _resolve_path(model_config.get("primary"), preset_path)
    if primary_path is None:
        raise SystemExit("Performance rig has no primary model.")
    primary, primary_details = _render_model(
        gated,
        input_rate,
        output_rate,
        primary_path,
        additional_trim,
        int(args.chunk_samples),
    )
    configured_morph = snapshot.get("model_morph", model_config.get("morph", 0.0))
    morph = float(configured_morph if args.model_morph is None else args.model_morph)
    if not 0.0 <= morph <= 1.0:
        raise SystemExit("Model morph must be between zero and one.")
    secondary_details = None
    secondary_path = _resolve_path(model_config.get("secondary"), preset_path)
    secondary = None
    if morph > 0.0:
        if secondary_path is None:
            raise SystemExit("This performance rig has no secondary model to morph toward.")
        secondary, secondary_details = _render_model(
            gated,
            input_rate,
            output_rate,
            secondary_path,
            additional_trim,
            int(args.chunk_samples),
        )
    path_mode = str(model_config.get("path_mode", "morph"))
    primary_level_db = float(snapshot.get("primary_level_db", model_config.get("primary_level_db", 0.0)))
    secondary_level_db = float(snapshot.get("secondary_level_db", model_config.get("secondary_level_db", 0.0)))
    output = blend_model_paths(
        primary,
        secondary,
        mode=path_mode,
        balance=morph,
        primary_level_db=primary_level_db,
        secondary_level_db=secondary_level_db,
    )

    tone = dict(preset.get("clone_tone", {}))
    output = _apply_three_band_tone(
        output,
        output_rate,
        float(snapshot.get("bass_db", tone.get("bass_db", 0.0))),
        float(snapshot.get("middle_db", tone.get("middle_db", 0.0))),
        float(snapshot.get("treble_db", tone.get("treble_db", 0.0))),
    )

    cabinet = dict(preset.get("cabinet", {}))
    cabinet_variant = _resolve_path(cabinet.get("variant_profile"), preset_path)
    cabinet_variant_b = _resolve_path(cabinet.get("variant_profile_b"), preset_path)
    cabinet_ir = _resolve_path(cabinet.get("ir_wav"), preset_path)
    if cabinet_variant_b and cabinet_variant is None:
        raise SystemExit("Performance rig has a second measured mic profile without a first profile.")
    if cabinet_ir and capture_type == "amp-cab":
        raise SystemExit("Performance rig refuses a full cabinet IR after an Amp & Cab capture.")
    _validate_snapshot_values(
        {str(snapshot_name or "runtime"): snapshot},
        capture_type=capture_type,
        has_full_cabinet=bool(cabinet_ir),
        has_secondary=secondary_path is not None,
        has_second_mic=cabinet_variant_b is not None,
    )

    requested_destination = (
        getattr(args, "destination", None)
        or snapshot.get("destination")
        or output_config.get("destination", "studio-frfr")
    )
    route = destination_plan(capture_type, str(requested_destination), has_full_cabinet=bool(cabinet_ir))

    speaker = dict(preset.get("speaker", {}))
    speaker.setdefault("name", "flat")
    speaker.setdefault("resonance_hz", 100.0)
    speaker.setdefault("resonance_db", 0.0)
    speaker.setdefault("resonance_q", 1.2)
    speaker.setdefault("presence_hz", 2800.0)
    speaker.setdefault("presence_db", 0.0)
    speaker.setdefault("presence_q", 0.8)
    speaker_drive = float(snapshot.get("speaker_drive", speaker.get("dynamic_drive", 0.0)))
    cone_cry = float(snapshot.get("cone_cry", speaker.get("cone_cry", 0.0)))
    if capture_type != "amp-preamp" and (
        str(speaker.get("name", "flat")) != "flat" or speaker_drive > 0.0 or cone_cry > 0.0
    ):
        raise SystemExit("Performance rig refuses an added speaker stage after a capture that already contains a cabinet.")
    if route["apply_speaker_stage"]:
        output = apply_speaker_impedance_response(output, output_rate, speaker)
        output = apply_dynamic_speaker(
            output,
            output_rate,
            drive=speaker_drive,
            cone_cry=cone_cry,
            cone_cry_hz=float(speaker.get("cone_cry_hz", 2600.0)),
            reference_level_dbfs=float(speaker.get("reference_level_dbfs", -18.0)),
        )

    cabinet_mix = float(cabinet.get("mix", 1.0))
    if not 0.0 <= cabinet_mix <= 1.0:
        raise SystemExit("Cabinet mix must be between zero and one.")
    low_cut = float(cabinet.get("low_cut_hz", 0.0))
    high_cut = float(cabinet.get("high_cut_hz", 0.0))
    cabinet_label = "captured reference"
    mic_position_morph = float(snapshot.get("mic_position_morph", cabinet.get("mic_position_morph", 0.0)))
    if not 0.0 <= mic_position_morph <= 1.0:
        raise SystemExit("Mic-position morph must be between zero and one.")
    if not route["apply_cabinet_stage"]:
        cabinet_label = f"bypassed for {requested_destination}"
    elif cabinet_variant and cabinet_variant_b:
        variant_check_a, _ = load_cabinet_variant(cabinet_variant)
        variant_check_b, _ = load_cabinet_variant(cabinet_variant_b)
        for path, metadata in ((cabinet_variant, variant_check_a), (cabinet_variant_b, variant_check_b)):
            if not bool(dict(metadata.get("validation", {})).get("accepted", False)):
                raise SystemExit(f"Cabinet variant has not passed validation: {path}")
        dry_before_mics = np.asarray(output, dtype=np.float64)
        mic_a, metadata_a = apply_cabinet_variant_audio(
            dry_before_mics,
            output_rate,
            cabinet_variant,
            mix=1.0,
            low_cut_hz=0.0,
            high_cut_hz=0.0,
        )
        mic_b, metadata_b = apply_cabinet_variant_audio(
            dry_before_mics,
            output_rate,
            cabinet_variant_b,
            mix=1.0,
            low_cut_hz=0.0,
            high_cut_hz=0.0,
        )
        measured_position = _constant_level_morph(mic_a, mic_b, mic_position_morph)
        measured_position = filter_variant_output(measured_position, output_rate, low_cut, high_cut)
        output = ((1.0 - cabinet_mix) * dry_before_mics[: len(measured_position)]) + (
            cabinet_mix * measured_position
        )
        cabinet_label = (
            f"{metadata_a.get('name', cabinet_variant)} -> {metadata_b.get('name', cabinet_variant_b)} "
            f"at {mic_position_morph:.2f}"
        )
    elif route["apply_cabinet_stage"] and cabinet_variant:
        variant_check, _ = load_cabinet_variant(cabinet_variant)
        if not bool(dict(variant_check.get("validation", {})).get("accepted", False)):
            raise SystemExit(f"Cabinet variant has not passed validation: {cabinet_variant}")
        output, variant_metadata = apply_cabinet_variant_audio(
            output,
            output_rate,
            cabinet_variant,
            mix=cabinet_mix,
            low_cut_hz=low_cut,
            high_cut_hz=high_cut,
        )
        cabinet_label = str(variant_metadata.get("name", cabinet_variant))
    elif route["apply_cabinet_stage"] and cabinet_ir:
        output = _apply_cabinet_ir(
            output,
            output_rate,
            cabinet_ir,
            int(cabinet.get("ir_samples", 2048)),
            cabinet_mix,
            low_cut,
            high_cut,
        )
        cabinet_label = str(cabinet_ir)
    elif route["apply_cabinet_stage"] and (low_cut > 0.0 or high_cut > 0.0):
        output = filter_variant_output(output, output_rate, low_cut, high_cut)

    configured_output_trim = float(snapshot.get("output_trim_db", output_config.get("trim_db", 0.0)))
    output *= db_to_linear(configured_output_trim + float(args.output_trim_db))
    normalize_mode = str(output_config.get("normalize", "off"))
    if normalize_mode == "peak":
        target_peak = db_to_linear(float(output_config.get("normalize_peak_dbfs", -1.0)))
        output *= target_peak / max(float(np.max(np.abs(output))), 1e-12)
    peak = float(np.max(np.abs(output)) + 1e-12)
    limiter = str(output_config.get("limiter", "soft"))
    if peak >= 1.0:
        if limiter == "off":
            raise SystemExit("Performance rig output clips. Reduce input/output trim or enable the soft limiter.")
        output = np.tanh(output) / math.tanh(1.0)
        output *= 0.98
    write_audio(Path(args.output), output_rate, output)
    print(f"Applied performance rig: {preset.get('name', preset_path.stem)}")
    print(f"Primary model: {primary_details['path']}")
    if secondary_details:
        print(f"Secondary model: {secondary_details['path']} | mode={path_mode} | balance={morph:.2f}")
    if snapshot_name:
        print(f"Snapshot: {snapshot_name}")
    print(
        f"Input gate={gate_threshold:.1f} dBFS | additional input trim={additional_trim:+.1f} dB | "
        f"input impedance={captured_impedance:g} kOhm"
    )
    speaker_status = (
        f"drive={speaker_drive:.2f}, cone cry={cone_cry:.2f}"
        if route["apply_speaker_stage"]
        else "bypassed"
    )
    print(
        f"Destination={requested_destination} | speaker stage={speaker_status} | "
        f"cabinet={cabinet_label} | normalize={normalize_mode}"
    )
    print(f"Output: {args.output} | {output_rate} Hz | peak={peak_dbfs(output):.2f} dBFS")
