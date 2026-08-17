#!/usr/bin/env python3
"""Regression checks for HeadRush-inspired capture types and performance rigs."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from cabinet_variant_workflow import CABINET_VARIANT_VERSION, save_cabinet_variant  # noqa: E402
from modeler_runtime import (  # noqa: E402
    apply_dynamic_speaker,
    apply_speaker_impedance_response,
    destination_plan,
    speaker_curve_config,
)
from performance_rig_workflow import (  # noqa: E402
    PERFORMANCE_RIG_VERSION,
    run_apply_performance_rig,
    run_build_performance_rig,
)
from rig_capture_workflow import (  # noqa: E402
    capture_routing_lines,
    clone_control_map,
    init_tcn_params,
    numpy_model_params,
    read_audio,
    require_mlx,
    save_rig_model,
    tcn_receptive_field,
    write_audio,
)


def accepted_model(path: Path, seed: int, capture_type: str) -> None:
    mx = require_mlx()
    levels = 5
    params = numpy_model_params(init_tcn_params(mx, channels=4, levels=levels, seed=seed))
    metadata = {
        "model_version": "mlx_causal_rig_capture_1.0",
        "sample_rate_hz": 48000,
        "levels": levels,
        "channels": 4,
        "receptive_field_samples": tcn_receptive_field(levels),
        "input_scale": 1.0,
        "target_scale": 0.08,
        "application_input_trim_db": 0.0,
        "capture_type": capture_type,
        "validation": {"accepted": True, "failures": []},
    }
    save_rig_model(path, metadata, params)


def build_args(directory: Path, primary: Path, secondary: Path, cabinet_ir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        preset=directory / "performance_rigs" / "smoke.json",
        name="Smoke Performance Rig",
        model=primary,
        secondary_model=secondary,
        model_morph=0.35,
        model_path_mode="parallel",
        primary_level_db=0.0,
        secondary_level_db=-1.0,
        input_trim_db=-1.0,
        input_impedance_kohm=1000.0,
        gate_threshold_dbfs=-70.0,
        bass_db=1.0,
        middle_db=-0.5,
        treble_db=0.75,
        cabinet_variant=None,
        cabinet_variant_b=None,
        mic_position_morph=0.0,
        cabinet_ir=cabinet_ir,
        cabinet_ir_samples=1024,
        cabinet_mix=1.0,
        cabinet_low_cut_hz=55.0,
        cabinet_high_cut_hz=12000.0,
        speaker_impedance_curve="closed-back-1x12",
        speaker_resonance_hz=100.0,
        speaker_resonance_db=0.0,
        speaker_resonance_q=1.2,
        speaker_presence_hz=2800.0,
        speaker_presence_db=0.0,
        speaker_presence_q=0.8,
        dynamic_speaker_drive=0.25,
        cone_cry=0.10,
        cone_cry_hz=2600.0,
        speaker_reference_level_dbfs=-18.0,
        destination="studio-frfr",
        snapshots_json=None,
        output_sample_rate=48000,
        output_trim_db=-3.0,
        normalize="off",
        normalize_peak_dbfs=-1.0,
        limiter="soft",
        notes="synthetic smoke",
    )


def check_routing_and_controls() -> None:
    controls = clone_control_map(["pre_gain=5", "resonance=6"])
    assert controls == {"pre_gain": "5", "resonance": "6"}
    args = SimpleNamespace(output_channel=1, target_channel=2, mic="Shure SM57")
    assert "speaker-rated" in " ".join(capture_routing_lines("amp-preamp", args))
    assert "Pedal output" in " ".join(capture_routing_lines("pedal-only", args))


def accepted_variant(path: Path, name: str, impulse: np.ndarray) -> None:
    metadata = {
        "profile_version": CABINET_VARIANT_VERSION,
        "name": name,
        "sample_rate_hz": 48000,
        "variant": {
            "cabinet": "smoke cabinet",
            "microphone": "smoke microphone",
            "mic_position": name,
        },
        "validation": {"accepted": True, "failures": []},
    }
    save_cabinet_variant(path, metadata, impulse)


def check_modeler_runtime() -> None:
    sample_rate = 48000
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    source = 0.18 * np.sin(2.0 * np.pi * 110.0 * t)
    neutral_curve = speaker_curve_config(
        "flat",
        resonance_hz=100.0,
        resonance_db=0.0,
        resonance_q=1.2,
        presence_hz=2800.0,
        presence_db=0.0,
        presence_q=0.8,
    )
    assert np.array_equal(source, apply_speaker_impedance_response(source, sample_rate, neutral_curve))
    assert np.array_equal(
        source,
        apply_dynamic_speaker(
            source,
            sample_rate,
            drive=0.0,
            cone_cry=0.0,
            cone_cry_hz=2600.0,
            reference_level_dbfs=-18.0,
        ),
    )
    curve = speaker_curve_config(
        "closed-back-1x12",
        resonance_hz=100.0,
        resonance_db=0.0,
        resonance_q=1.2,
        presence_hz=2800.0,
        presence_db=0.0,
        presence_q=0.8,
    )
    resonant = apply_speaker_impedance_response(source, sample_rate, curve)
    dynamic = apply_dynamic_speaker(
        resonant,
        sample_rate,
        drive=0.35,
        cone_cry=0.15,
        cone_cry_hz=2600.0,
        reference_level_dbfs=-18.0,
    )
    assert np.all(np.isfinite(dynamic))
    assert not np.allclose(source, dynamic)
    assert destination_plan("amp-preamp", "amp-return", has_full_cabinet=False)["apply_cabinet_stage"] is False
    assert destination_plan("pedal-only", "amp-input", has_full_cabinet=False)["apply_cabinet_stage"] is False


def check_performance_rig() -> None:
    rng = np.random.default_rng(5150)
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        primary = directory / "primary.npz"
        secondary = directory / "secondary.npz"
        amp_cab = directory / "amp_cab.npz"
        cabinet_ir = directory / "cabinet_ir.wav"
        input_wav = directory / "input.wav"
        output_wav = directory / "output.wav"
        accepted_model(primary, seed=1, capture_type="amp-preamp")
        accepted_model(secondary, seed=2, capture_type="amp-preamp")
        accepted_model(amp_cab, seed=3, capture_type="amp-cab")
        impulse = np.zeros(1024, dtype=np.float64)
        impulse[:4] = [0.65, 0.22, 0.09, 0.04]
        write_audio(cabinet_ir, 48000, impulse)
        source = rng.normal(0.0, 0.035, 10000)
        write_audio(input_wav, 48000, source)

        args = build_args(directory, primary, secondary, cabinet_ir)
        snapshots_path = directory / "snapshots.json"
        snapshots_path.write_text(
            json.dumps(
                {
                    "Rhythm": {"model_morph": 0.2, "speaker_drive": 0.18, "output_trim_db": -3.0},
                    "Lead": {"model_morph": 0.65, "middle_db": 1.5, "output_trim_db": -1.0},
                }
            ),
            encoding="utf-8",
        )
        args.snapshots_json = snapshots_path
        run_build_performance_rig(args)
        preset = json.loads(Path(args.preset).read_text(encoding="utf-8"))
        assert preset["preset_version"] == PERFORMANCE_RIG_VERSION
        assert not Path(preset["models"]["primary"]).is_absolute()
        assert preset["models"]["secondary"]
        assert preset["models"]["path_mode"] == "parallel"
        assert preset["cabinet"]["ir_samples"] == 1024
        assert preset["speaker"]["dynamic_drive"] == 0.25
        assert preset["output"]["destination"] == "studio-frfr"
        assert set(preset["snapshots"]) == {"Lead", "Rhythm"}
        assert "Universal Audio OX" in preset["workflow_inspiration"]["references"]

        apply_args = SimpleNamespace(
            input=input_wav,
            preset=args.preset,
            output=output_wav,
            input_trim_db=0.0,
            output_trim_db=0.0,
            model_morph=None,
            snapshot="Lead",
            destination=None,
            source_input_impedance_kohm=1000.0,
            allow_input_impedance_mismatch=False,
            output_sample_rate=None,
            chunk_samples=1024,
        )
        run_apply_performance_rig(apply_args)
        output_rate, output = read_audio(output_wav)
        assert output_rate == 48000
        assert len(output) == len(source)
        assert np.all(np.isfinite(output))
        assert float(np.max(np.abs(output))) < 1.0
        assert not np.allclose(source, output)

        apply_args.destination = "amp-return"
        apply_args.snapshot = None
        apply_args.output = directory / "amp_return.wav"
        run_apply_performance_rig(apply_args)
        _, amp_return = read_audio(apply_args.output)
        assert np.all(np.isfinite(amp_return))

        apply_args.source_input_impedance_kohm = 100.0
        mismatch_rejected = False
        try:
            run_apply_performance_rig(apply_args)
        except SystemExit as exc:
            mismatch_rejected = "Input impedance mismatch" in str(exc)
        assert mismatch_rejected

        stacked_args = build_args(directory, amp_cab, amp_cab, cabinet_ir)
        stacked_args.preset = directory / "stacked.json"
        rejected = False
        try:
            run_build_performance_rig(stacked_args)
        except SystemExit as exc:
            rejected = "stacks two cabinets" in str(exc)
        assert rejected

        mic_a = directory / "mic_a.npz"
        mic_b = directory / "mic_b.npz"
        accepted_variant(mic_a, "cap edge", np.array([0.75, 0.18, 0.07]))
        accepted_variant(mic_b, "cone edge", np.array([0.50, 0.28, 0.14, 0.08]))
        mic_args = build_args(directory, amp_cab, amp_cab, cabinet_ir)
        mic_args.preset = directory / "measured_mic_rig.json"
        mic_args.cabinet_ir = None
        mic_args.cabinet_variant = mic_a
        mic_args.cabinet_variant_b = mic_b
        mic_args.mic_position_morph = 0.4
        mic_args.speaker_impedance_curve = "flat"
        mic_args.dynamic_speaker_drive = 0.0
        mic_args.cone_cry = 0.0
        mic_args.snapshots_json = None
        run_build_performance_rig(mic_args)
        mic_apply = SimpleNamespace(
            input=input_wav,
            preset=mic_args.preset,
            output=directory / "measured_mic.wav",
            input_trim_db=0.0,
            output_trim_db=0.0,
            model_morph=None,
            snapshot=None,
            destination=None,
            source_input_impedance_kohm=1000.0,
            allow_input_impedance_mismatch=False,
            output_sample_rate=None,
            chunk_samples=1024,
        )
        run_apply_performance_rig(mic_apply)
        _, measured_mic = read_audio(mic_apply.output)
        assert np.all(np.isfinite(measured_mic))


def main() -> None:
    check_routing_and_controls()
    check_modeler_runtime()
    check_performance_rig()
    print("Performance rig smoke checks passed.")


if __name__ == "__main__":
    main()
