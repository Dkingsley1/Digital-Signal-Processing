#!/usr/bin/env python3
"""Guided PyCharm runner for controlled fixed-rig neural captures."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from tone_capture_engine import main


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CAPTURE = "peavey_6505_rhythm_maxon808_sm57"
DEFAULT_PROBE = Path("rig_captures/probes/rig_probe_96k.wav")


def ask(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def print_command(args: list[str]) -> None:
    command = [str(PROJECT_DIR / ".venv/bin/python"), "tone_capture_engine.py", *args]
    print(f"\n{shlex.join(command)}\n")


def generate_probe() -> None:
    args = [
        "rig-probe-generate",
        "--output", str(DEFAULT_PROBE),
        "--manifest", "rig_captures/probes/rig_probe_96k_manifest.json",
        "--sample-rate", "96000",
        "--peak-dbfs", "-18",
    ]
    print_command(args)
    main(args)


def record_probe(dry_run: bool) -> None:
    capture_name = ask("Fixed-rig capture name", DEFAULT_CAPTURE)
    capture_type = ask("Capture type (amp-cab/amp-preamp/pedal-only)", "amp-cab").lower()
    if capture_type not in {"amp-cab", "amp-preamp", "pedal-only"}:
        raise SystemExit("Capture type must be amp-cab, amp-preamp, or pedal-only.")
    input_device = ask("Input device index/name (blank = system default)")
    output_device = ask("Output device index/name (blank = system default)")
    send_trim = ask("Digital send trim in dB", "0")
    input_impedance = ask("Interface guitar-input impedance kOhm", "1000")
    amp_settings = ask("Amp settings", "rhythm channel; document every knob before capture")
    pedal = ask("Pedal/settings", "Maxon OD808: drive 0, tone 5, balance 10")
    mic_position = ask("Mic position", "SM57 close, directly in front of speaker")
    control_text = ask(
        "Physical controls as comma-separated NAME=VALUE entries",
        "pre_gain=5,low=5,mid=5,high=5,post_gain=5",
    )

    args = [
        "rig-probe-record",
        "--probe", str(DEFAULT_PROBE),
        "--probe-manifest", "rig_captures/probes/rig_probe_96k_manifest.json",
        "--capture-name", capture_name,
        "--output-dir", "rig_captures",
        "--input-channels", "2",
        "--target-channel", "2",
        "--output-channels", "2",
        "--output-channel", "1",
        "--send-trim-db", send_trim,
        "--input-impedance-kohm", input_impedance,
        "--capture-type", capture_type,
        "--pedal", pedal,
        "--amp", "Peavey 6505 Mini Head",
        "--amp-settings", amp_settings,
        "--cabinet", "Egnater Tweaker 1x12 Celestion",
        "--mic", "Shure SM57",
        "--mic-position", mic_position,
    ]
    for control in [item.strip() for item in control_text.split(",") if item.strip()]:
        args.extend(["--clone-control", control])
    if input_device:
        args.extend(["--input-device", input_device])
    if output_device:
        args.extend(["--output-device", output_device])
    if dry_run:
        args.append("--dry-run")
        print_command(args)
        main(args)
        return

    print("\nRequired routing:")
    if capture_type == "amp-cab":
        print("  Interface line output 1 -> proper reamp box -> pedal/amp input")
        print("  Amplifier speaker output -> compatible speaker cabinet only")
        print("  Microphone -> interface input 2")
    elif capture_type == "amp-preamp":
        print("  Interface line output 1 -> proper reamp box -> amp/preamp input")
        print("  Approved preamp/FX-send line output -> interface input 2")
        print("  Never connect an amplifier speaker output directly to the interface")
    else:
        print("  Interface line output 1 -> proper reamp box -> pedal input")
        print("  Pedal output -> suitable interface instrument/line input 2")
    print("  Start with the monitor volume low; the probe contains sweeps and noise.")
    print_command(args)
    if ask('Type "PLAY PROBE" to confirm the reamp routing', "cancel") != "PLAY PROBE":
        print("Capture cancelled. No audio was played.")
        return
    args.append("--confirm-reamp-routing")
    main(args)


def train_capture() -> None:
    capture_name = ask("Fixed-rig capture name", DEFAULT_CAPTURE)
    args = [
        "train-rig-capture",
        "--probe", f"rig_captures/{capture_name}_probe_input.wav",
        "--target", f"rig_captures/{capture_name}_target_return.wav",
        "--capture-manifest", f"rig_captures/{capture_name}_rig_capture_manifest.json",
        "--model", f"profiles/{capture_name}_causal_rig_96k.npz",
        "--output", f"outputs/{capture_name}_validation_render.wav",
        "--comparison-output", f"outputs/{capture_name}_target_then_model_10s.wav",
        "--render-sample-rate", "96000",
        "--oversample-factor", "2",
        "--levels", "13",
        "--epochs", "100",
    ]
    print_command(args)
    if ask("Start MLX training? (yes/no)", "yes").lower() not in {"y", "yes"}:
        print("Training cancelled.")
        return
    main(args)


def newest_di() -> Path | None:
    takes = list((PROJECT_DIR / "recordings").glob("*_clean_di.wav"))
    return max(takes, key=lambda path: path.stat().st_mtime) if takes else None


def newest_cabinet_variant() -> Path | None:
    variants = list((PROJECT_DIR / "profiles" / "cabinet_variants").glob("*.npz"))
    accepted = [path for path in variants if ".rejected." not in path.name]
    return max(accepted, key=lambda path: path.stat().st_mtime) if accepted else None


def newest_performance_rig() -> Path | None:
    presets = list((PROJECT_DIR / "profiles" / "performance_rigs").glob("*.json"))
    return max(presets, key=lambda path: path.stat().st_mtime) if presets else None


def matching_target(di_path: Path) -> Path:
    name = di_path.name
    if not name.endswith("_clean_di.wav"):
        return di_path.with_name(f"{di_path.stem}_amp_mic_target.wav")
    return di_path.with_name(name.replace("_clean_di.wav", "_amp_mic_target.wav"))


def refine_capture() -> None:
    capture_name = ask("Fixed-rig capture name", DEFAULT_CAPTURE)
    latest = newest_di()
    di_default = str(latest.relative_to(PROJECT_DIR)) if latest else "recordings/refine_clean_di.wav"
    di_path = Path(ask("Refinement clean DI WAV", di_default))
    target_default = str(matching_target(di_path))
    target_path = ask("Matching simultaneous amp/mic WAV", target_default)
    print("\nThe pedal, amp knobs, cabinet, mic position, interface gains, and sample rate")
    print("must match the controlled probe capture. The final part of the take must contain playing.")
    if ask("Confirm this is the exact same fixed rig (yes/no)", "no").lower() not in {"y", "yes"}:
        print("Refinement cancelled; the base model was not changed.")
        return
    args = [
        "refine-rig-capture",
        "--model", f"profiles/{capture_name}_causal_rig_96k.npz",
        "--di", str(di_path),
        "--target", target_path,
        "--output-model", f"profiles/{capture_name}_causal_rig_96k_refined.npz",
        "--output", f"outputs/{capture_name}_refined_validation.wav",
        "--comparison-output", f"outputs/{capture_name}_target_base_refined_10s.wav",
        "--output-sample-rate", "96000",
        "--epochs", "30",
    ]
    print_command(args)
    if ask("Start guitar refinement? (yes/no)", "yes").lower() not in {"y", "yes"}:
        print("Refinement cancelled.")
        return
    main(args)


def build_cabinet_variant() -> None:
    reference_name = ask("Reference fixed-rig capture name", DEFAULT_CAPTURE)
    variant_name = ask("Variant probe capture name", f"{reference_name}_sm57_off_axis")
    profile_name = ask("Cabinet variant profile name", "egnater_sm57_off_axis")
    reference_position = ask("Reference mic position", "SM57 cap edge, close, on-axis")
    variant_position = ask("Variant mic position", "SM57 cone edge, 4 inches")
    variant_axis = ask("Variant mic axis (on-axis/off-axis)", "off-axis")
    variant_cabinet = ask("Variant cabinet", "Egnater Tweaker 1x12 Celestion")
    print("\nBoth returns must use the same probe, pedal, amp settings, reamp level, and mic-preamp gain.")
    print("Only the documented cabinet/microphone choice or placement may change.")
    if ask("Confirm both controlled captures meet that rule (yes/no)", "no").lower() not in {"y", "yes"}:
        print("Cabinet variant build cancelled.")
        return
    args = [
        "build-cabinet-variant",
        "--probe", f"rig_captures/{reference_name}_probe_input.wav",
        "--probe-manifest", "rig_captures/probes/rig_probe_96k_manifest.json",
        "--reference-target", f"rig_captures/{reference_name}_target_return.wav",
        "--variant-target", f"rig_captures/{variant_name}_target_return.wav",
        "--profile", f"profiles/cabinet_variants/{profile_name}.npz",
        "--comparison-output", f"outputs/{profile_name}_reference_recorded_synthesized.wav",
        "--name", profile_name,
        "--reference-cabinet", "Egnater Tweaker 1x12 Celestion",
        "--reference-microphone", "Shure SM57",
        "--reference-mic-position", reference_position,
        "--variant-cabinet", variant_cabinet,
        "--variant-microphone", "Shure SM57",
        "--variant-mic-position", variant_position,
        "--variant-mic-axis", variant_axis,
    ]
    print_command(args)
    if ask("Build and validate cabinet variant? (yes/no)", "yes").lower() not in {"y", "yes"}:
        print("Cabinet variant build cancelled.")
        return
    main(args)


def build_separated_cabinet() -> None:
    base_name = ask("Fixed pedal/amp name", DEFAULT_CAPTURE)
    preamp_name = ask("Amp-preamp capture name", f"{base_name}_preamp")
    amp_cab_name = ask("Matching amp-cab capture name", f"{base_name}_amp_cab")
    profile_name = ask("Separated cabinet profile name", f"{base_name}_cabinet_sm57")
    args = [
        "build-separated-cabinet",
        "--preamp-capture-manifest",
        f"rig_captures/{preamp_name}_rig_capture_manifest.json",
        "--amp-cab-capture-manifest",
        f"rig_captures/{amp_cab_name}_rig_capture_manifest.json",
        "--profile", f"profiles/cabinet_variants/{profile_name}.npz",
        "--comparison-output", f"outputs/{profile_name}_line_then_recorded_then_modeled.wav",
        "--name", profile_name,
        "--mic-axis", ask("SM57 axis (on-axis/off-axis)", "on-axis"),
    ]
    print("\nBoth captures must use the identical calibrated probe, reamp send trim, pedal,")
    print("amp, and knob settings. The amp-preamp return must be an approved line-level output.")
    print("Never connect an amplifier speaker output directly to an audio interface.")
    print_command(args)
    if ask("Build and validate the separated cabinet/SM57 stage? (yes/no)", "yes").lower() in {"y", "yes"}:
        main(args)


def apply_capture() -> None:
    capture_name = ask("Fixed-rig capture name", DEFAULT_CAPTURE)
    latest = newest_di()
    input_default = str(latest.relative_to(PROJECT_DIR)) if latest else "recordings/new_clean_di.wav"
    input_path = ask("Clean DI WAV", input_default)
    output_default = f"outputs/{Path(input_path).stem}_{capture_name}_render.wav"
    output_path = ask("Output WAV", output_default)
    refined_model = Path(f"profiles/{capture_name}_causal_rig_96k_refined.npz")
    base_model = Path(f"profiles/{capture_name}_causal_rig_96k.npz")
    model_path = refined_model if refined_model.exists() else base_model
    latest_variant = newest_cabinet_variant()
    variant_default = str(latest_variant.relative_to(PROJECT_DIR)) if latest_variant else "none"
    variant_path = ask("Cabinet/mic variant profile (none = captured reference)", variant_default)
    args = [
        "apply-rig-capture",
        "--input", input_path,
        "--model", str(model_path),
        "--output", output_path,
        "--output-sample-rate", "96000",
    ]
    if variant_path.lower() not in {"", "none", "off"}:
        cabinet_mix = ask("Cabinet variant mix 0-1", "1")
        low_cut = ask("Cabinet low cut Hz (0 = off)", "0")
        high_cut = ask("Cabinet high cut Hz (0 = off)", "0")
        args.extend(
            [
                "--cabinet-variant", variant_path,
                "--cabinet-mix", cabinet_mix,
                "--cabinet-low-cut-hz", low_cut,
                "--cabinet-high-cut-hz", high_cut,
            ]
        )
    use_virtual_studio = ask("Add dual-mic virtual studio controls? (yes/no)", "no").lower() in {"y", "yes"}
    if use_virtual_studio:
        mic_b_default = str(latest_variant.relative_to(PROJECT_DIR)) if latest_variant else "none"
        mic_b_path = ask("Second measured cabinet/mic profile (none = no mic B)", mic_b_default)
        mic_morph = ask("Mic A to B morph 0-1", "0.5")
        variphi = ask("Mic B phase timing in milliseconds (-10 to 10)", "0")
        pan_a = ask("Mic A pan (-1 left, 1 right)", "-0.15")
        pan_b = ask("Mic B pan (-1 left, 1 right)", "0.15")
        room = ask("Room preset (off/tight/studio/live)", "tight")
        distance = ask("Virtual mic distance 0-1", "0.25")
        room_mix = ask("Room contribution 0-1", "0.20")
        overload = ask("Additional speaker overload 0-1", "0")
        args.extend(
            [
                "--virtual-mic-morph", mic_morph,
                "--virtual-variphi-ms", variphi,
                "--virtual-mic-a-pan", pan_a,
                "--virtual-mic-b-pan", pan_b,
                "--virtual-room-preset", room,
                "--virtual-distance", distance,
                "--virtual-room-mix", room_mix,
                "--virtual-speaker-overload", overload,
            ]
        )
        if mic_b_path.lower() not in {"", "none", "off"}:
            args.extend(["--virtual-mic-b", mic_b_path])
    print_command(args)
    main(args)


def build_performance_rig() -> None:
    capture_name = ask("Fixed-rig capture name", DEFAULT_CAPTURE)
    refined_model = Path(f"profiles/{capture_name}_causal_rig_96k_refined.npz")
    base_model = Path(f"profiles/{capture_name}_causal_rig_96k.npz")
    model_default = refined_model if refined_model.exists() else base_model
    model_path = ask("Primary accepted model", str(model_default))
    secondary = ask("Second accepted model for morph (none = off)", "none")
    morph = ask("Primary to secondary model morph 0-1", "0")
    path_mode = ask("Model graph mode (morph/parallel)", "morph")
    input_impedance = ask("Interface guitar-input impedance kOhm", "1000")
    latest_variant = newest_cabinet_variant()
    variant_default = str(latest_variant.relative_to(PROJECT_DIR)) if latest_variant else "none"
    cabinet_ir = ask("Full cabinet IR for Amp / Pre-Amp capture (none = off)", "none")
    cabinet_variant = "none"
    cabinet_variant_b = "none"
    mic_position_morph = "0"
    if cabinet_ir.lower() in {"", "none", "off"}:
        cabinet_variant = ask("First measured mic endpoint (none = captured reference)", variant_default)
        if cabinet_variant.lower() not in {"", "none", "off"}:
            cabinet_variant_b = ask("Second measured mic endpoint (none = off)", "none")
            if cabinet_variant_b.lower() not in {"", "none", "off"}:
                mic_position_morph = ask("Measured mic position morph 0-1", "0")
    preset_name = ask("Performance rig name", capture_name)
    preset_path = ask(
        "Performance rig preset path",
        f"profiles/performance_rigs/{capture_name}.json",
    )
    gate = ask("Input gate threshold dBFS (-100 = off)", "-100")
    bass = ask("Clone bass adjustment dB", "0")
    middle = ask("Clone middle adjustment dB", "0")
    treble = ask("Clone treble adjustment dB", "0")
    low_cut = ask("Cabinet low cut Hz (0 = off)", "0")
    high_cut = ask("Cabinet high cut Hz (0 = off)", "0")
    speaker_curve = ask(
        "Speaker curve (flat/open-back-1x12/closed-back-1x12/closed-back-4x12/custom)",
        "flat",
    )
    speaker_drive = ask("Dynamic speaker drive 0-1", "0")
    cone_cry = ask("Cone cry 0-1", "0")
    destination = ask(
        "Output destination (studio-frfr/headphones/amp-input/amp-return/power-amp-guitar-cab)",
        "studio-frfr",
    )
    snapshots_json = ask("Named snapshots JSON (none = off)", "none")
    args = [
        "build-performance-rig",
        "--preset", preset_path,
        "--name", preset_name,
        "--model", model_path,
        "--model-morph", morph,
        "--model-path-mode", path_mode,
        "--input-impedance-kohm", input_impedance,
        "--gate-threshold-dbfs", gate,
        "--bass-db", bass,
        "--middle-db", middle,
        "--treble-db", treble,
        "--cabinet-low-cut-hz", low_cut,
        "--cabinet-high-cut-hz", high_cut,
        "--speaker-impedance-curve", speaker_curve,
        "--dynamic-speaker-drive", speaker_drive,
        "--cone-cry", cone_cry,
        "--destination", destination,
        "--normalize", "off",
        "--output-sample-rate", "96000",
    ]
    if secondary.lower() not in {"", "none", "off"}:
        args.extend(["--secondary-model", secondary])
    if cabinet_variant.lower() not in {"", "none", "off"}:
        args.extend(["--cabinet-variant", cabinet_variant])
    if cabinet_variant_b.lower() not in {"", "none", "off"}:
        args.extend(["--cabinet-variant-b", cabinet_variant_b, "--mic-position-morph", mic_position_morph])
    if cabinet_ir.lower() not in {"", "none", "off"}:
        args.extend(["--cabinet-ir", cabinet_ir])
    if snapshots_json.lower() not in {"", "none", "off"}:
        args.extend(["--snapshots-json", snapshots_json])
    print_command(args)
    main(args)


def apply_performance_rig() -> None:
    latest_preset = newest_performance_rig()
    preset_default = (
        str(latest_preset.relative_to(PROJECT_DIR))
        if latest_preset
        else f"profiles/performance_rigs/{DEFAULT_CAPTURE}.json"
    )
    preset_path = ask("Performance rig preset", preset_default)
    latest = newest_di()
    input_default = str(latest.relative_to(PROJECT_DIR)) if latest else "recordings/new_clean_di.wav"
    input_path = ask("Clean DI WAV", input_default)
    output_path = ask(
        "Output WAV",
        f"outputs/{Path(input_path).stem}_{Path(preset_path).stem}_performance_rig.wav",
    )
    snapshot = ask("Snapshot name (none = preset defaults)", "none")
    destination = ask("Destination override (none = preset default)", "none")
    source_impedance = ask("Current interface input impedance kOhm (none = unchecked)", "none")
    args = [
        "apply-performance-rig",
        "--input", input_path,
        "--preset", preset_path,
        "--output", output_path,
    ]
    if snapshot.lower() not in {"", "none", "off"}:
        args.extend(["--snapshot", snapshot])
    if destination.lower() not in {"", "none", "off"}:
        args.extend(["--destination", destination])
    if source_impedance.lower() not in {"", "none", "off"}:
        args.extend(["--source-input-impedance-kohm", source_impedance])
    print_command(args)
    main(args)


def run() -> None:
    print("\nControlled fixed-rig capture")
    print("  1. Generate 96 kHz probe")
    print("  2. Check routing and command (no playback)")
    print("  3. Record amp/cab/mic response")
    print("  4. Train and validate causal rig model")
    print("  5. Refine with real guitar dynamics")
    print("  6. Build a measured cabinet/mic variant")
    print("  7. Apply accepted model to a clean DI")
    print("  8. Build a portable performance rig")
    print("  9. Apply a portable performance rig")
    print("  10. Build a separated cabinet/SM57 stage from matched captures")
    selection = ask("Choose", "1")
    actions = {
        "1": generate_probe,
        "2": lambda: record_probe(dry_run=True),
        "3": lambda: record_probe(dry_run=False),
        "4": train_capture,
        "5": refine_capture,
        "6": build_cabinet_variant,
        "7": apply_capture,
        "8": build_performance_rig,
        "9": apply_performance_rig,
        "10": build_separated_cabinet,
    }
    action = actions.get(selection)
    if action is None:
        raise SystemExit("Choose a number from 1 to 10.")
    action()


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    run()
