#!/usr/bin/env python3
"""Guided PyCharm runner for isolated PyTorch/NAM tone-model research."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from tone_capture_engine import main


PROJECT_DIR = Path(__file__).resolve().parent


def ask(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def newest_pair() -> tuple[Path | None, Path | None, Path | None]:
    takes = sorted((PROJECT_DIR / "recordings").glob("*_clean_di.wav"), key=lambda path: path.stat().st_mtime)
    for di in reversed(takes):
        target = di.with_name(di.name.replace("_clean_di.wav", "_amp_mic_target.wav"))
        manifest = di.with_name(di.name.replace("_clean_di.wav", "_hardware_manifest.json"))
        if target.exists():
            return di, target, manifest if manifest.exists() else None
    return None, None, None


def relative(path: Path | None, fallback: str) -> str:
    if path is None:
        return fallback
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def show_command(args: list[str]) -> None:
    print("\n" + shlex.join([".venv/bin/python", "tone_capture_engine.py", *args]) + "\n")


def check_stack() -> None:
    args = ["research-stack-check"]
    show_command(args)
    main(args)


def prepare_capture() -> None:
    di, target, manifest = newest_pair()
    di_value = ask("Clean DI WAV", relative(di, "recordings/take_clean_di.wav"))
    target_value = ask("Matching amp/mic WAV", relative(target, "recordings/take_amp_mic_target.wav"))
    manifest_value = ask("Hardware manifest (blank if unavailable)", relative(manifest, ""))
    name = ask("Research capture name", Path(di_value).name.replace("_clean_di.wav", ""))
    args = [
        "prepare-research-capture",
        "--di", di_value,
        "--target", target_value,
        "--name", name,
        "--output-dir", "research_captures",
        "--validation-fraction", "0.15",
    ]
    if manifest_value:
        args.extend(["--manifest", manifest_value])
    show_command(args)
    main(args)


def train_reference(architecture: str) -> None:
    captures = sorted((PROJECT_DIR / "research_captures").glob("*/research_capture.json"))
    latest = max(captures, key=lambda path: path.stat().st_mtime) if captures else None
    capture = ask("Prepared research_capture.json", relative(latest, "research_captures/name/research_capture.json"))
    name = Path(capture).parent.name
    args = [
        "train-torch-reference",
        "--capture-manifest", capture,
        "--architecture", architecture,
        "--model", f"profiles/research/{name}_{architecture}.pt",
        "--output", f"outputs/research/{name}_{architecture}_validation.wav",
        "--metrics-output", f"outputs/research/{name}_{architecture}_metrics.json",
        "--epochs", ask("Epochs", "30"),
        "--steps-per-epoch", ask("Chunks per epoch", "64"),
    ]
    show_command(args)
    if ask("Start isolated reference training? (yes/no)", "yes").lower() in {"y", "yes"}:
        main(args)


def build_dataset() -> None:
    args = [
        "build-conditioned-dataset",
        "--dataset", ask("Recording dataset", "datasets/6505_rhythm_sm57_all_guitars.json"),
        "--output", ask("Output manifest", "research_datasets/all_recordings_conditioned.json"),
    ]
    if ask("Allow explicitly labeled multiple rig groups? (yes/no)", "no").lower() in {"y", "yes"}:
        args.append("--allow-mixed-rigs")
    show_command(args)
    main(args)


def newest_conditioned_dataset() -> Path | None:
    manifests = []
    for path in (PROJECT_DIR / "research_datasets").glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("format") == "tone_capture_conditioned_dataset_1.0":
            manifests.append(path)
    return max(manifests, key=lambda path: path.stat().st_mtime) if manifests else None


def default_holdout(dataset_path: str) -> str:
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "TAKE_NAME"
    entries = list(payload.get("entries", []))
    return str(entries[-1].get("take_name", "TAKE_NAME")) if entries else "TAKE_NAME"


def train_conditioned_reference(architecture: str) -> None:
    latest = newest_conditioned_dataset()
    dataset = ask(
        "Conditioned dataset manifest",
        relative(latest, "research_datasets/all_recordings_conditioned.json"),
    )
    holdout = ask("Whole recording to hold out", default_holdout(dataset))
    name = ask("Candidate name", f"all_recordings_{holdout}_{architecture}_96k")
    args = [
        "train-conditioned-torch-reference",
        "--dataset-manifest", dataset,
        "--holdout-take", holdout,
        "--architecture", architecture,
        "--sample-rate", "96000",
        "--model", f"profiles/research/{name}.pt",
        "--output", f"outputs/research/{name}.wav",
        "--metrics-output", f"outputs/research/{name}_metrics.json",
        "--epochs", ask("Epochs", "30"),
        "--steps-per-epoch", ask("Chunks per epoch", "64"),
        "--chunk-samples", "8192",
        "--focus-rig-fraction", ask("Exact holdout-rig training share 0-1", "0.75"),
    ]
    show_command(args)
    if ask("Start whole-take guarded training? (yes/no)", "yes").lower() in {"y", "yes"}:
        main(args)


def freeze_conditioned_dataset() -> None:
    latest = newest_conditioned_dataset()
    dataset = ask(
        "Approved conditioned dataset manifest",
        relative(latest, "research_datasets/all_recordings_conditioned.json"),
    )
    output = ask("Frozen hash manifest", "research_datasets/frozen_active_24.json")
    args = [
        "freeze-conditioned-dataset",
        "--dataset-manifest", dataset,
        "--output", output,
    ]
    show_command(args)
    main(args)


def train_accuracy_lane() -> None:
    latest = newest_conditioned_dataset()
    dataset = ask(
        "Conditioned dataset manifest",
        relative(latest, "research_datasets/all_recordings_conditioned.json"),
    )
    frozen = ask("Frozen hash manifest", "research_datasets/frozen_active_24.json")
    holdout = ask("Whole recording to hold out", default_holdout(dataset))
    args = [
        "train-amp-accuracy-lane",
        "--dataset-manifest", dataset,
        "--frozen-manifest", frozen,
        "--holdout-take", holdout,
    ]
    show_command(args)
    print(
        "This 96 kHz lane uses every approved recording, favors the holdout's exact rig, "
        "and promotes only a held-out listening improvement."
    )
    if ask("Start the long guarded accuracy run? (yes/no)", "yes").lower() in {"y", "yes"}:
        main(args)


def prepare_nam_a2() -> None:
    latest = newest_conditioned_dataset()
    dataset = ask(
        "Conditioned dataset manifest",
        relative(latest, "research_datasets/all_recordings_conditioned.json"),
    )
    holdout = ask("Whole recording to hold out", default_holdout(dataset))
    name = ask("NAM A2 capture name", f"{holdout}_holdout")
    output_dir = ask(
        "Output folder on the capped research image",
        f"/Volumes/ToneCaptureResearch/captures/{name}",
    )
    args = [
        "prepare-conditioned-nam-a2",
        "--dataset-manifest", dataset,
        "--holdout-take", holdout,
        "--name", name,
        "--output-dir", output_dir,
        "--sample-rate", "96000",
        "--nam-epochs", ask("NAM epochs", "30"),
        "--nam-batch-size", "8",
        "--nam-window-samples", "4096",
        "--nam-train-batches-per-epoch", ask("Training batches per epoch (0 = all)", "24"),
        "--nam-validation-batches-per-epoch", ask("Validation batches per epoch (0 = all)", "8"),
    ]
    if ask("Start NAM A2 training after preparation? (yes/no)", "no").lower() in {"y", "yes"}:
        args.append("--start-training")
    show_command(args)
    main(args)


def apply_nam() -> None:
    input_path = ask("Clean DI or held-out validation input WAV")
    model_path = ask("Exported NAM model.nam")
    output_path = ask("NAM render WAV", "outputs/research/nam_a2_validation.wav")
    args = [
        "apply-nam-reference",
        "--input", input_path,
        "--model", model_path,
        "--output", output_path,
    ]
    show_command(args)
    main(args)


def compare_models() -> None:
    di = ask("Held-out DI WAV")
    target = ask("Matching real amp/mic WAV")
    output = ask("Metrics JSON", "outputs/research/hybrid_model_comparison.json")
    candidates = []
    while True:
        value = ask("Candidate NAME=render.wav (blank when done)")
        if not value:
            break
        candidates.extend(["--candidate", value])
    if not candidates:
        print("No candidates entered.")
        return
    args = ["hybrid-model-compare", "--di", di, "--target", target, *candidates, "--output", output]
    show_command(args)
    main(args)


def main_menu() -> None:
    os.chdir(PROJECT_DIR)
    print("Isolated Tone Model Research")
    print("1. Check capped research stack")
    print("2. Prepare aligned DI/amp capture")
    print("3. Train PyTorch causal TCN reference")
    print("4. Train PyTorch GRU reference")
    print("5. Build all-recordings conditioned index")
    print("6. Train all-recordings conditioned TCN with whole-take guard")
    print("7. Train all-recordings conditioned GRU with whole-take guard")
    print("8. Prepare or train exact-rig NAM A2 with whole-take guard")
    print("9. Render an exported NAM A2 model")
    print("10. Compare MLX, PyTorch, and NAM renders")
    print("11. Train legacy PyTorch LSTM reference")
    print("12. Freeze the approved conditioned dataset (hashes only, no copied audio)")
    print("13. Train the 96 kHz long-memory amp accuracy lane")
    choice = ask("Choice", "1")
    actions = {
        "1": check_stack,
        "2": prepare_capture,
        "3": lambda: train_reference("tcn"),
        "4": lambda: train_reference("gru"),
        "5": build_dataset,
        "6": lambda: train_conditioned_reference("tcn"),
        "7": lambda: train_conditioned_reference("gru"),
        "8": prepare_nam_a2,
        "9": apply_nam,
        "10": compare_models,
        "11": lambda: train_reference("lstm"),
        "12": freeze_conditioned_dataset,
        "13": train_accuracy_lane,
    }
    action = actions.get(choice)
    if action is None:
        raise SystemExit("Unknown choice.")
    action()


if __name__ == "__main__":
    main_menu()
