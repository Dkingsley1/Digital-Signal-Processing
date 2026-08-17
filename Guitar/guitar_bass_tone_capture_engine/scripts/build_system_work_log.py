#!/usr/bin/env python3
"""Build a PDF and mobile-readable work log for the tone capture system."""

from __future__ import annotations

import argparse
import base64
import html
import importlib.metadata
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A5, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_BASENAME = "tone_capture_system_work_log"

PRIMARY_STACK = [
    ("numpy", "core arrays and numeric data movement"),
    ("scipy", "DSP, filters, convolution, WAV fallback, correlation"),
    ("sounddevice", "audio interface capture and live input stream"),
    ("matplotlib", "legacy live graph path"),
    ("pyqtgraph", "fast live frequency/waveform display"),
    ("PySide6", "Qt runtime for live scope UI"),
    ("mlx", "Apple Silicon neural amp/residual training"),
    ("mlx-metal", "Apple GPU acceleration for MLX"),
    ("soundfile", "float WAV read/write path"),
    ("soxr", "very-high-quality resampling"),
    ("librosa", "advanced spectral descriptors for quality gates"),
    ("pyloudnorm", "integrated LUFS checks for DI/mic balance"),
    ("noisereduce", "explicit denoise-preview command"),
    ("pedalboard", "explicit effect-chain and future plugin preview path"),
    ("scikit-learn", "guarded live pickup/blower reference classifier"),
    ("reportlab", "PDF work-log generation"),
    ("pdfplumber", "PDF text/layout inspection"),
    ("pypdf", "PDF structural validation"),
]

ISOLATED_RESEARCH_STACK = [
    ("torch", "2.13.0 (isolated)", "MPS causal conditioned TCN/GRU/LSTM research and guarded DI rendering"),
    ("torchaudio", "2.11.0 (isolated)", "PyTorch audio compatibility layer"),
    ("auraloss", "0.4.0 (isolated)", "linear and mel multi-resolution spectral training and ranking"),
    ("neural-amp-modeler", "0.13.0 (isolated)", "independent exact-rig NAM A2/PackedWaveNet training and export"),
    ("nablafx", "1.0.0 (isolated)", "architecture, conditioning, and gray-box research recipes"),
]

SOURCE_LINKS = [
    (
        "Pedalboard",
        "https://spotify.github.io/pedalboard/",
        "Python audio effects, VST3, Audio Unit, and offline effect-chain processing.",
    ),
    (
        "pyloudnorm",
        "https://github.com/csteinmetz1/pyloudnorm",
        "ITU-R BS.1770-4 integrated loudness measurement.",
    ),
    (
        "librosa spectral flatness",
        "https://librosa.org/doc/main/generated/librosa.feature.spectral_flatness.html",
        "Noise-like versus tone-like spectral descriptor.",
    ),
    (
        "python-soxr",
        "https://python-soxr.readthedocs.io/en/stable/soxr.html",
        "High-quality resampling API.",
    ),
    (
        "noisereduce",
        "https://github.com/timsainb/noisereduce",
        "Spectral-gating noise reduction for explicit preview renders.",
    ),
    (
        "scikit-learn",
        "https://scikit-learn.org/stable/modules/multiclass.html",
        "Classical classifiers used as a guarded live reference route.",
    ),
    (
        "ReportLab",
        "https://www.reportlab.com/dev/docs/",
        "PDF generation used for this system work log.",
    ),
    (
        "Neural Amp Modeler",
        "https://neural-amp-modeler.readthedocs.io/en/stable/tutorials/full.html",
        "Candidate separate research benchmark for amp modeling.",
    ),
    (
        "auraloss",
        "https://github.com/csteinmetz1/auraloss",
        "Multi-resolution spectral and perceptual audio losses used by the isolated benchmark.",
    ),
    (
        "NablAFx",
        "https://github.com/mcomunita/nablafx",
        "PyTorch black-box and gray-box audio-effect architectures and conditioning reference.",
    ),
    (
        "GuitarML Proteus",
        "https://guitarml.com/index.html",
        "Reference for parameter-conditioned captures across gain, EQ, and device controls.",
    ),
    (
        "AIDA-X",
        "https://aidadsp.github.io/",
        "Reference for compact recurrent neural amp models and interoperable model playback.",
    ),
    (
        "TONEX",
        "https://www.ikmultimedia.com/products/tonex/",
        "Reference for prepared excitation, pedal/amp/cab capture, and real-guitar validation.",
    ),
    (
        "Two notes CODEX",
        "https://www.two-notes.com/en/discover-codex/",
        "Reference for NAM/AIDA-X/Proteus interoperability and modular cabinet/FX routing.",
    ),
    (
        "TorchCodec",
        "https://meta-pytorch.org/torchcodec/",
        "Candidate PyTorch media decoding path if a torch trainer is added.",
    ),
    (
        "ONNX Runtime",
        "https://onnxruntime.ai/docs/api/python/api_summary.html",
        "Candidate deployment/inference runtime if model export is added.",
    ),
    (
        "Quad Cortex capture manual",
        "https://neuraldsp.com/manual/quad-cortex",
        "Reference workflow for controlled test signals, latency/level capture, training, and A/B validation.",
    ),
    (
        "Kemper profiling technology",
        "https://www.kemper-amps.com/profiling",
        "Reference workflow for musical refinement, calibrated response, detailed frequency analysis, and cabinet resonance.",
    ),
    (
        "Fender Tone Master Pro",
        "https://www.fender.com/products/tone-master-pro",
        "Reference architecture for separate amp models and selectable cabinet/microphone impulse responses.",
    ),
    (
        "Fender Tone Master Pro updates",
        "https://support.fender.com/hc/en-gb/articles/46768129660315-Fender-Tone-Master-Pro-Firmware-and-Pro-Control-App-Updates",
        "Reference for speaker impedance curves, external cabinets, cabinet filters, and model updates.",
    ),
    (
        "Two notes Wall of Sound manual",
        "https://wiki.two-notes.com/doku.php?id=torpedo_wall_of_sound%3Atorpedo_wall_of_sound_user_s_manual",
        "Reference for virtual mic center/distance, dual-channel phase interaction, speaker overload, room, and explicit power-amp controls.",
    ),
    (
        "HeadRush Prime and Amp Cloner",
        "https://www.headrushfx.com/products/prime/",
        "Reference for guided clone categories, target/clone audition, portable rigs, cabinet bypass and IR blocks, clone tone controls, and multi-clone performance workflows.",
    ),
    (
        "Universal Audio OX",
        "https://www.uaudio.com/products/ox-amp-top-box",
        "Reference for level-dependent speaker breakup, cone cry, close/room microphones, and behavior beyond a static cabinet IR.",
    ),
    (
        "Fractal Audio Blocks Guide",
        "https://www.fractalaudio.com/downloads/manuals/fas-guides/Fractal-Audio-Blocks-Guide.pdf",
        "Reference for cabinet-linked speaker impedance curves and continuously adjustable measured cabinet/microphone positions.",
    ),
    (
        "BOSS GT-1000 AIRD",
        "https://www.boss.info/us/products/gt-1000/support/",
        "Reference for destination-aware output routing to recording, full-range, amp-return, power-amp, and guitar-cabinet systems.",
    ),
    (
        "Line 6 Helix",
        "https://line6.com/support/manuals/helix",
        "Reference for snapshots, parallel paths, explicit input-impedance handling, and portable signal-graph workflows.",
    ),
]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def file_count(paths: list[Path]) -> str:
    return str(len(paths))


def mb(path: Path) -> str:
    try:
        return f"{path.stat().st_size / (1024 * 1024):.1f} MB"
    except OSError:
        return "unknown"


def rel(project_dir: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_dir.resolve()))
    except ValueError:
        return str(path)


def collect_commands(project_dir: Path) -> list[tuple[str, str]]:
    try:
        import __main__

        if hasattr(__main__, "build_parser"):
            parser = __main__.build_parser()
        else:
            import tone_capture_engine

            parser = tone_capture_engine.build_parser()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return sorted((item.dest, item.help or "") for item in action._choices_actions)
    except Exception:
        pass

    return [
        ("system-on", "Prepare folders and launch the live scope."),
        ("record", "Capture paired clean DI and amp/mic WAV files."),
        ("train-all-recordings-amp", "Train the all-recordings amp-dominant MLX model."),
    ]


def collect_manifest_summary(project_dir: Path) -> dict[str, Any]:
    manifests = sorted((project_dir / "recordings").glob("*_hardware_manifest.json"))
    active = []
    created_dates = []
    guitars = Counter()
    pickups = Counter()
    boost = Counter()
    sample_rates = Counter()
    amps = Counter()
    cabinets = Counter()
    usable = 0
    preferred = 0

    for path in manifests:
        data = safe_json(path)
        if not isinstance(data, dict):
            continue
        take_name = str(data.get("take_name", path.stem))
        if take_name.startswith("level_test"):
            continue
        active.append(path)
        created = str(data.get("created_at", ""))
        if created:
            created_dates.append(created)
        metadata = dict(data.get("take_metadata", {}))
        interface = dict(data.get("audio_interface", {}))
        box = dict(data.get("di_box", {}))
        levels = dict(data.get("recording_levels", {}))
        guitars[str(metadata.get("guitar") or "unlabeled")] += 1
        pickups[str(metadata.get("pickup") or "unlabeled")] += 1
        boost[str(metadata.get("boost_pedal") or "none/unlabeled")] += 1
        sample_rates[str(interface.get("sample_rate_hz") or "unknown")] += 1
        amps[str(box.get("amp_name") or "unlabeled")] += 1
        cabinets[str(box.get("cabinet_name") or "unlabeled")] += 1
        usable += int(bool(levels.get("usable_for_training", False)))
        preferred += int(bool(levels.get("preferred_for_training", False)))

    return {
        "manifest_count": len(manifests),
        "active_take_count": len(active),
        "created_first": min(created_dates) if created_dates else "unknown",
        "created_last": max(created_dates) if created_dates else "unknown",
        "guitars": guitars,
        "pickups": pickups,
        "boost": boost,
        "sample_rates": sample_rates,
        "amps": amps,
        "cabinets": cabinets,
        "usable": usable,
        "preferred": preferred,
    }


def collect_dataset_summary(project_dir: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((project_dir / "datasets").glob("*.json")):
        data = safe_json(path)
        takes = list(data.get("takes", [])) if isinstance(data, dict) else []
        summaries.append(
            {
                "path": rel(project_dir, path),
                "takes": len(takes),
                "usable": sum(1 for item in takes if bool(item.get("usable_for_training", False))),
                "preferred": sum(1 for item in takes if bool(item.get("preferred_for_training", False))),
                "updated": str(data.get("updated_at", data.get("created_at", "unknown"))) if isinstance(data, dict) else "unknown",
            }
        )
    return summaries


def collect_model_summary(project_dir: Path) -> list[dict[str, str]]:
    models = sorted((project_dir / "profiles").glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "path": rel(project_dir, path),
            "size": mb(path),
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        for path in models[:12]
    ]


def collect_inventory(project_dir: Path) -> dict[str, Any]:
    recordings = project_dir / "recordings"
    outputs = project_dir / "outputs"
    profiles = project_dir / "profiles"
    logs = project_dir / "logs"

    clean_di = sorted(recordings.glob("*_clean_di.wav"))
    targets = sorted(recordings.glob("*_amp_mic_target.wav"))
    paired = [path for path in clean_di if path.with_name(path.name.replace("_clean_di.wav", "_amp_mic_target.wav")).exists()]
    non_level_pairs = [path for path in paired if not path.name.startswith("level_test")]

    return {
        "clean_di": clean_di,
        "targets": targets,
        "paired": paired,
        "non_level_pairs": non_level_pairs,
        "manifests": sorted(recordings.glob("*_hardware_manifest.json")),
        "profile_json": sorted(profiles.glob("*.json")),
        "profile_npz": sorted(profiles.glob("*.npz")),
        "output_wav": sorted(outputs.glob("**/*.wav")),
        "output_json": sorted(outputs.glob("**/*.json")),
        "datasets": sorted((project_dir / "datasets").glob("*.json")),
        "feature_logs": sorted(logs.glob("**/*.jsonl")) if logs.exists() else [],
        "commands": collect_commands(project_dir),
        "manifest_summary": collect_manifest_summary(project_dir),
        "dataset_summary": collect_dataset_summary(project_dir),
        "model_summary": collect_model_summary(project_dir),
    }


def counter_items(counter: Counter, limit: int = 8) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{name}: {count}" for name, count in counter.most_common(limit))


def build_sections(project_dir: Path) -> list[dict[str, Any]]:
    inventory = collect_inventory(project_dir)
    manifest = inventory["manifest_summary"]
    generated = datetime.now().isoformat(timespec="seconds")

    command_rows = [[name, help_text] for name, help_text in inventory["commands"]]
    dependency_rows = [[name, package_version(name), route] for name, route in PRIMARY_STACK]
    dependency_rows.extend([list(item) for item in ISOLATED_RESEARCH_STACK])
    dataset_rows = [
        [item["path"], str(item["takes"]), str(item["usable"]), str(item["preferred"]), item["updated"]]
        for item in inventory["dataset_summary"]
    ] or [["none", "0", "0", "0", ""]]
    model_rows = [[item["path"], item["size"], item["modified"]] for item in inventory["model_summary"]] or [
        ["none", "", ""]
    ]

    sections: list[dict[str, Any]] = [
        {
            "title": "Source And Scope",
            "blocks": [
                {
                    "type": "p",
                    "text": (
                        f"Generated {generated} from the local project at {project_dir}. "
                        "This project folder is not a git repository, so this is a reconstructed inception log "
                        "based on the current files, manifests, outputs, dependency state, and known session work."
                    ),
                },
                {
                    "type": "ul",
                    "items": [
                        "Primary system: guitar and bass DI-to-amp/mic tone capture engine.",
                        "Current hardware capture path: clean DI on channel 1, amp/mic target on channel 2.",
                        "Main modeling direction: quality-gated all-recordings MLX amp model that applies amp/cab/SM57 behavior to DI.",
                        "Report formats generated: PDF, HTML, mobile HTML, Markdown, and plain text.",
                    ],
                },
            ],
        },
        {
            "title": "Work Done Since Inception",
            "blocks": [
                {
                    "type": "ul",
                    "items": [
                        "Built the core DSP prototype: WAV import/export, DI/target alignment, profile capture, tone-match rendering, nonlinear drive, sag, compression, and cabinet/tone impulse behavior.",
                        "Added hardware capture support: audio devices, two-channel interface capture, level checks, DI box/mic/amp/cab manifests, dataset manifests, and safe routing notes.",
                        "Added PyCharm startup path through system_on.py and the system-on command, including folder preparation and live scope launch.",
                        "Added a dedicated 08 Tone System - Performance Rig PyCharm Run configuration and performance_rig.py terminal launcher so portable rig build/apply actions no longer need to be found inside the larger capture wizard.",
                        "Expanded the live monitor: waveform lanes, frequency spectrum, tone-difference view, level meters, clipping checks, transient metrics, noise floor, rolloff, centroid, and machine-readable feature logs.",
                        "Iterated on pickup/blower visibility: pickup-sensitive spectrum, slower eye-friendly smoothing, activity gating, output deltas, held switch events, and reference-library matching from labeled recordings.",
                        "Added all-recordings training: automatic paired-take discovery, balanced sampling, source conditioning, amp-dominant tone anchors, per-take validation renders, and quality-gate exclusion/downweighting.",
                        "Hardened all-recordings modeling with canonical fixed-rig fingerprints and one-hot rig conditioning, level-preserving pickup/output training, fractional latency and polarity alignment, 20 ms waveform context, robust losses, gradient clipping, learning-rate recovery, and unstable-step rejection.",
                        "Added production candidate promotion: every selected take reserves an unseen tail, candidate and existing models are scored on the same audio, DI-gain-only behavior fails the amp-tone guard, pair regressions block promotion, and rejected candidates cannot overwrite the current model.",
                        "Connected the optional hybrid cabinet stage so an accepted measured cabinet/microphone response can follow the nonlinear MLX model during both validation and application without substituting a generic response when no measurement exists.",
                        "Added MLX model paths: residual layer, mic bridge, full-spectrum bridge, direct neural amp model, 96 kHz rendering, Apple Silicon performance launcher, and model application commands.",
                        "Added amp-tone regression protection to catch the failure mode where the model only follows DI gain instead of moving toward amp/mic spectral behavior.",
                        "Added controlled fixed-rig capture: a 96 kHz multilevel reamp probe, automatic latency/polarity measurement, preserved I/O gain relationship, causal long-memory MLX modeling, 2x anti-alias oversampling, and held-out production-model rejection.",
                        "Added real-guitar refinement inspired by musical profiler workflows: automatic DI-to-reamp level calibration, hard-chord/transient fine-tuning, protected best checkpoints, reference/base/refined A/B files, and rejection of held-out regressions.",
                        "Added 262,144-point response analysis with 131,073 frequency bins, measured dynamic-compression slope, cabinet bass-resonance detection, and high-frequency rolloff diagnostics.",
                        "Added a Fender Tone Master-inspired modular cabinet/microphone path: repeated-probe response extraction, minimum-phase correction FIRs, mic/cabinet/position/axis labels, selectable mix and cut controls, and held-out rejection against the original cabinet response.",
                        "Added a Two notes Wall of Sound-inspired virtual studio: dual accepted measured-mic rendering, equal-power endpoint morphing, independent mic level/pan, sub-sample phase timing and polarity, distance-scaled early reflections, optional speaker overload, and an explicit guard against stacking another power amp over the captured Peavey rig.",
                        "Added a HeadRush-inspired performance workflow: explicit Amp & Cab, Amp / Pre-Amp, and Pedal Only capture types; documented physical clone controls; portable JSON rig presets; accepted two-model constant-level morphing; input gate and tone controls; 1024/2048-sample cabinet IR blocks; normalization choice; and refusal to stack a full cabinet IR after an Amp & Cab capture.",
                        "Extended portable rigs with four guarded public-workflow influences: OX-style oversampled level-dependent speaker drive and cone-cry controls; Fractal-style cabinet-linked resonance approximations and two-endpoint measured mic morphing; AIRD-style destination validation and cabinet bypass; and Helix-style accepted-model parallel paths, named snapshots, and physical input-impedance mismatch checks.",
                        "Added an isolated hybrid research lane: PyTorch MPS TCN/GRU/LSTM candidates, linear and mel multi-resolution spectral losses, NAM 0.13 A2/PackedWaveNet exports, NablAFx architecture/conditioning recipes, and file-based comparison with production MLX renders.",
                        "Added strict research regression controls: target-aligned metrics, level-invariant DI-gain negative control, accepted/rejected model state, guarded application to new DI, and automated real-target-versus-DI smoke coverage.",
                        "Added all-recordings research indexing: automatic discovery of complete pairs, exact rig fingerprints, guitar/pickup/tuning/volume/tone conditions, matching-rig-focused sampling, and refusal to silently merge incompatible targets.",
                        "Added whole-recording holdouts and distributed full-take scoring so no validation recording can leak into optimization and later riffs influence candidate acceptance.",
                        "Added a calibrated multilevel 96 kHz probe with latency impulses, sweeps, level-stepped multisines, dynamic guitar-band noise, transients, and unseen validation sections.",
                        "Added a guarded separated amp/cab route: matched amp-preamp line-return and amp-cab/SM57 captures must share the probe, sample rate, reamp trim, pedal, amp, and controls before cabinet extraction is allowed.",
                        "Completed the first exact-rig NAM A2 whole-take benchmark: the NAM candidate passed the amp-tone guard at 12.56 dB spectral error and 0.676 correlation, while the research TCN and GRU candidates were quarantined as rejected.",
                        "Added a high-accuracy research lane: the 24 approved pairs are hash-locked without copying audio; a 96 kHz stacked gated TCN provides about 341 ms of causal memory; training uses guitar-band, multiresolution spectral, transient, crest, and multiscale log-RMS fullness losses.",
                        "Added promotion by audible held-out behavior instead of training loss: internal training tails select checkpoints, a complete recording remains untouched, and dry A/B checks cover quiet, medium, loud, transient, and sustained playing with one global level correction.",
                        "Ran the first long-memory take-031 candidate and correctly rejected it at 19.94 dB spectral error, -0.326 correlation, and a zero percent listening-section pass rate; the accepted NAM A2 result remains the stronger research reference.",
                        "Archived nine lower-quality recording sets instead of deleting them, leaving 24 active complete pairs that all pass the current preferred quality gate; future external archives now refuse projected use above the 5 GiB working cap.",
                        "Made preferred-only maintenance refresh every active pair from its current WAVs and saved level profile before selection, preventing old quality flags from archiving a newly approved take.",
                        "Added space and project isolation: an APFS research image mounted at /Volumes/ToneCaptureResearch, a 5 GiB working cap, cache routing inside that image, and hard rejection of Schwab project paths.",
                        "Added advanced audio stack routing: float WAV I/O, VHQ resampling, loudness checks, spectral descriptors, denoise previews, pedalboard previews, and now scikit-learn live reference classification.",
                        "Added reporting infrastructure in this pass: reportlab PDF generation, PDF inspection dependencies, and mobile-friendly fallback formats.",
                    ],
                }
            ],
        },
        {
            "title": "Current Project Inventory",
            "blocks": [
                {
                    "type": "table",
                    "headers": ["Item", "Count"],
                    "rows": [
                        ["Clean DI WAV files", file_count(inventory["clean_di"])],
                        ["Amp/mic target WAV files", file_count(inventory["targets"])],
                        ["Paired DI/amp recordings", file_count(inventory["paired"])],
                        ["Non-level-test paired recordings", file_count(inventory["non_level_pairs"])],
                        ["Hardware manifests", file_count(inventory["manifests"])],
                        ["Dataset manifests", file_count(inventory["datasets"])],
                        ["JSON tone profiles", file_count(inventory["profile_json"])],
                        ["MLX/NPZ model files", file_count(inventory["profile_npz"])],
                        ["Output WAV renders", file_count(inventory["output_wav"])],
                        ["Output JSON reports", file_count(inventory["output_json"])],
                        ["Live feature logs", file_count(inventory["feature_logs"])],
                        ["CLI commands", str(len(inventory["commands"]))],
                    ],
                }
            ],
        },
        {
            "title": "Recording Dataset",
            "blocks": [
                {
                    "type": "ul",
                    "items": [
                        f"Active non-level-test takes: {manifest['active_take_count']}",
                        f"Created range from manifests: {manifest['created_first']} to {manifest['created_last']}",
                        f"Manifests with saved recording-time usable flag: {manifest['usable']}",
                        f"Manifests with saved recording-time preferred flag: {manifest['preferred']}",
                        f"Guitars: {counter_items(manifest['guitars'])}",
                        f"Pickups: {counter_items(manifest['pickups'])}",
                        f"Boost states: {counter_items(manifest['boost'])}",
                        f"Sample rates: {counter_items(manifest['sample_rates'])}",
                        f"Amplifiers: {counter_items(manifest['amps'])}",
                        f"Cabinets: {counter_items(manifest['cabinets'])}",
                    ],
                },
                {"type": "table", "headers": ["Dataset", "Takes", "Usable", "Preferred", "Updated"], "rows": dataset_rows},
            ],
        },
        {
            "title": "Current Model Outputs",
            "blocks": [
                {"type": "table", "headers": ["Recent model", "Size", "Modified"], "rows": model_rows},
                {
                    "type": "p",
                    "text": (
                        "The most important current production-style models are the all-recordings amp-dominant NPZ files "
                        "under profiles/. The comparison renders under outputs/ and per-take validation folders are the "
                        "right place to listen for whether the DI has taken on amp/cab/SM57 behavior."
                    ),
                },
                {
                    "type": "p",
                    "text": (
                        "The current accepted research leader is the exact-rig NAM A2 benchmark at 12.56 dB spectral "
                        "error and 0.676 correlation. The newer long-memory PyTorch candidate is stored only as a rejected "
                        "artifact and did not replace any accepted or production model."
                    ),
                },
            ],
        },
        {
            "title": "Command Surface",
            "blocks": [
                {"type": "table", "headers": ["Command", "Purpose"], "rows": command_rows},
            ],
        },
        {
            "title": "Installed And Routed Libraries",
            "blocks": [
                {"type": "table", "headers": ["Library", "Version", "Route"], "rows": dependency_rows},
                {
                    "type": "p",
                    "text": (
                        "Training audio is intentionally not denoised or passed through pedalboard effects implicitly. "
                        "Those libraries are routed as explicit preview/audition paths so they cannot teach the model a fake target."
                    ),
                },
            ],
        },
        {
            "title": "Library Assessment",
            "blocks": [
                {
                    "type": "ul",
                    "items": [
                        "Added now: scikit-learn is routed into a guarded live pickup/blower classifier when enough labeled references exist.",
                        "Added now: reportlab, pdfplumber, and pypdf are routed into repeatable reporting and PDF validation.",
                        "Added separately: PyTorch, torchaudio, auraloss, NAM, and NablAFx are installed in the capped ToneCaptureResearch image, never in the production MLX interpreter.",
                        "Routed now: direct PyTorch TCN/GRU/LSTM uses MPS; NAM uses bounded CPU batches because its pinned Lightning MPS detector rejects the newer Torch build.",
                        "Routed now: NablAFx contributes explicit architecture, parameter-conditioning, and gray-box recipes; its optional Frechet/CLAP import is disabled because it eagerly requests external language-model data.",
                        "Routed now: MLX and PyTorch cooperate through aligned WAVs, manifests, candidate renders, and common guarded metrics instead of copying tensors between frameworks.",
                        "Possible future add: ONNX Runtime only makes sense after an export path exists.",
                        "Possible future add: Essentia may add deeper audio descriptors, but should be tested separately for macOS and Python 3.14 compatibility before routing into the live system.",
                        "Possible future route: pedalboard AudioUnit/VST3 loading can compare real amp-sim or IR-loader plugins against the captured amp/mic target if stable plugin paths are provided.",
                    ],
                }
            ],
        },
        {
            "title": "Current Best Commands",
            "blocks": [
                {
                    "type": "code",
                    "text": "\n".join(
                        [
                            ".venv/bin/python tone_capture_engine.py system-on",
                            ".venv/bin/python tone_capture_engine.py audio-stack-check",
                            ".venv/bin/python rig_capture.py",
                            ".venv/bin/python tone_capture_engine.py refine-rig-capture --help",
                            ".venv/bin/python tone_capture_engine.py build-cabinet-variant --help",
                            ".venv/bin/python tone_capture_engine.py apply-virtual-studio --help",
                            ".venv/bin/python tone_capture_engine.py build-performance-rig --help",
                            ".venv/bin/python tone_capture_engine.py apply-performance-rig --help",
                            ".venv/bin/python research_model.py",
                            ".venv/bin/python tone_capture_engine.py research-stack-check",
                            ".venv/bin/python tone_capture_engine.py freeze-conditioned-dataset --dataset-manifest research_datasets/all_recordings_conditioned.json --output research_datasets/frozen_active_24.json",
                            ".venv/bin/python tone_capture_engine.py train-amp-accuracy-lane --holdout-take TAKE_NAME",
                            ".venv/bin/python tone_capture_engine.py train-torch-reference --help",
                            ".venv/bin/python tone_capture_engine.py hybrid-model-compare --help",
                            ".venv/bin/python tone_capture_engine.py train-all-recordings-amp --list-only",
                            "scripts/mlx_train_performance.sh train-all-recordings-amp --model-sample-rate 96000 --loss-mode detail-spectral --epochs 180",
                            ".venv/bin/python tone_capture_engine.py system-work-log",
                        ]
                    ),
                }
            ],
        },
        {
            "title": "Open Technical Risks",
            "blocks": [
                {
                    "type": "ul",
                    "items": [
                        "No git history exists in this folder, so future work should initialize version control or write append-only system logs after major changes.",
                        "Pickup/blower detection can only be as good as the labeled reference recordings. Bridge, neck, middle, split, blower, boost, and tuning states need clean labels and enough examples.",
                        "The model still needs listening-based validation against DI, mic, and model clips. Spectral metrics alone cannot prove it feels like the amp.",
                        "Direct Amp and merged-cabinet capture remain disabled until a speaker-level-rated DI or load box with speaker thru is available; the passive instrument DI must never receive an amplifier speaker output.",
                        "Cabinet variants are relative to the accepted reference rig. Changing the amp, pedal, reamp level, or preamp gain while recording a variant invalidates the measured correction.",
                        "Virtual mic movement is limited by the measured endpoints supplied as mic A and mic B. More positions, distances, cabinets, and microphones require additional controlled probe captures; the room presets are deterministic early reflections rather than proprietary DynIR measurements.",
                        "The new speaker-impedance stage is an explicit resonance approximation, not a physical reactive load. Exact amp-speaker feedback still requires safe speaker-level measurement hardware and controlled data.",
                        "Input-impedance metadata can detect a mismatched interface setting, but software after A/D conversion cannot recreate pickup loading that was never captured.",
                        "Research candidates remain separate and cannot replace the production MLX model unless held-out amp-target metrics and listening tests support promotion.",
                        "Only one approved training take currently shares the take-031 Maxon/6505/cab/SM57 fingerprint. At least three repeated exact-rig takes are needed before the new high-capacity lane is likely to generalize reliably.",
                        "The research sparse image enforces a 5 GiB working cap. Optional NablAFx CLAP/Frechet checkpoints are intentionally not downloaded.",
                    ],
                }
            ],
        },
        {
            "title": "Reference Links",
            "blocks": [
                {"type": "table", "headers": ["Source", "Use", "URL"], "rows": [[name, use, url] for name, url, use in SOURCE_LINKS]},
            ],
        },
    ]
    return sections


def markdown_from_sections(sections: list[dict[str, Any]]) -> str:
    lines = ["# Guitar/Bass Tone Capture System Work Log", ""]
    for section in sections:
        lines.extend([f"## {section['title']}", ""])
        for block in section["blocks"]:
            if block["type"] == "p":
                lines.extend([block["text"], ""])
            elif block["type"] == "ul":
                lines.extend([f"- {item}" for item in block["items"]])
                lines.append("")
            elif block["type"] == "code":
                lines.extend(["```bash", block["text"], "```", ""])
            elif block["type"] == "table":
                headers = block["headers"]
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in block["rows"]:
                    lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def text_from_sections(sections: list[dict[str, Any]]) -> str:
    lines = ["Guitar/Bass Tone Capture System Work Log", "=" * 46, ""]
    for section in sections:
        lines.extend([section["title"].upper(), "-" * len(section["title"]), ""])
        for block in section["blocks"]:
            if block["type"] == "p":
                lines.extend([block["text"], ""])
            elif block["type"] == "ul":
                lines.extend([f"- {item}" for item in block["items"]])
                lines.append("")
            elif block["type"] == "code":
                lines.extend([block["text"], ""])
            elif block["type"] == "table":
                lines.append(" | ".join(block["headers"]))
                lines.append("-" * 80)
                for row in block["rows"]:
                    lines.append(" | ".join(str(cell).replace("\n", " ") for cell in row))
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def html_from_sections(sections: list[dict[str, Any]], title: str, extra_intro: str = "") -> str:
    body = [f"<h1>{html.escape(title)}</h1>"]
    if extra_intro:
        body.append(extra_intro)
    for section in sections:
        body.append(f"<h2>{html.escape(section['title'])}</h2>")
        for block in section["blocks"]:
            if block["type"] == "p":
                body.append(f"<p>{html.escape(block['text'])}</p>")
            elif block["type"] == "ul":
                body.append("<ul>")
                body.extend(f"<li>{html.escape(str(item))}</li>" for item in block["items"])
                body.append("</ul>")
            elif block["type"] == "code":
                body.append(f"<pre>{html.escape(block['text'])}</pre>")
            elif block["type"] == "table":
                body.append("<table>")
                body.append("<tr>" + "".join(f"<th>{html.escape(str(cell))}</th>" for cell in block["headers"]) + "</tr>")
                for row in block["rows"]:
                    body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
                body.append("</table>")

    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.45; margin: 24px; color: #111827; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.2rem; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #d1d5db; padding-bottom: 0.25rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.25rem; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.45rem; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    pre {{ background: #111827; color: #f9fafb; padding: 1rem; overflow-x: auto; border-radius: 6px; }}
    .handoff {{ background: #f9fafb; border: 1px solid #d1d5db; padding: 1rem; border-radius: 6px; }}
    a.button {{ display: inline-block; padding: 0.65rem 0.9rem; margin: 0.25rem 0.25rem 0.25rem 0; background: #111827; color: white; border-radius: 6px; text-decoration: none; }}
  </style>
</head>
<body>
{body}
</body>
</html>
""".format(title=html.escape(title), body="\n".join(body))


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(7.5 * inch, 0.25 * inch, f"Page {doc.page}")
    canvas.restoreState()


def add_phone_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawCentredString(A5[0] / 2.0, 0.20 * inch, f"Page {doc.page}")
    canvas.restoreState()


def phone_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "Source And Scope",
        "Work Done Since Inception",
        "Recording Dataset",
        "Current Model Outputs",
        "Library Assessment",
        "Open Technical Risks",
    }
    result = [
        {
            "title": "At A Glance",
            "blocks": [
                {
                    "type": "p",
                    "text": (
                        "This is a plain-language history and status report for the guitar and bass tone capture system. "
                        "It describes the recording workflow, amp-model training, live monitor, validation safeguards, "
                        "research tools, storage limits, and remaining work without including source code or terminal commands."
                    ),
                },
                {
                    "type": "ul",
                    "items": [
                        "The production model remains the Apple MLX amp-capture path.",
                        "All complete recordings are discovered and balanced so long takes do not dominate training.",
                        "The recorded amplifier, cabinet, and microphone channel is always the target; DI gain alone is rejected.",
                        "PyTorch, NAM, auraloss, and NablAFx are isolated in the capped ToneCaptureResearch environment.",
                        "The tone system is separated from every Schwab project and rejects Schwab paths.",
                    ],
                },
            ],
        }
    ]
    for section in sections:
        if section["title"] not in keep:
            continue
        blocks = [block for block in section["blocks"] if block["type"] in {"p", "ul"}]
        if blocks:
            result.append({"title": section["title"], "blocks": blocks})
    result.append(
        {
            "title": "Using It In PyCharm",
            "blocks": [
                {
                    "type": "ul",
                    "items": [
                        "Choose 01 Tone System - Start to open the live frequency and waveform monitor.",
                        "Choose 05 Tone System - Record Take for the guided labeled two-channel recorder.",
                        "Choose 06 Tone System - Rig Capture for controlled probe capture, MLX training, refinement, cabinet variants, and rendering.",
                        "Choose 07 Tone System - Research Models for isolated PyTorch, NAM, model comparison, dataset freezing, and the guarded long-memory accuracy run.",
                        "Choose 08 Tone System - Performance Rig to directly build or apply portable accepted-model rigs.",
                    ],
                }
            ],
        }
    )
    return result


def render_phone_pdf(sections: list[dict[str, Any]], pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PhoneTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PhoneHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#111827"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="PhoneBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14.5,
            spaceAfter=7,
            textColor=colors.HexColor("#1f2937"),
        )
    )
    story = [Paragraph("Guitar Tone Capture System", styles["PhoneTitle"])]
    story.append(Paragraph("Phone-Friendly Work Log", styles["PhoneHeading"]))
    story.append(Spacer(1, 5))
    for section in phone_sections(sections):
        story.append(Paragraph(html.escape(section["title"]), styles["PhoneHeading"]))
        for block in section["blocks"]:
            if block["type"] == "p":
                story.append(Paragraph(html.escape(block["text"]), styles["PhoneBody"]))
            elif block["type"] == "ul":
                for item in block["items"]:
                    story.append(Paragraph(f"- {html.escape(str(item))}", styles["PhoneBody"]))
        story.append(Spacer(1, 4))
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A5,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.85 * inch,
        title="Guitar Tone Capture System Phone-Friendly Work Log",
        author="Tone Capture System",
    )
    doc.build(story, onFirstPage=add_phone_page_number, onLaterPages=add_phone_page_number)


def render_pdf(sections: list[dict[str, Any]], pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#111827"),
        )
    )
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8.8, leading=11))
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            backColor=colors.HexColor("#f3f4f6"),
            borderPadding=6,
        )
    )

    story = [Paragraph("Guitar/Bass Tone Capture System Work Log", styles["ReportTitle"])]
    for section_index, section in enumerate(sections):
        if section_index in {5, 7}:
            story.append(PageBreak())
        story.append(Paragraph(section["title"], styles["ReportHeading"]))
        for block in section["blocks"]:
            if block["type"] == "p":
                story.append(Paragraph(html.escape(block["text"]), styles["BodyText"]))
                story.append(Spacer(1, 6))
            elif block["type"] == "ul":
                for item in block["items"]:
                    story.append(Paragraph(f"- {html.escape(str(item))}", styles["SmallBody"]))
                story.append(Spacer(1, 6))
            elif block["type"] == "code":
                story.append(Paragraph(html.escape(block["text"]).replace("\n", "<br/>"), styles["CodeBlock"]))
                story.append(Spacer(1, 8))
            elif block["type"] == "table":
                table_data = [
                    [Paragraph(html.escape(str(cell)), styles["SmallBody"]) for cell in block["headers"]]
                ]
                for row in block["rows"]:
                    table_data.append([Paragraph(html.escape(str(cell)), styles["SmallBody"]) for cell in row])
                col_count = max(1, len(block["headers"]))
                available = 7.2 * inch
                widths = [available / col_count] * col_count
                if col_count == 2:
                    widths = [2.0 * inch, 5.2 * inch]
                elif col_count == 3:
                    widths = [1.8 * inch, 2.0 * inch, 3.4 * inch]
                table = Table(
                    table_data,
                    colWidths=widths,
                    repeatRows=1,
                    splitByRow=1,
                    splitInRow=0,
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 8))

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=1.0 * inch,
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def build_report(project_dir: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = build_sections(project_dir)
    pdf_path = output_dir / f"{REPORT_BASENAME}.pdf"
    phone_pdf_path = output_dir / f"{REPORT_BASENAME}_phone.pdf"
    md_path = output_dir / f"{REPORT_BASENAME}.md"
    txt_path = output_dir / f"{REPORT_BASENAME}.txt"
    html_path = output_dir / f"{REPORT_BASENAME}.html"
    mobile_path = output_dir / f"{REPORT_BASENAME}_mobile.html"

    md_path.write_text(markdown_from_sections(sections), encoding="utf-8")
    txt_path.write_text(text_from_sections(sections), encoding="utf-8")
    html_path.write_text(html_from_sections(sections, "Guitar/Bass Tone Capture System Work Log"), encoding="utf-8")
    render_pdf(sections, pdf_path)
    render_phone_pdf(sections, phone_pdf_path)

    pdf_b64 = base64.b64encode(phone_pdf_path.read_bytes()).decode("ascii")
    intro = f"""
<div class="handoff">
  <p>This page is the mobile fallback. It contains the full report below and includes an embedded PDF download link.</p>
  <p>
    <a class="button" download="{REPORT_BASENAME}_phone.pdf" href="data:application/pdf;base64,{pdf_b64}">Download phone PDF</a>
    <a class="button" href="{REPORT_BASENAME}_phone.pdf">Open phone PDF</a>
    <a class="button" href="{REPORT_BASENAME}.pdf">Open detailed PDF</a>
    <a class="button" href="{REPORT_BASENAME}.txt">Open text fallback</a>
  </p>
</div>
"""
    mobile_path.write_text(
        html_from_sections(sections, "Mobile Tone Capture System Work Log", extra_intro=intro),
        encoding="utf-8",
    )
    return {
        "pdf": pdf_path,
        "phone_pdf": phone_pdf_path,
        "markdown": md_path,
        "text": txt_path,
        "html": html_path,
        "mobile_html": mobile_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PDF and mobile-readable tone capture system work log.")
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    paths = build_report(project_dir=project_dir, output_dir=output_dir)
    print("Wrote system work log:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
