#!/usr/bin/env python3
"""Interactive PyCharm runner for a labeled two-channel training take."""

from __future__ import annotations

import re
import shlex
import os
from pathlib import Path

from tone_capture_engine import main


PROJECT_DIR = Path(__file__).resolve().parent

GUITARS = {
    "1": {
        "guitar": "Les Paul Custom Axcess",
        "slug": "les_paul_custom_axcess",
        "pickup_mode": "bridge humbucker",
    },
    "2": {
        "guitar": "Strandberg Boden Mahogany/Maple",
        "slug": "strandberg_nazgul_sentient_mahogany_maple",
        "pickup_mode": "Seymour Duncan Nazgûl/Sentient set, bridge Nazgûl full humbucker",
    },
    "3": {
        "guitar": "Strandberg Boden Swamp Ash/Richlite",
        "slug": "strandberg_pegasus_sentient_swamp_ash_richlite",
        "pickup_mode": "Seymour Duncan Pegasus/Sentient set, bridge Pegasus full humbucker",
    },
    "4": {
        "guitar": "Modded Telecaster",
        "slug": "telecaster",
        "pickup_mode": "Hot Rails full humbucker",
    },
}


def ask(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def slugify(value: str) -> str:
    normalized = value.lower().replace("nazgûl", "nazgul")
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def next_take_number() -> int:
    take_numbers = []
    for path in (PROJECT_DIR / "recordings").glob("sm57_amp_take_*_clean_di.wav"):
        match = re.match(r"sm57_amp_take_(\d+)", path.name)
        if match:
            take_numbers.append(int(match.group(1)))
    return max(take_numbers, default=0) + 1


def choose_guitar() -> dict[str, str]:
    print("\nGuitar preset")
    print("  1. Les Paul Custom Axcess")
    print("  2. Strandberg Mahogany/Maple - Nazgûl/Sentient")
    print("  3. Strandberg Swamp Ash/Richlite - Pegasus/Sentient")
    print("  4. Modded Telecaster")
    print("  5. Custom label")
    selection = ask("Choose", "1")
    if selection in GUITARS:
        return dict(GUITARS[selection])

    guitar = ask("Full guitar label")
    if not guitar:
        raise SystemExit("A guitar label is required.")
    return {
        "guitar": guitar,
        "slug": slugify(guitar),
        "pickup_mode": "full pickup",
    }


def pickup_mode_default(guitar: str, pickup: str, preset_default: str) -> str:
    pickup = pickup.lower()
    if guitar == "Les Paul Custom Axcess":
        if pickup == "middle":
            return "mixed neck and bridge humbuckers"
        return f"{pickup} humbucker"
    if pickup != "bridge" and "bridge" in preset_default.lower():
        return preset_default.replace("bridge", pickup).replace("Bridge", pickup.title())
    return preset_default


def build_record_args() -> list[str]:
    guitar_preset = choose_guitar()
    pickup = ask("Pickup (bridge/neck/middle)", "bridge").lower()
    mode_default = pickup_mode_default(
        guitar_preset["guitar"], pickup, guitar_preset["pickup_mode"]
    )
    pickup_mode = ask("Pickup/wiring label", mode_default)
    tuning = ask("Tuning", "Standard")
    volume = ask("Guitar volume", "10")
    tone = ask("Guitar tone", "10")

    print("\nBoost pedal")
    print("  1. None")
    print("  2. Maxon OD808 - drive 0, tone 5, balance 10")
    print("  3. Custom label")
    boost_choice = ask("Choose", "1")
    if boost_choice == "2":
        boost = "Maxon OD808: drive 0, tone 5, balance 10"
        boost_slug = "maxon808"
    elif boost_choice == "3":
        boost = ask("Boost pedal/settings")
        boost_slug = slugify(boost) if boost else "no_boost"
    else:
        boost = "none"
        boost_slug = "no_boost"

    duration = ask("Take duration in seconds", "120")
    performance = ask(
        "Performance",
        "palm mutes, open chords, sustained chords, and single-note riffs",
    )
    notes = ask("Extra take notes (optional)")
    device = ask("Audio device index/name (blank = system default)")

    number = next_take_number()
    default_take_name = "_".join(
        [
            f"sm57_amp_take_{number:03d}",
            guitar_preset["slug"],
            slugify(tuning),
            slugify(pickup),
            f"v{slugify(volume)}",
            f"t{slugify(tone)}",
            boost_slug,
        ]
    )
    take_name = ask("Take name", default_take_name)

    args = [
        "record",
        "--take-name", take_name,
        "--dataset", "datasets/6505_rhythm_sm57_all_guitars.json",
        "--sample-rate", "96000",
        "--duration-s", duration,
        "--input-channels", "2",
        "--di-channel", "1",
        "--target-channel", "2",
        "--di-box", "Livewire SPDI passive direct box",
        "--mic", "Shure SM57",
        "--amp", "Peavey 6505 Mini Head",
        "--cabinet", "Egnater Tweaker 1x12 Celestion",
        "--profile-family", "6505_rhythm_sm57_all_guitars",
        "--guitar", guitar_preset["guitar"],
        "--tuning", tuning,
        "--pickup", pickup,
        "--pickup-mode", pickup_mode,
        "--guitar-volume", volume,
        "--guitar-tone", tone,
        "--amp-channel", "rhythm",
        "--boost-pedal", boost,
        "--mic-position", "SM57 close, directly in front of speaker",
        "--performance", performance,
        "--level-profile", "aggressive",
    ]
    if notes:
        args.extend(["--take-notes", notes])
    if device:
        args.extend(["--device", device])
    return args


def run() -> None:
    args = build_record_args()
    command = shlex.join([str(PROJECT_DIR / ".venv/bin/python"), "tone_capture_engine.py", *args])
    print("\nReady to record both channels:")
    print(command)
    print("\nChannel 1 = clean DI | Channel 2 = amp/cab/SM57")
    if ask("Start recording now? (yes/no)", "no").lower() not in {"y", "yes"}:
        print("Recording cancelled. No files were written.")
        return

    take_name = args[args.index("--take-name") + 1]
    existing = [
        PROJECT_DIR / "recordings" / f"{take_name}_clean_di.wav",
        PROJECT_DIR / "recordings" / f"{take_name}_amp_mic_target.wav",
        PROJECT_DIR / "recordings" / f"{take_name}_hardware_manifest.json",
    ]
    if any(path.exists() for path in existing):
        raise SystemExit(f"Take already exists; choose a new take name: {take_name}")
    main(args)


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    run()
