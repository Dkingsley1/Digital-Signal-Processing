#!/usr/bin/env python3
"""Smoke-test the research guard with a real target and a DI-only negative control."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_DIR / "scripts/research_python.sh"
WORKER = PROJECT_DIR / "scripts/research_audio_worker.py"


def main() -> None:
    di = PROJECT_DIR / "recordings/sm57_amp_take_001_clean_di.wav"
    target = PROJECT_DIR / "recordings/sm57_amp_take_001_amp_mic_target.wav"
    if not di.exists() or not target.exists():
        raise SystemExit("Smoke-test recording pair is unavailable.")
    with tempfile.TemporaryDirectory(prefix="tone_research_guard_") as directory:
        report = Path(directory) / "metrics.json"
        subprocess.run(
            [
                str(LAUNCHER),
                str(WORKER),
                "metrics",
                "--di", str(di),
                "--target", str(target),
                "--candidate", f"REAL_TARGET={target}",
                "--candidate", f"DI_GAIN_ONLY={di}",
                "--output", str(report),
            ],
            cwd=PROJECT_DIR,
            check=True,
        )
        payload = json.loads(report.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    assert candidates["REAL_TARGET"]["amp_tone_guard_passed"] is True
    assert candidates["DI_GAIN_ONLY"]["amp_tone_guard_passed"] is False
    assert abs(candidates["DI_GAIN_ONLY"]["movement_from_gain_only_db"]) < 1e-6
    print("Research amp-tone guard smoke passed: real target accepted, DI gain rejected.")


if __name__ == "__main__":
    main()
