#!/usr/bin/env python3
"""
PyCharm runner for the guitar/bass tone capture system.

Open this file in PyCharm and press Run to prepare the workspace and launch the
live DI plus amp/mic scope.

PyCharm interpreter:
    <project>/.venv/bin/python

Commands that work from any terminal, including a prompt ending in ~:
    cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine" && .venv/bin/python system_on.py
    cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine" && .venv/bin/python performance_rig.py

Useful commands:
    .venv/bin/python tone_capture_engine.py system-on --check-only
    .venv/bin/python tone_capture_engine.py audio-stack-check
    .venv/bin/python tone_capture_engine.py pedalboard-preview --input recordings/example_clean_di.wav --output outputs/example_clean_di_pedalboard_preview.wav
    .venv/bin/python tone_capture_engine.py devices
    .venv/bin/python tone_capture_engine.py level-check --sample-rate 96000 --duration-s 8
    .venv/bin/python record_take.py  # interactive, fully labeled recorder
    .venv/bin/python tone_capture_engine.py train-all-recordings-amp --list-only  # quality-gated plan
    .venv/bin/python tone_capture_engine.py train-all-recordings-amp --model-sample-rate 96000 --loss-mode detail-spectral --epochs 180
    .venv/bin/python tone_capture_engine.py apply-mlx-amp --input recordings/tele_middle_v10_t10_clean_di.wav --model profiles/sm57_amp_mlx_amp_all_recordings_amp_dominant.npz --output outputs/tele_middle_render.wav
    .venv/bin/python tone_capture_engine.py system-work-log
    .venv/bin/python research_model.py  # isolated PyTorch/NAM/NablAFx benchmarks
    .venv/bin/python performance_rig.py  # portable performance-rig build/apply menu

Advanced audio stack routes scikit-learn into a guarded live pickup/blower
classifier when enough labeled recordings exist.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tone_capture_engine import main


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ARGS = ["system-on"]


if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    main(DEFAULT_ARGS + sys.argv[1:])
