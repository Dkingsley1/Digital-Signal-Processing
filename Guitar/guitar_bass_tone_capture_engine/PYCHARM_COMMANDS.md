# PyCharm Commands

Open this folder as the PyCharm project:

`/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine`

Set the project interpreter to:

`/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine/.venv/bin/python`

## Run Configurations

Use the selector beside the green Run button:

1. `01 Tone System - Start` opens the complete live system and frequency monitor.
2. `02 Tone System - Check` verifies routing without opening the audio window.
3. `03 Tone System - Audio Devices` lists interface device numbers.
4. `04 Tone System - Level Check` checks DI and microphone gain without saving WAV files.
5. `05 Tone System - Record Take` asks for all labels, confirms them, then records both channels.
6. `06 Tone System - Rig Capture` guides probe generation, safe reamp recording,
   causal MLX training, real-guitar refinement, measured cabinet/mic variants,
   validation, Two notes-inspired dual-mic/room rendering, and portable guarded
   performance rigs with dynamic speaker, impedance-response, destination,
   parallel-path, measured-mic, and snapshot controls.
7. `07 Tone System - Research Models` opens the isolated PyTorch/NAM/NablAFx
   benchmark menu. It includes whole-take guarded all-recordings TCN/GRU jobs
   and exact-rig NAM A2 preparation. Menu option 12 freezes the approved dataset,
   and option 13 starts the 96 kHz long-memory accuracy lane. Its external
   research volume is capped and remains separate from the production MLX
   interpreter and every trading project.
8. `08 Tone System - Performance Rig` directly opens the build/apply menu for
   accepted model graphs, snapshots, speaker/cabinet controls, and output destinations.

## PyCharm Terminal

Before using any relative command, enter the project folder. Your prompt should
end in `guitar_bass_tone_capture_engine`, not `~`.

```bash
cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine"
```

Start the full system and live frequency monitor from any terminal:

```bash
cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine" && .venv/bin/python system_on.py
```

Start the guided recording tool:

```bash
.venv/bin/python record_take.py
```

Start the controlled fixed-rig capture wizard:

```bash
.venv/bin/python rig_capture.py
```

Start the portable performance-rig menu directly from any terminal:

```bash
cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine" && .venv/bin/python performance_rig.py
```

Start the isolated research-model wizard:

```bash
.venv/bin/python research_model.py
```

Full research commands are in `RESEARCH_MODEL_COMMANDS.md`.

The direct guarded accuracy command is:

```bash
.venv/bin/python tone_capture_engine.py train-amp-accuracy-lane \
  --holdout-take "TAKE_NAME"
```

It verifies the frozen 24-pair manifest, uses all approved recordings while
favoring the exact amp rig, writes five dry 10-second A/B auditions, and saves a
failed candidate as `*.rejected.pt` instead of replacing an accepted model.

## Recording Library Maintenance

The active library contains only quality-approved recording pairs. Lower-quality
takes are archived outside `recordings/`, not permanently deleted. Preview any
future dataset cleanup before applying it:

```bash
.venv/bin/python tone_capture_engine.py cleanup-unused-takes \
  --dataset datasets/6505_rhythm_sm57_all_guitars.json \
  --preferred-only \
  --cleanup-mode archive \
  --archive-dir /Volumes/ToneCaptureResearch/recording_archive
```

`--preferred-only` refreshes each active pair from the current WAVs and its saved
level profile before selecting takes, so stale quality flags cannot remove a good
recording. Review the listed files, then repeat the command with `--apply` to
archive them and save the refreshed flags. External archives are refused when
their projected use would exceed the 5 GiB Tone Capture Research working cap.
Permanently deleting takes is intentionally separate and requires both
`--cleanup-mode delete` and `--confirm-delete-unused`.

The rig wizard does not play the test signal until its record step and requires
the exact confirmation `PLAY PROBE`. Full direct commands and speaker-routing
safety are in `RIG_CAPTURE_COMMANDS.md`.

List audio devices:

```bash
.venv/bin/python tone_capture_engine.py devices
```

Check levels without writing a take:

```bash
.venv/bin/python tone_capture_engine.py level-check \
  --sample-rate 96000 \
  --duration-s 8 \
  --input-channels 2 \
  --di-channel 1 \
  --target-channel 2 \
  --level-profile aggressive
```

## Direct Recording Command

This example records the Les Paul Custom Axcess bridge pickup in Drop D with
the Maxon OD808. Change the take name, tuning, pickup, and guitar controls for
each new take.

```bash
.venv/bin/python tone_capture_engine.py record \
  --take-name "sm57_amp_take_029_les_paul_custom_axcess_drop_d_bridge_v10_t10_maxon808" \
  --dataset "datasets/6505_rhythm_sm57_all_guitars.json" \
  --sample-rate 96000 \
  --duration-s 120 \
  --input-channels 2 \
  --di-channel 1 \
  --target-channel 2 \
  --di-box "Livewire SPDI passive direct box" \
  --mic "Shure SM57" \
  --amp "Peavey 6505 Mini Head" \
  --cabinet "Egnater Tweaker 1x12 Celestion" \
  --profile-family "6505_rhythm_sm57_all_guitars" \
  --guitar "Les Paul Custom Axcess" \
  --tuning "Drop D" \
  --pickup "bridge" \
  --pickup-mode "bridge humbucker" \
  --guitar-volume "10" \
  --guitar-tone "10" \
  --amp-channel "rhythm" \
  --boost-pedal "Maxon OD808: drive 0, tone 5, balance 10" \
  --mic-position "SM57 close, directly in front of speaker" \
  --performance "palm mutes, open chords, sustained chords, and single-note riffs" \
  --level-profile aggressive
```

The recorder writes three matching files under `recordings/`: clean DI WAV,
amp/cab/SM57 target WAV, and the hardware/label manifest. The dataset receives
the same metadata. Future all-recordings training discovers the WAV pair even
if the level report marks it unsuitable, while the quality gate decides whether
the take should influence the main model.

## Train From All Recordings

Preview the exact rig groups and quality decisions first:

```bash
.venv/bin/python tone_capture_engine.py train-all-recordings-amp \
  --input recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_clean_di.wav \
  --rig-policy conditioned \
  --list-only
```

Run the protected 96 kHz training job:

```bash
scripts/mlx_train_performance.sh train-all-recordings-amp \
  --input recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_clean_di.wav \
  --rig-policy conditioned \
  --model-sample-rate 96000 \
  --render-sample-rate 96000 \
  --loss-mode detail-spectral \
  --skip-per-take-validation
```

`conditioned` uses all quality-approved recordings but gives each fixed
pedal/amp/cab/mic setup its own model condition. Pickup and guitar-volume levels
are preserved instead of peak-normalized. The last 10% of every take is kept
out of training, and a failed candidate is saved as `*.rejected.npz` without
overwriting the current model. `--skip-per-take-validation` saves disk space by
omitting the repeated WAV exports; the safety scores still run.

For the most exact single-rig result, use `06 Tone System - Rig Capture` and
follow `RIG_CAPTURE_COMMANDS.md`. That workflow adds the synchronized probe,
about 85 ms of causal memory, and the optional measured cabinet/microphone stage.
Menu option 10 builds a separate speaker/cabinet/SM57 stage from matched
`amp-preamp` and `amp-cab` probe captures, with routing and mismatch guards.

For the six-part high-accuracy research route, run `07 Tone System - Research
Models`: build the conditioned index, train both whole-take guarded TCN and GRU
candidates, prepare the exact-rig NAM A2 benchmark, and compare only accepted
renders. Full reproducible commands are in `RESEARCH_MODEL_COMMANDS.md`.
