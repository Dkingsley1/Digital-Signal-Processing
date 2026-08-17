# Isolated Research Model Commands

The production tone system still uses `.venv` and MLX. PyTorch, NAM, auraloss,
and NablAFx are isolated under `/Volumes/ToneCaptureResearch`. The launcher
enforces a 5 GiB working cap, and every research route rejects paths containing
`schwab_trading_bot`.

In PyCharm, run `07 Tone System - Research Models` for the guided menu.

Verify the stack:

```bash
.venv/bin/python tone_capture_engine.py research-stack-check
```

Prepare one simultaneous DI and amp/mic pair. This aligns latency and polarity,
keeps an unseen validation tail, and writes NAM/NablAFx research recipes:

```bash
.venv/bin/python tone_capture_engine.py prepare-research-capture \
  --di recordings/TAKE_clean_di.wav \
  --target recordings/TAKE_amp_mic_target.wav \
  --manifest recordings/TAKE_hardware_manifest.json \
  --name TAKE
```

Train the PyTorch causal TCN reference candidate:

```bash
.venv/bin/python tone_capture_engine.py train-torch-reference \
  --capture-manifest research_captures/TAKE/research_capture.json \
  --architecture tcn \
  --model profiles/research/TAKE_tcn.pt \
  --output outputs/research/TAKE_tcn_validation.wav \
  --metrics-output outputs/research/TAKE_tcn_metrics.json \
  --epochs 30 \
  --steps-per-epoch 64
```

Use `--architecture gru` for the recurrent comparison, or `lstm` for the legacy
comparison.
A failed model is saved with `.rejected.pt`; it is never promoted over the MLX
production model. The guard requires the candidate to improve against the real
amp target and move audibly away from a level-matched DI.

Apply an accepted PyTorch reference model to a new DI:

```bash
.venv/bin/python tone_capture_engine.py apply-torch-reference \
  --input recordings/NEW_TAKE_clean_di.wav \
  --model profiles/research/TAKE_tcn.pt \
  --output outputs/research/NEW_TAKE_torch_render.wav
```

Rejected models cannot be applied by the normal route.

Compare MLX, PyTorch, and NAM renders with the same auraloss/amp-tone gate:

```bash
.venv/bin/python tone_capture_engine.py hybrid-model-compare \
  --di research_captures/TAKE/validation_input.wav \
  --target research_captures/TAKE/validation_amp_target.wav \
  --candidate MLX=outputs/TAKE_mlx.wav \
  --candidate TORCH=outputs/research/TAKE_tcn_validation.wav \
  --candidate NAM=outputs/research/TAKE_nam.wav \
  --output outputs/research/TAKE_hybrid_comparison.json
```

Index all complete recordings with fixed-rig and guitar/pickup/tuning labels:

```bash
.venv/bin/python tone_capture_engine.py build-conditioned-dataset \
  --dataset datasets/6505_rhythm_sm57_all_guitars.json \
  --output research_datasets/all_recordings_conditioned.json \
  --allow-mixed-rigs
```

The index keeps explicit rig, guitar, tuning, pickup, volume, and tone labels.
Different rigs are never silently merged into one target. Mixed rigs are only
allowed here because the trainer receives those labels as condition channels.

Train across all recordings while excluding one complete recording from every
optimization step. The matching rig receives 75% of update sampling and all
other recordings remain active as labeled regularization data:

```bash
.venv/bin/python tone_capture_engine.py train-conditioned-torch-reference \
  --dataset-manifest research_datasets/all_recordings_conditioned.json \
  --holdout-take TAKE_NAME \
  --architecture tcn \
  --sample-rate 96000 \
  --model profiles/research/all_recordings_TAKE_NAME_tcn_96k.pt \
  --output outputs/research/all_recordings_TAKE_NAME_tcn_96k.wav \
  --metrics-output outputs/research/all_recordings_TAKE_NAME_tcn_96k_metrics.json \
  --epochs 30 \
  --steps-per-epoch 64 \
  --chunk-samples 8192 \
  --focus-rig-fraction 0.75
```

Change `tcn` to `gru` for the recurrent benchmark. Training combines waveform,
ESR, pre-emphasis, linear and mel multi-resolution STFT, envelope, transient,
and DC losses. A candidate must pass the amp-tone guard on the untouched whole
take; a failure is renamed `*.rejected.pt` and cannot replace production.

## High-Accuracy Lane

Hash-lock the approved 24-pair dataset before a long experiment. This stores
paths, byte sizes, and SHA-256 hashes; it does not duplicate the WAV files:

```bash
.venv/bin/python tone_capture_engine.py freeze-conditioned-dataset \
  --dataset-manifest research_datasets/all_recordings_conditioned.json \
  --output research_datasets/frozen_active_24.json
```

Run the 96 kHz long-memory accuracy preset with one untouched whole-take
holdout:

```bash
.venv/bin/python tone_capture_engine.py train-amp-accuracy-lane \
  --holdout-take TAKE_NAME
```

This preset verifies every frozen asset, uses a stacked gated causal TCN with
about 341 ms of memory, reserves 10% tails from training takes for checkpoint
selection, and keeps the complete named holdout outside optimization. Every
approved recording remains active, while 85% of updates favor the holdout's
exact pedal/amp/cab/microphone fingerprint.

The `fullness-v2` objective combines phase-sensitive waveform and transient
terms with linear/mel multi-resolution STFT, six guitar-band energy checks,
multiscale log-RMS envelopes, level, and crest behavior. Promotion requires the
untouched take to pass quiet, medium, loud, transient, and sustained listening
sections; training loss alone cannot promote it.

The endpoint is the complete dry Peavey/cab/close-SM57 chain. The validation
render adds no room and no extra cabinet. Five A-real/B-model clips are written
under `outputs/research/amp_accuracy_lane_auditions`, using one global model
level correction rather than normalizing each section independently.

The first run against take 031 was correctly rejected: `19.94 dB` spectral
error, `-0.326` correlation, and `0%` listening-section pass rate. Only one
approved training take currently shares that exact Maxon/6505 rig fingerprint,
so at least three repeated exact-rig training takes are the next data priority.
The accepted NAM A2 reference remains better at `12.56 dB` and `0.676`
correlation.

Prepare the exact-rig NAM 0.13 A2/PackedWaveNet benchmark. Only recordings with
the holdout's exact pedal/amp/cab/mic fingerprint enter NAM training:

```bash
.venv/bin/python tone_capture_engine.py prepare-conditioned-nam-a2 \
  --dataset-manifest research_datasets/all_recordings_conditioned.json \
  --holdout-take TAKE_NAME \
  --name TAKE_NAME_holdout \
  --output-dir /Volumes/ToneCaptureResearch/captures/TAKE_NAME_holdout \
  --sample-rate 96000 \
  --nam-epochs 30 \
  --nam-batch-size 8 \
  --nam-window-samples 4096 \
  --nam-train-batches-per-epoch 24 \
  --nam-validation-batches-per-epoch 8
```

The generated `NAM_A2_TRAIN_COMMAND.txt` starts training later. Add
`--start-training` only when the capped research image is mounted and you are
ready for a CPU job. The defaults bound internal iteration time; setting either
batch limit to `0` intentionally enables an unbounded full-data epoch. Final
acceptance still scores windows distributed across the complete untouched take.

Render the highest-quality submodel from the exported A2 container, then pass
that WAV to `hybrid-model-compare` before treating it as an accepted candidate:

```bash
.venv/bin/python tone_capture_engine.py apply-nam-reference \
  --input /Volumes/ToneCaptureResearch/captures/TAKE_NAME_holdout/validation_input.wav \
  --model /Volumes/ToneCaptureResearch/captures/TAKE_NAME_holdout/nam_a2/runs/model.nam \
  --output outputs/research/TAKE_NAME_nam_a2_validation.wav
```

Build a separate speaker/cabinet/SM57 stage after recording the identical
calibrated probe through an approved amp-preamp line return and through the
complete amp/cab/microphone path:

```bash
.venv/bin/python tone_capture_engine.py build-separated-cabinet \
  --preamp-capture-manifest rig_captures/RIG_preamp_rig_capture_manifest.json \
  --amp-cab-capture-manifest rig_captures/RIG_amp_cab_rig_capture_manifest.json \
  --profile profiles/cabinet_variants/RIG_cabinet_sm57.npz \
  --comparison-output outputs/RIG_cabinet_sm57_comparison.wav \
  --name RIG_cabinet_sm57
```

The command rejects mismatched probes, sample rates, send trim, pedal, amp, or
knob settings. Never connect an amplifier speaker output to an interface.

This hybrid system exchanges WAV files, manifests, metrics, and approved model
artifacts. It does not convert tensors between MLX and PyTorch inside a live
training graph, which would add latency and duplicate memory without improving
the capture target.
