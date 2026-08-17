# Guitar/Bass Amp Tone Capture Engine

This is a compact Python audio DSP prototype for capturing and reusing guitar or
bass amplifier tones.

It is inspired by tone-profile workflows: provide a clean DI recording and a
processed amp/cab target recording, then save a reusable JSON profile. The saved
profile can later be applied to another DI performance.

This is a portfolio prototype, not a commercial neural amp modeler or hardware
clone.

## What It Demonstrates

- WAV audio loading and exporting
- DI/target alignment
- Dynamic nonlinear saturation, sag, and compression modeling
- Regularized frequency-domain deconvolution
- Cabinet/tone impulse response capture
- JSON tone profile storage with nonlinear behavior parameters
- Profile recall and application to new guitar/bass DI audio
- Optional two-channel audio interface recording layer
- DI box, mic, amp, pad, ground-lift, and channel-map metadata
- Optional Apple MLX neural residual layer for Apple Silicon
- Demo guitar and bass signal generation

## Quick Start

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 tone_capture_engine.py demo
```

The demo writes:

```text
outputs/
  audio/
    demo_guitar_clean_di.wav
    demo_guitar_amp_target.wav
    demo_guitar_captured_match.wav
    demo_guitar_profiled_new_di.wav
    demo_bass_clean_di.wav
    demo_bass_amp_target.wav
    demo_bass_captured_match.wav
    demo_bass_profiled_new_di.wav
  profiles/
    demo_guitar_tone_profile.json
    demo_bass_tone_profile.json
  tone_capture_summary.txt
```

## PyCharm System On

Open `guitar_bass_tone_capture_engine` as the PyCharm project, choose the
`01 Tone System - Start` run configuration, and press Run. You can also open
`system_on.py` directly and press Run. The complete short command reference is
in `PYCHARM_COMMANDS.md`.

That launcher runs:

```bash
cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine" && .venv/bin/python system_on.py
```

The direct performance-rig launcher is:

```bash
cd "/Users/dankingsley/Documents/New project/guitar_bass_tone_capture_engine" && .venv/bin/python performance_rig.py
```

`system-on` prepares the capture folders, uses the system default audio input,
routes channel 1 as clean DI and channel 2 as the amp/mic target, writes live
scope telemetry to `logs/live_scope/latest.jsonl`, and opens the fast PyQtGraph
live scope.

To verify the PyCharm path without opening the audio window:

```bash
.venv/bin/python system_on.py --check-only
```

For recording, select `05 Tone System - Record Take` in PyCharm or run the
guided recorder. It supplies the normal 96 kHz two-channel routing and asks for
the guitar, pickup set, tuning, controls, boost, performance, and take name
before it writes anything:

```bash
.venv/bin/python record_take.py
```

For a controlled fixed-rig neural capture, select
`06 Tone System - Rig Capture` or run:

```bash
.venv/bin/python rig_capture.py
```

This path generates a known multilevel probe, records it through a reamp box and
one unchanged pedal/amp/cab/SM57 setup, measures latency and polarity, trains a
causal long-memory MLX model with 2x oversampling, and rejects models that fail
held-out validation or collapse to DI gain. A second real-guitar refinement pass
then corrects chord intermodulation, pick attack, palm mutes, sustain, and volume
cleanup while learning the DI-to-reamp input calibration. See
`RIG_CAPTURE_COMMANDS.md` for the complete commands and required speaker-routing
precautions.

The same wizard also provides a Fender Tone Master-inspired modular cabinet and
microphone layer. Repeating the controlled probe with only the cabinet, mic,
position, distance, or axis changed produces a validated correction impulse that
can be selected after the nonlinear rig model. This keeps the amp capture stable
while allowing measured cabinet/mic choices, blend amount, low cut, and high cut.

A Two notes Wall of Sound-inspired virtual studio can then combine two accepted
measured cabinet/mic variants. It provides continuous equal-power morphing,
independent levels and stereo pans, sub-sample phase timing, polarity inversion,
distance-controlled early reflections, and optional additional speaker overload.
These controls affect rendering only. The virtual power-amp stage stays off
because a full Peavey rig capture already contains the real power amp; stacking
another one would distort the learned result. Use `apply-virtual-studio --help`
or the application step in `rig_capture.py`.

## Capture Your Own Profile

From existing WAV files:

```bash
python3 tone_capture_engine.py capture \
  --di clean_guitar_di.wav \
  --target amp_recording.wav \
  --instrument guitar \
  --name modern_gain_profile \
  --profile profiles/modern_gain_profile.json \
  --reconstructed outputs/modern_gain_match.wav
```

With a hardware manifest from a DI/interface recording:

```bash
python3 tone_capture_engine.py capture \
  --di recordings/take_001_clean_di.wav \
  --target recordings/take_001_amp_mic_target.wav \
  --manifest recordings/take_001_hardware_manifest.json \
  --instrument guitar \
  --name sm57_amp_capture \
  --profile profiles/sm57_amp_capture.json \
  --reconstructed outputs/sm57_amp_capture_match.wav
```

## Real Interface + DI Box Capture

Install the optional recording dependency if you want the script to talk to your
audio interface directly:

```bash
python3 -m pip install -r requirements-interface.txt
```

List available audio input devices:

```bash
python3 tone_capture_engine.py devices
```

Typical routing:

```text
Guitar/bass -> DI box 1/4 inch input
DI box XLR output -> interface channel 1 clean DI
DI box 1/4 inch THRU -> amplifier input
SM57 on speaker cabinet -> interface channel 2 amp/mic target
```

Before committing to a full take, run a short level check. This writes no WAV
files; it only tells you whether the clean DI and mic target are in a useful
recording range.

```bash
python3 tone_capture_engine.py level-check \
  --device 1 \
  --sample-rate 96000 \
  --duration-s 8 \
  --input-channels 2 \
  --di-channel 1 \
  --target-channel 2 \
  --di-box "Livewire SPDI passive direct box" \
  --mic "Shure SM57"
```

The script treats `-24 to -6 dBFS` as usable and `-18 to -10 dBFS` as the
preferred range. Retake or adjust gain if either channel is marked too quiet,
silent, or clipping risk.

For guitar takes with wider picking dynamics, add a level profile. This changes
the advice and target ranges so a hard palm-muted take is judged with more
transient headroom than a controlled clean part:

```bash
python3 tone_capture_engine.py level-check \
  --device 1 \
  --sample-rate 96000 \
  --duration-s 10 \
  --input-channels 2 \
  --di-channel 1 \
  --target-channel 2 \
  --level-profile aggressive
```

Available profiles are `light`, `normal`, `dynamic`, `aggressive`, and
`extreme`. For hard-picked humbuckers into the 6505 rhythm channel, start with
`aggressive`. Use `extreme` only if the real take keeps jumping far above the
short level-check peaks.

For a live oscilloscope-style view, install the optional visual dependency:

```bash
python3 -m pip install -r requirements-visual.txt
```

Then open a live graph of the clean DI and mic target. The graph shows the
sound wave over time, the live frequency spectrum, and a level-match meter with
the usable and ideal dBFS ranges:

```bash
python3 tone_capture_engine.py live-scope \
  --device 1 \
  --sample-rate 96000 \
  --input-channels 2 \
  --di-channel 1 \
  --target-channel 2 \
  --responsive \
  --visual-smoothing hyperfluid \
  --fft-size 8192 \
  --level-profile aggressive
```

Use `--view waveform` for only the live sound wave, `--view spectrum` for only
the frequency graph, or leave the default `--view both`. Use `--responsive` when
you want the scope to react faster to pick attack. Use `--visual-smoothing
hyperfluid` for the most detailed fast mode: an 8192-point spectrum plus a short
attack analyzer blended into the display, with the level meters and capture
metrics tied to a short response window. Use `--visual-smoothing fluid` for a
slightly smoother/liquid analyzer, `--visual-smoothing studio` for a
faster/tighter analyzer, or `--visual-smoothing ultra` when you want slower,
glossier movement.

The detailed view also includes a live `Amp/Mic - DI` tone-difference curve. This
shows the frequency areas where the amp/cab/SM57 chain is adding or removing
energy compared with the clean DI. The readout summarizes low, mid, bite, and air
differences, while the level meter shows peak, RMS, crest factor, and the dB
difference between the DI and mic channels.

The waveform panel defaults to stacked oscilloscope lanes so the DI and mic do
not visually cover each other. It also shows RMS rails, peak dots, peak-to-peak
amplitude, zero-crossing rate, and slew rate. Use `--waveform-layout overlay` if
you want both waveforms drawn on the same zero line.

The capture metrics panel is laid out as a wide dashboard row with live
delay/correlation, clipping percentage, estimated noise floor, transient rate,
spectral centroid, 85% rolloff, and band energy percentages for
low/body/mid/bite/air ranges. Use `--hide-metrics` for a lighter graph.

The PyQtGraph scope also has a pickup/blower readout. Fast level meters still
use the short `--metrics-window-ms` window, while pickup tone, output, resonant
peak, rolloff, bite, and air changes use the longer `--source-analysis-ms`
window. That makes pickup selector moves and blower-switch changes show up as
stable "hotter/brighter", "blower-like", or "blower off" events instead of only
as pick-attack spikes. The same values are written to the live feature log under
`rolloff_pickup`.

Write a hardware plan/manifest without recording:

```bash
python3 tone_capture_engine.py hardware-plan \
  --take-name les_paul_sm57_take_001 \
  --di-box "Livewire SPDI passive direct box" \
  --di-channel 1 \
  --target-channel 2 \
  --mic "Shure SM57" \
  --output hardware/les_paul_sm57_plan.json
```

Record both channels into WAV files:

```bash
python3 tone_capture_engine.py record \
  --take-name les_paul_sm57_take_001 \
  --duration-s 25 \
  --di-box "Livewire SPDI passive direct box" \
  --di-channel 1 \
  --target-channel 2
```

Record and immediately capture a reusable tone profile:

```bash
python3 tone_capture_engine.py record-capture \
  --take-name les_paul_sm57_take_001 \
  --duration-s 25 \
  --instrument guitar \
  --name les_paul_sm57_amp_profile \
  --di-box "Livewire SPDI passive direct box" \
  --di-channel 1 \
  --target-channel 2 \
  --profile profiles/les_paul_sm57_amp_profile.json \
  --reconstructed outputs/les_paul_sm57_amp_match.wav
```

For profile-building sessions, add take metadata and a dataset manifest. This
keeps each WAV pair tied to the guitar, pickup, knob settings, amp channel, mic
position, and profile family that produced it.

```bash
python3 tone_capture_engine.py record-capture \
  --take-name sm57_amp_take_013_tele_bridge_volume_7_tone_10 \
  --dataset datasets/6505_rhythm_sm57_tele.json \
  --device 1 \
  --sample-rate 96000 \
  --duration-s 120 \
  --input-channels 2 \
  --di-channel 1 \
  --target-channel 2 \
  --instrument guitar \
  --name sm57_amp_profile_take_013_tele_bridge_volume_7_tone_10 \
  --di-box "Livewire SPDI passive direct box" \
  --mic "Shure SM57" \
  --amp "Peavey 6505 Mini Head" \
  --cabinet "Egnater Tweaker 1x12 Celestion" \
  --profile-family "6505_rhythm_sm57_tele" \
  --guitar "Modded Telecaster" \
  --pickup "bridge" \
  --pickup-mode "Hot Rails full" \
  --guitar-volume "7" \
  --guitar-tone "10" \
  --amp-channel "rhythm" \
  --boost-pedal "none" \
  --mic-position "SM57 close, directly in front of speaker" \
  --performance "palm mutes, open chords, single-note riffs" \
  --profile profiles/sm57_amp_profile_take_013_tele_bridge_volume_7_tone_10.json \
  --reconstructed outputs/sm57_amp_match_take_013_tele_bridge_volume_7_tone_10.wav
```

Safety note: never connect an amplifier speaker output directly to an audio
interface. The mic hears the cabinet; the interface does not receive speaker
power.

## Apply a Saved Profile

```bash
python3 tone_capture_engine.py apply \
  --input another_clean_di.wav \
  --profile profiles/modern_gain_profile.json \
  --output outputs/another_clean_di_profiled.wav
```

## Audible Tone-Match Audition

If a real mic capture sounds too close to the raw DI, render a stronger spectral
tone-match audition. This uses the SM57 target's frequency shape directly, so it
is useful for checking that the DI can take on the amp/cab color.

```bash
python3 tone_capture_engine.py tone-match \
  --di recordings/sm57_amp_take_001_clean_di.wav \
  --target recordings/sm57_amp_take_001_amp_mic_target.wav \
  --profile profiles/sm57_amp_profile.json \
  --amp-style mic-layer \
  --output outputs/sm57_amp_mic_layer.wav \
  --comparison-output outputs/sm57_amp_mic_then_mic_layer.wav
```

## Optional MLX Neural Residual Layer

The MLX layer is optional. The main DSP system works without it. Install MLX
only inside this project environment:

```bash
python3 -m pip install -r requirements-mlx.txt
```

For the best audio/modeling path on this project, install the advanced audio
stack. This keeps MLX, adds float WAV I/O, high-quality resampling, loudness
analysis, and optional effect/plugin tooling:

```bash
.venv/bin/python -m pip install -r requirements-audio-advanced.txt
```

Verify the active routes:

```bash
.venv/bin/python tone_capture_engine.py audio-stack-check
```

Advanced stack routing:

- `soundfile`: float WAV read/write.
- `soxr`: very-high-quality resampling.
- `librosa`: extra spectral descriptors in the recording quality gate.
- `pyloudnorm`: integrated loudness checks in the recording quality gate.
- `noisereduce`: explicit `denoise-preview` command only; training audio is not denoised implicitly.
- `pedalboard`: explicit `pedalboard-preview` effect-chain command and future plugin/augmentation tests.
- `scikit-learn`: optional calibrated live pickup/blower reference classifier when enough labeled recordings exist.

Noise reduction preview, when a take has hiss or room noise you want to inspect:

```bash
.venv/bin/python tone_capture_engine.py denoise-preview \
  --input recordings/example_clean_di.wav \
  --output outputs/example_clean_di_denoise_preview.wav
```

Pedalboard preview, for explicit effect-chain tests:

```bash
.venv/bin/python tone_capture_engine.py pedalboard-preview \
  --input recordings/example_clean_di.wav \
  --output outputs/example_clean_di_pedalboard_preview.wav \
  --preset tighten
```

The heavier PyTorch/NAM/NablAFx research stack is separate because it is not
required by the current MLX trainer. Do not install it into `.venv`. It lives in
the isolated `/Volumes/ToneCaptureResearch` sparse image and uses a 5 GiB
working cap. It is routed by:

```bash
.venv/bin/python tone_capture_engine.py research-stack-check
.venv/bin/python research_model.py
```

The research lane adds conditioned MPS PyTorch TCN/GRU/LSTM candidates, linear
and mel multi-resolution spectral losses, multiscale envelope and transient
losses, NAM 0.13 A2/PackedWaveNet reference exports, NablAFx recipes, and a
common DI-gain-only rejection guard. All-recordings training preserves explicit
rig and guitar labels, favors the matching rig, and excludes complete holdout
takes from optimization. A rejected candidate cannot replace MLX production.
The `train-amp-accuracy-lane` preset adds a hash-locked 24-pair dataset, a
96 kHz stacked gated TCN with about 341 ms of causal memory, guitar-band and
log-RMS fullness losses, training-tail checkpoint selection, five dry listening
sections, and audible held-out promotion gates. Its first take-031 candidate
was rejected, so the accepted NAM A2 reference remains the current research
leader; more repeated captures of that exact Maxon/6505 rig are still needed.
See `RESEARCH_MODEL_COMMANDS.md`.

## Recording Library Maintenance

Keep the active `recordings/` folder limited to quality-approved DI/amp pairs.
Use `cleanup-unused-takes` without `--apply` first to preview every affected
file. Archive weaker takes to `/Volumes/ToneCaptureResearch/recording_archive`
instead of deleting them; the command refuses an external archive whose
projected usage would exceed the 5 GiB research working cap. Archived or
inactive dataset entries are also excluded when the conditioned all-recordings
index is rebuilt. With `--preferred-only`, the command recalculates active-pair
quality from the current WAVs and each take's saved level profile before making
its selection, preventing stale quality flags from archiving a good take.

The ready-to-run maintenance command is in `PYCHARM_COMMANDS.md`.

## System Work Log PDF

Install the reporting stack when you need a PDF, mobile fallback page, Markdown,
and text copy of the system history:

```bash
.venv/bin/python -m pip install -r requirements-reporting.txt
```

Build the report:

```bash
.venv/bin/python tone_capture_engine.py system-work-log
```

The command writes:

```text
reports/tone_capture_system_work_log.pdf
reports/tone_capture_system_work_log_mobile.html
reports/tone_capture_system_work_log.html
reports/tone_capture_system_work_log.md
reports/tone_capture_system_work_log.txt
```

Use the mobile HTML file if the app will not open the PDF directly. It contains
the full report text and an embedded PDF download link.

For long MLX training runs, use the performance launcher. It keeps the Mac awake
with `caffeinate`, uses the project venv, and sets NumPy/Accelerate thread counts
from the detected performance-core count. macOS still chooses exact P-core/E-core
placement, so this is a performance-friendly launch mode rather than hard CPU
pinning.

```bash
scripts/mlx_train_performance.sh train-mlx-amp --help
```

For the main "learn the amp from everything and apply it to a DI" workflow, use
the amp-dominant all-recordings command. It auto-discovers every paired
`*_clean_di.wav` and `*_amp_mic_target.wav` take in `recordings/`, skips
`level_test*` files unless requested, trains the DI-to-amp/mic model with
quality-gated balanced sampling, and renders the learned amplifier/cab/SM57 tone
onto the requested DI. The quality gate excludes DI-like captures and downweights
fizz/body outliers before they can teach the model the wrong tone. Fixed rigs
are represented by separate one-hot conditions, capture levels are preserved,
fractional latency and polarity are corrected, and every take reserves an unseen
tail for model promotion:

```bash
scripts/mlx_train_performance.sh train-all-recordings-amp \
  --model-sample-rate 96000 \
  --render-sample-rate 96000 \
  --rig-policy conditioned \
  --loss-mode detail-spectral \
  --skip-per-take-validation \
  --input recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_clean_di.wav
```

Preview the takes it will use without training. This shows KEEP/DOWNWEIGHT/EXCLUDE
decisions, rig fingerprints, quality weights, DI baseline spectral distance, and
DI/mic correlation:

```bash
.venv/bin/python tone_capture_engine.py train-all-recordings-amp --list-only
```

If you only want the lower-level trainer, the equivalent all-recordings input is:

```bash
scripts/mlx_train_performance.sh train-mlx-amp \
  --recordings-dir recordings \
  --model profiles/sm57_amp_mlx_amp_all_recordings_amp_dominant.npz \
  --output outputs/sm57_amp_mlx_amp_all_recordings_amp_dominant_training_render.wav \
  --comparison-output outputs/sm57_amp_mic_then_all_recordings_amp_dominant_training.wav \
  --per-take-output-dir outputs/per_take_validation_all_recordings_amp_dominant \
  --model-sample-rate 96000 \
  --epochs 180 \
  --context-radius 960 \
  --hidden-dim 384 \
  --take-sampling balanced \
  --conditioning-mode source-stats \
  --loss-mode detail-spectral \
  --detail-chunks-per-epoch 160 \
  --transient-loss-weight 0.50 \
  --highfreq-loss-weight 0.50 \
  --envelope-loss-weight 0.18 \
  --esr-loss-weight 0.35 \
  --spectral-loss-weight 0.22 \
  --cab-lowpass-hz 7800 \
  --cab-presence-db 3.5 \
  --cab-air-db 0.8 \
  --render-sample-rate 96000
```

Training writes a candidate first. It replaces the requested production model
only when the unseen tails meet the absolute spectral, correlation, level, and
amp-tone requirements and do not regress against the existing model. The
optional `--hybrid-cabinet-profile` applies an accepted measured cabinet/mic
response after the nonlinear model; it is not synthesized when no measured
profile exists.

For real two-channel captures, use the MLX bridge mode. It first learns a
mic-derived nonlinear DSP bridge from DI to SM57, then trains MLX on the
remaining difference:

```bash
python3 tone_capture_engine.py train-mlx-bridge \
  --di recordings/sm57_amp_take_001_clean_di.wav \
  --target recordings/sm57_amp_take_001_amp_mic_target.wav \
  --base-model profiles/sm57_amp_mic_bridge.json \
  --model profiles/sm57_amp_mlx_bridge.npz \
  --base-output outputs/sm57_amp_mic_bridge.wav \
  --output outputs/sm57_amp_mlx_bridge.wav \
  --comparison-output outputs/sm57_amp_mic_then_mlx_bridge.wav
```

For the most direct "compare the whole DI frequency range to the whole SM57
frequency range" workflow, train the full-spectrum MLX bridge:

```bash
python3 tone_capture_engine.py train-mlx-spectrum \
  --di recordings/sm57_amp_take_001_clean_di.wav \
  --target recordings/sm57_amp_take_001_amp_mic_target.wav \
  --model profiles/sm57_amp_mlx_spectrum.npz \
  --output outputs/sm57_amp_mlx_spectrum.wav \
  --comparison-output outputs/sm57_amp_mic_then_mlx_spectrum.wav \
  --fft-size 4096 \
  --hop-size 1024 \
  --epochs 90
```

Apply that full-spectrum model to another DI:

```bash
python3 tone_capture_engine.py apply-mlx-spectrum \
  --input recordings/new_clean_di.wav \
  --model profiles/sm57_amp_mlx_spectrum.npz \
  --output outputs/new_clean_di_mlx_spectrum.wav
```

If the output still sounds like bright DI, use the direct neural amp model. The
default `detail` loss trains on short contiguous waveform chunks, so MLX compares
the DI against the SM57 target by waveform shape, pick attack, high-frequency
detail, and envelope behavior instead of only matching isolated sample levels:

```bash
python3 tone_capture_engine.py train-mlx-amp \
  --di recordings/sm57_amp_take_003_clean_di.wav \
  --target recordings/sm57_amp_take_003_amp_mic_target.wav \
  --model profiles/sm57_amp_mlx_amp_take_003_detail.npz \
  --output outputs/sm57_amp_mlx_amp_take_003_detail.wav \
  --comparison-output outputs/sm57_amp_mic_then_mlx_amp_take_003_detail.wav \
  --epochs 140 \
  --context-radius 480 \
  --hidden-dim 256 \
  --max-training-seconds 90 \
  --loss-mode detail \
  --detail-chunk-samples 2048 \
  --detail-chunks-per-epoch 128 \
  --transient-loss-weight 0.45 \
  --highfreq-loss-weight 0.35 \
  --envelope-loss-weight 0.12 \
  --cab-lowpass-hz 7800 \
  --cab-presence-db 3.5 \
  --cab-air-db 0.8
```

Once you have more than one take with the same amp, cab, mic position, and amp
settings, train across them together. This helps MLX learn the shared amp/cab/SM57
character instead of overfitting one guitar performance:

If you captured takes with `--dataset`, train directly from the dataset manifest:

```bash
scripts/mlx_train_performance.sh train-mlx-amp \
  --dataset datasets/6505_rhythm_sm57_tele.json \
  --profile-family "6505_rhythm_sm57_tele" \
  --exclude-take sm57_amp_take_013_tele_bridge_hot_rails_full_v10_t10 \
  --model profiles/sm57_amp_mlx_amp_tele_dataset.npz \
  --output outputs/sm57_amp_mlx_amp_tele_dataset.wav \
  --comparison-output outputs/sm57_amp_mic_then_mlx_amp_tele_dataset.wav \
  --per-take-output-dir outputs/per_take_validation_tele_dataset \
  --epochs 180 \
  --context-radius 480 \
  --hidden-dim 384 \
  --max-training-seconds 90 \
  --take-sampling balanced \
  --conditioning-mode none \
  --loss-mode detail
```

The trainer prints per-take validation after the render, so judge multi-guitar
models by the average/per-take scores instead of only the first comparison file.
`--conditioning-mode source-stats` adds DI-derived source descriptors so the
shared amp model can adapt to hotter/brighter/darker guitar inputs without using
manual guitar labels.

```bash
scripts/mlx_train_performance.sh train-mlx-amp \
  --di recordings/sm57_amp_take_003_clean_di.wav \
  --target recordings/sm57_amp_take_003_amp_mic_target.wav \
  --extra-pair recordings/sm57_amp_take_004_les_paul_clean_di.wav recordings/sm57_amp_take_004_les_paul_amp_mic_target.wav \
  --extra-pair recordings/sm57_amp_take_005_strandberg_nazgul_sentient_clean_di.wav recordings/sm57_amp_take_005_strandberg_nazgul_sentient_amp_mic_target.wav \
  --extra-pair recordings/sm57_amp_take_006_strandberg_pegasus_sentient_clean_di.wav recordings/sm57_amp_take_006_strandberg_pegasus_sentient_amp_mic_target.wav \
  --model profiles/sm57_amp_mlx_amp_multi_003_004_005_006_conditioned.npz \
  --output outputs/sm57_amp_mlx_amp_multi_003_004_005_006_conditioned.wav \
  --comparison-output outputs/sm57_amp_mic_then_mlx_amp_multi_003_004_005_006_conditioned.wav \
  --per-take-output-dir outputs/per_take_validation_003_004_005_006_conditioned \
  --epochs 140 \
  --context-radius 480 \
  --hidden-dim 256 \
  --max-training-seconds 90 \
  --take-sampling balanced \
  --conditioning-mode source-stats \
  --loss-mode detail \
  --detail-chunk-samples 2048 \
  --detail-chunks-per-epoch 128 \
  --transient-loss-weight 0.45 \
  --highfreq-loss-weight 0.35 \
  --envelope-loss-weight 0.12 \
  --cab-lowpass-hz 7800 \
  --cab-presence-db 3.5 \
  --cab-air-db 0.8
```

Apply that neural amp model to another DI:

```bash
python3 tone_capture_engine.py apply-mlx-amp \
  --input recordings/new_clean_di.wav \
  --model profiles/sm57_amp_mlx_amp.npz \
  --output outputs/new_clean_di_mlx_amp.wav
```

If the amp model is close but muffled, re-render the same model with more SM57
presence before retraining:

```bash
python3 tone_capture_engine.py apply-mlx-amp \
  --input recordings/sm57_amp_take_003_clean_di.wav \
  --model profiles/sm57_amp_mlx_amp_take_003.npz \
  --output outputs/sm57_amp_mlx_amp_take_003_crisp.wav \
  --comparison-target recordings/sm57_amp_take_003_amp_mic_target.wav \
  --comparison-output outputs/sm57_amp_mic_then_mlx_amp_take_003_crisp.wav \
  --cab-lowpass-hz 8200 \
  --cab-presence-db 4.0 \
  --cab-air-db 1.0
```

Apply that learned bridge to another DI:

```bash
python3 tone_capture_engine.py apply-mlx-bridge \
  --input recordings/new_clean_di.wav \
  --base-model profiles/sm57_amp_mic_bridge.json \
  --model profiles/sm57_amp_mlx_bridge.npz \
  --output outputs/new_clean_di_mlx_bridge.wav
```

Train the MLX layer after you already have a DSP profile:

```bash
python3 tone_capture_engine.py train-mlx \
  --di recordings/take_001_clean_di.wav \
  --target recordings/take_001_amp_mic_target.wav \
  --base-profile profiles/sm57_amp_capture.json \
  --model profiles/sm57_amp_capture_mlx_residual.npz \
  --enhanced-output outputs/sm57_amp_capture_mlx_match.wav
```

Apply the DSP profile plus the MLX residual model to a new DI:

```bash
python3 tone_capture_engine.py apply-mlx \
  --input recordings/new_clean_di.wav \
  --profile profiles/sm57_amp_capture.json \
  --model profiles/sm57_amp_capture_mlx_residual.npz \
  --output outputs/new_clean_di_mlx_profiled.wav
```

The MLX model learns the residual difference between the DSP-profiled output and
the SM57 amp/cab target. It does not replace the DSP engine; it enhances it.
The MLX package is not imported unless you run `train-mlx` or `apply-mlx`.

## Guarded Multi-Modeler Performance Rigs

The controlled workflow adopts public workflow ideas from HeadRush, Universal
Audio OX, Fractal Audio, BOSS AIRD, and Line 6 Helix without copying proprietary
algorithms or file formats. `rig-probe-record --capture-type` distinguishes
`amp-cab`, `amp-preamp`, and `pedal-only` returns and stores physical
`--clone-control` labels. `build-performance-rig` packages accepted local models
with a morph or parallel model graph, input gate, three-band tone controls,
named snapshots, capture input-impedance metadata, destination routing, measured
mic endpoints, or a full cabinet IR.

For `amp-preamp` captures, neutral-by-default runtime blocks can add a selected
cabinet-linked resonance approximation followed by oversampled, level-dependent
speaker compression and cone-cry resonance. A pair of accepted measured cabinet
variant profiles supports continuous mic-position morphing on an `amp-cab`
capture. These stages are explicit audition controls; they do not replace the
accepted-model requirement and are not presented as a physical reactive load.

`studio-frfr` and `headphones` retain the digital cabinet path. `amp-return` and
`power-amp-guitar-cab` bypass digital speaker/cabinet processing and require an
`amp-preamp` capture. `amp-input` requires `pedal-only`. An Amp & Cab model may
use measured relative mic variants, but it is rejected anywhere that would need
its recorded cabinet removed or would stack another full cabinet. See
`RIG_CAPTURE_COMMANDS.md` for complete commands, snapshot keys, and routing
safety.

For a genuinely modular amp and cabinet route, capture the same calibrated
multilevel probe twice: first from an approved post-preamp line output and then
from the complete speaker/cabinet/SM57 path. `build-separated-cabinet` verifies
the probe, sample rate, reamp trim, pedal, amp, and control settings before it
derives the separate measured cabinet/SM57 stage. Speaker output must never be
connected directly to an audio interface.

## How It Works

1. The DI and target recordings are aligned.
2. The script searches a compact dynamic asymmetric `tanh` model with
   level-dependent sag and compression.
3. For each nonlinear candidate, it estimates a cabinet/tone impulse response
   with regularized deconvolution.
4. The best profile is saved as JSON with nonlinear parameters, tone features,
   hardware metadata, and validation metrics.
5. Optionally, MLX trains either a residual layer or a direct neural amp model
   on Apple Silicon; the direct amp path uses detail loss to preserve attack,
   high-frequency texture, and compression behavior from the SM57 target.
6. The profile can be recalled and applied to another DI recording, with or
   without the MLX residual layer.

## Portfolio Positioning

This project fits audio DSP, music technology, guitar/bass processing,
profiling-style tone capture, creative tools, and Python signal-processing
portfolios.
