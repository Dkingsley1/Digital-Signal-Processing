# Controlled Rig Capture Commands

This is the Quad Cortex-style workflow in this project: a known multilevel test
signal is reamped through one fixed pedal/amp/cab/mic setup, the return is
latency-aligned without independent peak normalization, and a causal nonlinear
MLX model is accepted only if it beats the DI-gain-only guard.

In PyCharm, choose `06 Tone System - Rig Capture`. The guided runner exposes nine
actions and requires the exact phrase `PLAY PROBE` before audio output. Its final
actions build and apply a portable guarded multi-modeler performance rig around the
accepted local MLX captures.

## Safety And Routing

1. Interface line output 1 -> proper reamp box -> Maxon OD808 -> amplifier input.
2. Amplifier speaker output -> compatible speaker cabinet only.
3. SM57 -> interface microphone input 2.
4. Keep the amp, every knob, pedal, cabinet, mic position, preamp gain, and
   sample rate fixed for the entire capture.

Never connect an amplifier speaker output to an interface. The probe contains
sweeps and noise, so start at low monitoring volume and use hearing protection.

## 1. Generate The Probe

```bash
.venv/bin/python tone_capture_engine.py rig-probe-generate \
  --output rig_captures/probes/rig_probe_96k.wav \
  --manifest rig_captures/probes/rig_probe_96k_manifest.json \
  --sample-rate 96000 \
  --peak-dbfs -18
```

## 2. Check The Command Without Playback

```bash
.venv/bin/python tone_capture_engine.py rig-probe-record \
  --probe rig_captures/probes/rig_probe_96k.wav \
  --probe-manifest rig_captures/probes/rig_probe_96k_manifest.json \
  --capture-name peavey_6505_rhythm_maxon808_sm57 \
  --input-channels 2 \
  --target-channel 2 \
  --output-channels 2 \
  --output-channel 1 \
  --capture-type amp-cab \
  --input-impedance-kohm 1000 \
  --pedal "Maxon OD808: drive 0, tone 5, balance 10" \
  --amp "Peavey 6505 Mini Head" \
  --amp-settings "rhythm channel; document every knob before capture" \
  --cabinet "Egnater Tweaker 1x12 Celestion" \
  --mic "Shure SM57" \
  --mic-position "SM57 close, directly in front of speaker" \
  --clone-control "pre_gain=5" \
  --clone-control "low=5" \
  --clone-control "mid=5" \
  --clone-control "high=5" \
  --clone-control "post_gain=5" \
  --dry-run
```

## 3. Record The Fixed Rig

After the dry run is correct, repeat the same command with `--dry-run` removed
and `--confirm-reamp-routing` added. Use `--input-device` and `--output-device`
when the interface is not the macOS default. Adjust `--send-trim-db` only while
calibrating the reamp output; do not normalize the saved return afterward.

```bash
.venv/bin/python tone_capture_engine.py rig-probe-record \
  --probe rig_captures/probes/rig_probe_96k.wav \
  --probe-manifest rig_captures/probes/rig_probe_96k_manifest.json \
  --capture-name peavey_6505_rhythm_maxon808_sm57 \
  --input-channels 2 \
  --target-channel 2 \
  --output-channels 2 \
  --output-channel 1 \
  --send-trim-db 0 \
  --capture-type amp-cab \
  --input-impedance-kohm 1000 \
  --pedal "Maxon OD808: drive 0, tone 5, balance 10" \
  --amp "Peavey 6505 Mini Head" \
  --amp-settings "rhythm channel; document every knob before capture" \
  --cabinet "Egnater Tweaker 1x12 Celestion" \
  --mic "Shure SM57" \
  --mic-position "SM57 close, directly in front of speaker" \
  --clone-control "pre_gain=5" \
  --clone-control "low=5" \
  --clone-control "mid=5" \
  --clone-control "high=5" \
  --clone-control "post_gain=5" \
  --confirm-reamp-routing
```

The recorder saves the exact sent probe, the mic return, untouched multichannel
input, latency/polarity, stream errors, levels, physical input impedance, and the complete rig manifest.
Aim for a mic return near `-12 dBFS` peak with no clipped samples.

## 4. Train And Validate

```bash
.venv/bin/python tone_capture_engine.py train-rig-capture \
  --probe rig_captures/peavey_6505_rhythm_maxon808_sm57_probe_input.wav \
  --target rig_captures/peavey_6505_rhythm_maxon808_sm57_target_return.wav \
  --capture-manifest rig_captures/peavey_6505_rhythm_maxon808_sm57_rig_capture_manifest.json \
  --model profiles/peavey_6505_rhythm_maxon808_sm57_causal_rig_96k.npz \
  --output outputs/peavey_6505_rhythm_maxon808_sm57_validation_render.wav \
  --comparison-output outputs/peavey_6505_rhythm_maxon808_sm57_target_then_model_10s.wav \
  --render-sample-rate 96000 \
  --oversample-factor 2 \
  --levels 13 \
  --epochs 100
```

Training runs internally at 192 kHz for anti-aliasing, restores the best
checkpoint, clips unstable gradients, decays the learning rate, and stops early
when validation no longer improves. A failed validation is written as
`*.rejected.npz`; it cannot replace the requested production model unless
`--allow-failed-validation` is explicitly supplied.

## 5. Refine With Real Guitar Playing

Record a normal simultaneous DI and amp/mic take without changing anything in
the captured rig. Play hard chords, palm mutes, sustained notes, single-note
attacks, and guitar-volume cleanup. Keep playing through the final 20% because
that ending is reserved for validation.

```bash
.venv/bin/python tone_capture_engine.py refine-rig-capture \
  --model profiles/peavey_6505_rhythm_maxon808_sm57_causal_rig_96k.npz \
  --di recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_clean_di.wav \
  --target recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_amp_mic_target.wav \
  --output-model profiles/peavey_6505_rhythm_maxon808_sm57_causal_rig_96k_refined.npz \
  --output outputs/peavey_6505_rhythm_maxon808_sm57_refined_validation.wav \
  --comparison-output outputs/peavey_6505_rhythm_maxon808_sm57_target_base_refined_10s.wav \
  --output-sample-rate 96000 \
  --epochs 30
```

The refinement command automatically measures the level trim between the clean
DI input and the calibrated reamp send. It stores that trim in the refined model
so pickup and guitar-volume differences remain meaningful during later renders.
The comparison file plays reference amp, base model, then refined model. A
refinement that fails held-out metrics is saved as `*.rejected.npz` and cannot
replace either accepted model.

## 6. Build A Measured Cabinet/Microphone Variant

Run step 3 again under a new capture name after changing only the cabinet,
microphone, position, distance, or axis. Keep the probe, reamp level, pedal, amp
settings, and interface preamp gain unchanged. Then build a modular response:

```bash
.venv/bin/python tone_capture_engine.py build-cabinet-variant \
  --probe rig_captures/peavey_6505_rhythm_maxon808_sm57_probe_input.wav \
  --probe-manifest rig_captures/probes/rig_probe_96k_manifest.json \
  --reference-target rig_captures/peavey_6505_rhythm_maxon808_sm57_target_return.wav \
  --variant-target rig_captures/peavey_6505_rhythm_maxon808_sm57_sm57_off_axis_target_return.wav \
  --profile profiles/cabinet_variants/egnater_sm57_off_axis.npz \
  --comparison-output outputs/egnater_sm57_off_axis_reference_recorded_synthesized.wav \
  --name "Egnater SM57 off-axis" \
  --reference-cabinet "Egnater Tweaker 1x12 Celestion" \
  --reference-microphone "Shure SM57" \
  --reference-mic-position "cap edge, close, on-axis" \
  --variant-cabinet "Egnater Tweaker 1x12 Celestion" \
  --variant-microphone "Shure SM57" \
  --variant-mic-position "cone edge, 4 inches" \
  --variant-mic-axis off-axis
```

This Fender Tone Master-inspired block analyzes the complete low-level sweep,
creates a minimum-phase correction impulse, and tests it against the unseen
probe ending. The comparison order is reference cabinet/mic, recorded variant,
then synthesized variant. Failed variants are saved as `*.rejected.npz`.

## 7. Render A Two-Mic Virtual Studio

First build at least two accepted measured cabinet/mic variants with step 6.
Then the standalone renderer can move continuously between those measured
endpoints and add phase, stereo, room, and speaker controls:

```bash
.venv/bin/python tone_capture_engine.py apply-virtual-studio \
  --input outputs/les_paul_d_standard_peavey_6505_causal_rig.wav \
  --output outputs/les_paul_d_standard_peavey_6505_virtual_studio.wav \
  --mic-a profiles/cabinet_variants/egnater_sm57_cap_edge.npz \
  --mic-b profiles/cabinet_variants/egnater_sm57_cone_edge.npz \
  --mic-morph 0.35 \
  --mic-a-pan -0.15 \
  --mic-b-pan 0.15 \
  --variphi-ms 0.18 \
  --room-preset tight \
  --distance 0.25 \
  --room-mix 0.20 \
  --speaker-overload 0.10 \
  --low-cut-hz 60 \
  --high-cut-hz 12000
```

`--mic-morph 0` is mic A, `1` is mic B, and values between them use an
equal-power blend. `--variphi-ms` delays or advances mic B by up to 10 ms for
phase interaction. The room presets are deterministic early-reflection render
effects, not measured Two notes DynIR data. Distance scales their contribution.
All cabinet/mic profiles must pass the measured-variant validation guard.

The virtual studio deliberately does not add a second power amp. The controlled
rig model already captured the Peavey power amp, cabinet, and microphone.

## 8. Apply To A Clean DI

```bash
.venv/bin/python tone_capture_engine.py apply-rig-capture \
  --input recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_clean_di.wav \
  --model profiles/peavey_6505_rhythm_maxon808_sm57_causal_rig_96k_refined.npz \
  --output outputs/les_paul_d_standard_peavey_6505_causal_rig.wav \
  --cabinet-variant profiles/cabinet_variants/egnater_sm57_off_axis.npz \
  --cabinet-mix 1 \
  --cabinet-low-cut-hz 60 \
  --cabinet-high-cut-hz 12000 \
  --virtual-mic-b profiles/cabinet_variants/egnater_sm57_cone_edge.npz \
  --virtual-mic-morph 0.35 \
  --virtual-mic-a-pan -0.15 \
  --virtual-mic-b-pan 0.15 \
  --virtual-variphi-ms 0.18 \
  --virtual-room-preset tight \
  --virtual-distance 0.25 \
  --virtual-room-mix 0.20 \
  --virtual-speaker-overload 0.10 \
  --output-sample-rate 96000
```

`apply-rig-capture` preserves the DI level presented to the model. Use
`--input-trim-db` only as an additional deliberate offset when a new DI was
recorded through a different gain path. Refined models automatically apply their
stored input calibration and never silently peak-normalize the DI.
Omit `--cabinet-variant` to hear the original captured cabinet and microphone.
`--cabinet-mix` continuously moves between that reference response and the
measured variant; the cut filters are optional cabinet-block controls.
Omit all `--virtual-*` options to preserve the original mono application path.

## 9. Build And Apply A Portable Performance Rig

This independent layer packages accepted models and practical runtime controls
into one JSON preset. It combines non-proprietary workflow ideas from HeadRush,
Universal Audio OX, Fractal Audio, BOSS AIRD, and Line 6 Helix. It is not compatible
with any manufacturer's preset format. Normalization defaults to off so pickup and
guitar-volume dynamics remain meaningful.

```bash
.venv/bin/python tone_capture_engine.py build-performance-rig \
  --preset profiles/performance_rigs/peavey_6505_rhythm_maxon808_sm57.json \
  --name "Peavey 6505 Rhythm Maxon 808 SM57" \
  --model profiles/peavey_6505_rhythm_maxon808_sm57_causal_rig_96k_refined.npz \
  --secondary-model profiles/peavey_6505_lead_maxon808_sm57_causal_rig_96k_refined.npz \
  --model-path-mode parallel \
  --model-morph 0.30 \
  --input-impedance-kohm 1000 \
  --gate-threshold-dbfs -70 \
  --bass-db 0 \
  --middle-db 0 \
  --treble-db 0 \
  --cabinet-ir cabinet_ir/egnater_tweaker_1x12_sm57.wav \
  --cabinet-ir-samples 2048 \
  --speaker-impedance-curve closed-back-1x12 \
  --dynamic-speaker-drive 0.18 \
  --cone-cry 0.05 \
  --cabinet-low-cut-hz 60 \
  --cabinet-high-cut-hz 12000 \
  --destination studio-frfr \
  --snapshots-json examples/performance_rig_snapshots.json \
  --normalize off \
  --output-sample-rate 96000
```

`--model-path-mode morph` makes a constant-level transition between two accepted
captures. `parallel` runs both accepted model paths and combines them with a
normalized equal-power balance. Both models must use the same capture type and
remain independently validated.

For an `amp-preamp` capture, `--cabinet-ir cabinet.wav` loads a full cabinet IR;
choose `--cabinet-ir-samples 2048` for full quality or `1024` for a lighter
cabinet block. The optional impedance curve, dynamic speaker drive, and cone-cry
stages run before that IR. They are refused after an `amp-cab` model so two speaker
responses cannot be stacked accidentally. These are controlled approximations,
not a physical reactive load.

For continuous measured microphone movement on an `amp-cab` capture, omit the
full IR and provide two accepted relative profiles:

```bash
--cabinet-variant profiles/cabinet_variants/egnater_sm57_cap_edge.npz \
--cabinet-variant-b profiles/cabinet_variants/egnater_sm57_cone_edge.npz \
--mic-position-morph 0.35
```

The input impedance value records the interface's real guitar-input setting.
Software cannot change pickup loading after A/D conversion, so application can
reject a DI made through substantially different hardware impedance.

```bash
.venv/bin/python tone_capture_engine.py apply-performance-rig \
  --input recordings/sm57_amp_take_030_les_paul_custom_axcess_bridge_d_standard_v10_t10_maxon808_clean_di.wav \
  --preset profiles/performance_rigs/peavey_6505_rhythm_maxon808_sm57.json \
  --output outputs/les_paul_d_standard_peavey_6505_performance_rig.wav \
  --snapshot Rhythm \
  --source-input-impedance-kohm 1000
```

Snapshots can change model balance, path levels, gate/trim, tone, speaker drive,
cone cry, measured mic position, output trim, and destination. Use
`--destination amp-return` or `--destination power-amp-guitar-cab` only with an
`amp-preamp` capture; those routes bypass the digital speaker and cabinet stages.
An `amp-cab` capture cannot use those destinations because its recorded cabinet
cannot be removed. A `pedal-only` capture is restricted to `amp-input`.

The applied order is input impedance check, gate and calibrated trim, accepted
causal model graph, tone controls, guarded speaker dynamics/resonance, measured
mic endpoints or cabinet IR, destination routing, cuts, output trim, optional
normalization, and final limiter.

## Direct Amp And Cabinet Separation

The software records a high-resolution cabinet-response diagnostic and measured
bass-resonance summary, but speaker-output Direct Amp capture is intentionally
disabled. The Livewire passive instrument DI must not be treated as a
speaker-level DI or load. Direct/merged-style capture should only be added after
a speaker-level-rated DI or load box with a speaker thru connection is available.
