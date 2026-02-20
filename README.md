# Digital-Signal-Processing

Real-time and offline audio DSP utilities focused on high-sample-rate monitoring and calibration workflows.

## Included Scripts

- `96khz_audio_lock_limiter.py`
  - Real-time stereo widening + limiter chain.
- `96khz_phantom_center_limiter.py`
  - Real-time mid/side focus + limiter chain.
- `audient_24bit_validation.py`
  - Generates a 96kHz / 24-bit pink-noise calibration WAV.
- `hardware_check_audient_jbl.py`
  - Generates a 96kHz calibration WAV for quick hardware checks.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run any script directly:

```bash
python 96khz_audio_lock_limiter.py
```

## Notes

- Real-time scripts currently use hardcoded device names:
  - Input: `BlackHole 2ch`
  - Output: `Audient iD14`
- If your device names differ, edit `INPUT_ID` / `OUTPUT_ID` in the script.

## Safety

- Start with low output volume.
- Confirm routing before enabling real-time processing.
- Use limiters and conservative gain staging to avoid clipping.
