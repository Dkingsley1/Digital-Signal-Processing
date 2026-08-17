#!/usr/bin/env python3
"""
Guitar/Bass Amp Tone Capture Engine

This compact portfolio prototype captures a reusable tone profile from a clean
DI recording and a processed amp/cab target recording. The saved JSON profile can
then be applied to another guitar or bass DI file.

It is inspired by tone capture/profiling workflows, but it is intentionally a
small DSP prototype rather than a commercial amp-modeling product.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from math import gcd
from pathlib import Path

from rig_identity import rig_fingerprint, rig_identity_from_manifest

try:
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import correlate, fftconvolve, lfilter, resample_poly
except ImportError as exc:
    raise SystemExit(
        "Missing core tone-capture DSP libraries.\n"
        "Install the live system dependencies in your PyCharm interpreter with:\n"
        "  python3 -m pip install -r requirements-live.txt"
    ) from exc


PROFILE_VERSION = "1.1"
MLX_MODEL_VERSION = "mlx_residual_mlp_1.0"
MLX_SPECTRAL_MODEL_VERSION = "mlx_full_spectrum_bridge_1.0"
MLX_AMP_MODEL_VERSION = "mlx_windowed_neural_amp_1.1"
MLX_AMP_LEGACY_MODEL_VERSIONS = {"mlx_windowed_neural_amp_1.0"}
AMP_TONE_ANCHOR_FFT_SIZE = 8192
AMP_TONE_ANCHOR_SMOOTHING_BINS = 61
AMP_TONE_ANCHOR_MAX_GAIN_DB = 18.0
AMP_TONE_GUARD_MIN_IMPROVEMENT_DB = 1.25
AMP_TONE_GUARD_MIN_MOVEMENT_DB = 0.75
AMP_DYNAMICS_GUARD_MAX_CREST_RATIO = 1.28
AMP_DYNAMICS_GUARD_MAX_PEAK_OVER_RMS_DB_DELTA = 2.0
AMP_QUALITY_MIN_WEIGHT = 0.12
AMP_QUALITY_WEAK_SPECTRAL_ERROR_DB = 12.0
AMP_QUALITY_EXCLUDE_SPECTRAL_ERROR_DB = 9.5
AMP_QUALITY_DI_LIKE_CORRELATION = 0.35
AMP_QUALITY_PARTIAL_DI_LIKE_CORRELATION = 0.20
AMP_QUALITY_EXACT_MATCH_MIN_WEIGHT = 0.35
AMP_HELDOUT_MAX_MEAN_SPECTRAL_ERROR_DB = 8.0
AMP_HELDOUT_MIN_MEAN_CORRELATION = 0.12
AMP_HELDOUT_MIN_PASS_RATE = 0.80
AMP_HELDOUT_MAX_PAIR_REGRESSION_DB = 1.0
SOURCE_MATCH_EXACT_WEIGHT = 0.72
SOURCE_MATCH_MIN_TOP_WEIGHT = 0.52
SOURCE_MATCH_SEGMENT_EXACT_WEIGHT = 0.86
AMP_TONE_SEGMENT_SECONDS = 10.0
AMP_TONE_SEGMENT_HOP_SECONDS = 5.0
HAMMERSTEIN_LAYER_TOP_N = 3
HAMMERSTEIN_LAYER_MIN_WEIGHT = 0.055
HAMMERSTEIN_LAYER_MAX_SECONDS = 16.0
HAMMERSTEIN_LAYER_MIX = 0.42
DEFAULT_INFERRED_IR_MIX = 0.0
TONE_FILTER_FIR_MS = 24.0
DATASET_TAKE_PATH_KEYS = (
    "clean_di_wav",
    "amp_mic_target_wav",
    "hardware_manifest",
    "profile_json",
    "reconstructed_wav",
)
TONE_RESEARCH_MOUNT = Path("/Volumes/ToneCaptureResearch")
TONE_RESEARCH_WORKING_CAP_BYTES = 5 * 1024**3
SYSTEM_ON_WORKSPACE_DIRS = (
    Path("recordings"),
    Path("rig_captures"),
    Path("profiles"),
    Path("outputs"),
    Path("logs/live_scope"),
)
SYSTEM_ON_SCOPE_DEFAULTS = {
    "duration_s": 1.0,
    "block_ms": 4.0,
    "window_ms": 120.0,
    "refresh_ms": 8,
    "fft_size": 4096,
    "display_points": 1200,
    "min_freq": 20.0,
    "max_freq": 12000.0,
    "min_db": -110.0,
    "max_db": 0.0,
    "diff_min_db": -36.0,
    "diff_max_db": 36.0,
    "spectrum_smoothing_bins": 7,
    "tone_diff_smoothing_bins": 71,
    "smoothing_attack": 0.92,
    "smoothing_release": 0.16,
    "pickup_frequency_sensitivity": True,
    "pickup_view_boost": 2.2,
    "pickup_view_release": 0.42,
    "frequency_eye_attack": 0.34,
    "frequency_eye_release": 0.18,
    "pickup_view_fast_delta_db": 0.9,
    "pickup_view_baseline_alpha": 0.004,
    "pickup_view_max_delta_db": 14.0,
    "output_change_delta_db": 0.45,
    "output_hot_delta_db": 1.20,
    "output_baseline_alpha": 0.025,
    "output_hold_alpha": 0.0025,
    "pickup_switch_score_threshold": 0.75,
    "pickup_switch_hold_ms": 3200.0,
    "pickup_switch_baseline_alpha": 0.10,
    "pickup_switch_hold_alpha": 0.012,
    "pickup_signal_floor_dbfs": -62.0,
    "pickup_activity_margin_db": 5.0,
    "pickup_activity_peak_margin_db": 8.0,
    "pickup_activity_hold_ms": 1500.0,
    "live_pickup_reference_enabled": True,
    "live_pickup_reference_dir": Path("recordings"),
    "live_pickup_reference_seconds": 14.0,
    "live_pickup_reference_amp_weight": 2.75,
    "live_pickup_reference_margin": 0.18,
    "live_pickup_reference_max_distance": 3.2,
    "amplitude_range": 1.0,
    "clip_guard": 0.95,
    "metrics_window_ms": 36.0,
    "source_analysis_ms": 420.0,
    "analysis_fft_frames": 6,
    "feature_log_interval_ms": 120.0,
    "width": 1500,
    "height": 950,
    "responsive": True,
    "log_frequency": True,
    "opengl": False,
    "antialias": False,
}
RECORDING_CLEAN_DI_SUFFIX = "_clean_di.wav"
RECORDING_AMP_TARGET_SUFFIX = "_amp_mic_target.wav"
RECORDING_HARDWARE_MANIFEST_SUFFIX = "_hardware_manifest.json"
LIVE_PICKUP_FEATURE_NAMES = (
    "centroid_hz",
    "rolloff_hz",
    "low_pct",
    "body_pct",
    "mid_pct",
    "upper_pct",
    "bite_pct",
    "air_pct",
    "resonant_hz",
    "body_to_bite",
    "rms_dbfs",
    "p999_dbfs",
)
LIVE_PICKUP_FEATURE_SCALES = np.array(
    [1800.0, 3200.0, 10.0, 10.0, 10.0, 8.0, 8.0, 6.0, 2200.0, 5.0, 12.0, 10.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CaptureConfig:
    instrument: str = "guitar"
    profile_name: str = "captured_tone"
    ir_ms: float = 32.0
    regularization: float = 0.002
    search_bias: bool = True


@dataclass(frozen=True)
class AudioInterfaceConfig:
    device: str | int | None = None
    sample_rate: int = 44100
    duration_s: float = 20.0
    input_channels: int = 2
    di_channel: int = 1
    target_channel: int = 2


@dataclass(frozen=True)
class DIBoxConfig:
    name: str = "Passive DI box"
    box_type: str = "passive"
    pad_db: float = 0.0
    ground_lift: bool = False
    phantom_power_to_di: bool = False
    thru_to_amp: bool = True
    mic_name: str = "Shure SM57"
    amp_name: str = "Guitar/bass amplifier"
    cabinet_name: str = "Speaker cabinet"
    notes: str = ""


@dataclass(frozen=True)
class TakeMetadata:
    profile_family: str = ""
    guitar: str = ""
    tuning: str = ""
    pickup: str = ""
    pickup_mode: str = ""
    guitar_volume: str = ""
    guitar_tone: str = ""
    amp_channel: str = ""
    boost_pedal: str = ""
    mic_position: str = ""
    performance: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CaptureResult:
    profile: dict
    reconstructed: np.ndarray
    aligned_di: np.ndarray
    aligned_target: np.ndarray
    match_rmse: float
    match_correlation: float
    spectral_error_db: float


def read_wav_float(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV file and return mono float audio in [-1, 1]."""
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    if sf is not None:
        data, sample_rate = sf.read(path, always_2d=False, dtype="float64")
        if getattr(data, "ndim", 1) == 2:
            data = np.mean(data, axis=1)
        audio = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return int(sample_rate), np.clip(audio, -1.0, 1.0)

    sample_rate, data = wavfile.read(path)

    if data.ndim == 2:
        data = np.mean(data, axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_abs = float(np.iinfo(data.dtype).max)
        audio = data.astype(np.float64) / max_abs
    else:
        audio = data.astype(np.float64)

    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return int(sample_rate), np.clip(audio, -1.0, 1.0)


def write_wav_float(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    """Write float audio. Uses float WAV output when soundfile is installed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = soft_limiter(audio)
    audio = np.clip(audio, -1.0, 1.0)

    try:
        import soundfile as sf
    except ImportError:
        sf = None

    if sf is not None:
        sf.write(path, audio.astype(np.float32), sample_rate, subtype="FLOAT")
        return

    wavfile.write(path, sample_rate, np.int16(audio * 32767))


def resample_if_needed(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    try:
        import soxr
    except ImportError:
        soxr = None

    if soxr is not None:
        return np.asarray(soxr.resample(audio, source_rate, target_rate, quality="VHQ"), dtype=np.float64)

    factor = gcd(source_rate, target_rate)
    up = target_rate // factor
    down = source_rate // factor
    return resample_poly(audio, up, down).astype(np.float64)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def normalize_peak(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    current_peak = float(np.max(np.abs(audio)) + 1e-12)
    return audio * min(1.0, peak / current_peak)


def normalize_for_audition(audio: np.ndarray, peak: float = 0.86) -> np.ndarray:
    """Scale rendered comparison files to a useful listening level."""
    audio = remove_dc(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0))
    current_peak = float(np.max(np.abs(audio)) + 1e-12)
    return audio * (peak / current_peak)


def db_to_linear(db_value: float) -> float:
    return float(10.0 ** (db_value / 20.0))


def remove_dc(audio: np.ndarray) -> np.ndarray:
    return audio - float(np.mean(audio))


def soft_limiter(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    """Prevent clipping while keeping the level practical for listening."""
    audio = remove_dc(audio)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    if peak <= ceiling:
        return audio

    scaled = audio / peak
    limited = np.tanh(2.2 * scaled) / np.tanh(2.2)
    return limited * ceiling


def align_pair(di_audio: np.ndarray, target_audio: np.ndarray, max_lag_s: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Align target audio to DI using FFT cross-correlation and polarity detection."""
    compare_len = min(len(di_audio), len(target_audio), int(sample_rate * 4.0))
    if compare_len < sample_rate // 2:
        min_len = min(len(di_audio), len(target_audio))
        return di_audio[:min_len], target_audio[:min_len], 0, 1

    di_ref = remove_dc(di_audio[:compare_len])
    target_ref = remove_dc(target_audio[:compare_len])
    max_lag = int(round(max_lag_s * sample_rate))

    corr = correlate(target_ref, di_ref, mode="full", method="fft")
    center = len(di_ref) - 1
    start = max(0, center - max_lag)
    end = min(len(corr), center + max_lag + 1)
    search = corr[start:end]
    peak_index = int(np.argmax(np.abs(search)))
    peak_value = float(search[peak_index])
    polarity = -1 if peak_value < 0.0 else 1
    lag = int(peak_index + start - center)

    if lag > 0:
        target_aligned = target_audio[lag:]
        di_aligned = di_audio[: len(target_aligned)]
    elif lag < 0:
        di_aligned = di_audio[-lag:]
        target_aligned = target_audio[: len(di_aligned)]
    else:
        min_len = min(len(di_audio), len(target_audio))
        di_aligned = di_audio[:min_len]
        target_aligned = target_audio[:min_len]

    min_len = min(len(di_aligned), len(target_aligned))
    target_aligned = target_aligned[:min_len] * polarity
    return di_aligned[:min_len], target_aligned, lag, polarity


def align_pair_fractional(
    di_audio: np.ndarray,
    target_audio: np.ndarray,
    max_lag_s: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Align a paired capture with integer, polarity, and fractional-sample correction."""
    compare_len = min(len(di_audio), len(target_audio), int(sample_rate * 8.0))
    if compare_len < sample_rate // 2:
        di, target, lag, polarity = align_pair(di_audio, target_audio, max_lag_s, sample_rate)
        return di, target, float(lag), polarity

    source = remove_dc(di_audio[:compare_len])
    returned = remove_dc(target_audio[:compare_len])
    corr = correlate(returned, source, mode="full", method="fft")
    center = len(source) - 1
    radius = int(round(max_lag_s * sample_rate))
    lo = max(1, center - radius)
    hi = min(len(corr) - 1, center + radius + 1)
    magnitude = np.abs(corr[lo:hi])
    local_index = int(np.argmax(magnitude))
    peak_index = lo + local_index
    lag_int = int(peak_index - center)
    polarity = -1 if float(corr[peak_index]) < 0.0 else 1

    fraction = 0.0
    if 0 < local_index < len(magnitude) - 1:
        left = float(magnitude[local_index - 1])
        middle = float(magnitude[local_index])
        right = float(magnitude[local_index + 1])
        denominator = left - (2.0 * middle) + right
        if abs(denominator) > 1e-20:
            fraction = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    di_aligned, target_aligned, _, _ = align_pair(di_audio, target_audio, max_lag_s, sample_rate)
    if abs(fraction) > 1e-4 and len(target_aligned) > 8:
        positions = np.arange(len(target_aligned), dtype=np.float64) + fraction
        target_aligned = np.interp(
            positions,
            np.arange(len(target_aligned), dtype=np.float64),
            target_aligned,
            left=0.0,
            right=0.0,
        )
    return di_aligned, target_aligned, float(lag_int + fraction), polarity


def align_reference_to_source_timeline(
    source_audio: np.ndarray,
    reference_audio: np.ndarray,
    sample_rate: int,
    max_lag_s: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """Place a captured mic reference on the zero-latency DI/model timeline."""
    _, _, lag, polarity = align_pair_fractional(
        source_audio,
        reference_audio,
        max_lag_s=max_lag_s,
        sample_rate=sample_rate,
    )
    positions = np.arange(len(source_audio), dtype=np.float64) + float(lag)
    aligned = np.interp(
        positions,
        np.arange(len(reference_audio), dtype=np.float64),
        reference_audio,
        left=0.0,
        right=0.0,
    )
    aligned *= int(polarity)
    return aligned.astype(np.float64), {
        "lag_samples": float(lag),
        "lag_ms": float(1000.0 * lag / max(1, sample_rate)),
        "polarity": int(polarity),
    }


def envelope_follower(
    audio: np.ndarray,
    sample_rate: int,
    attack_ms: float = 4.0,
    release_ms: float = 85.0,
) -> np.ndarray:
    """Track playing intensity for level-dependent nonlinear behavior."""
    attack_coeff = np.exp(-1.0 / max(1.0, sample_rate * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0))
    magnitude = np.abs(audio)
    envelope = np.zeros_like(audio, dtype=np.float64)

    for index, sample in enumerate(magnitude):
        previous = envelope[index - 1] if index else 0.0
        coeff = attack_coeff if sample > previous else release_coeff
        envelope[index] = coeff * previous + (1.0 - coeff) * sample

    return envelope


def apply_saturation(audio: np.ndarray, drive: float, bias: float) -> np.ndarray:
    """Apply a static asymmetric tanh saturation model."""
    driven = drive * (audio + bias)
    centered = np.tanh(driven) - np.tanh(drive * bias)
    normalizer = float(np.max(np.abs(centered)) + 1e-12)
    return centered / normalizer


def apply_dynamic_nonlinearity(
    audio: np.ndarray,
    sample_rate: int,
    input_gain: float,
    drive: float,
    bias: float,
    sag: float,
    compression: float,
) -> np.ndarray:
    """
    Apply level-dependent amp-style nonlinearity.

    The envelope follower makes hard playing reduce effective drive and output
    gain slightly, approximating sag/compression in a compact prototype form.
    """
    gained = audio * input_gain
    envelope = envelope_follower(gained, sample_rate)
    normalized_env = envelope / (float(np.percentile(envelope, 99.0)) + 1e-12)
    normalized_env = np.clip(normalized_env, 0.0, 2.5)

    sag_gain = 1.0 / (1.0 + sag * normalized_env)
    dynamic_drive = drive / (1.0 + 0.65 * sag * normalized_env)
    driven = dynamic_drive * ((gained * sag_gain) + bias)
    centered = np.tanh(driven) - np.tanh(dynamic_drive * bias)

    compression_gain = 1.0 / (1.0 + compression * normalized_env)
    processed = centered * compression_gain
    normalizer = float(np.max(np.abs(processed)) + 1e-12)
    return processed / normalizer


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def estimate_ir_regularized(
    source: np.ndarray,
    target: np.ndarray,
    ir_samples: int,
    regularization: float,
) -> np.ndarray:
    """
    Estimate a compact linear tone/cabinet impulse response by deconvolution.

    H = Y * conj(X) / (|X|^2 + lambda)
    """
    nfft = next_power_of_two(len(source) + ir_samples - 1)
    source_fft = np.fft.rfft(source, nfft)
    target_fft = np.fft.rfft(target, nfft)
    denom = np.abs(source_fft) ** 2
    reg = regularization * float(np.max(denom) + 1e-12)
    transfer = target_fft * np.conj(source_fft) / (denom + reg)
    impulse = np.fft.irfft(transfer, nfft)[:ir_samples]

    window = np.hanning(ir_samples * 2)[ir_samples:]
    impulse *= window
    impulse = remove_dc(impulse)
    impulse /= float(np.max(np.abs(impulse)) + 1e-12)
    impulse *= 0.75
    return impulse.astype(np.float64)


def estimate_gain(source: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.dot(source, source) + 1e-12)
    return float(np.dot(target, source) / denom)


def spectral_magnitude(audio: np.ndarray, sample_rate: int, bins: int = 512) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(audio), sample_rate * 8)
    if n < 1024:
        n = len(audio)

    window = np.hanning(n)
    spectrum = np.fft.rfft(remove_dc(audio[:n]) * window)
    magnitude_db = 20.0 * np.log10(np.abs(spectrum) + 1e-9)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    if len(freqs) <= bins:
        return freqs, magnitude_db

    target_freqs = np.geomspace(max(20.0, freqs[1]), sample_rate / 2.0, bins)
    interp = np.interp(target_freqs, freqs, magnitude_db)
    return target_freqs, interp


def spectral_error_db(reference: np.ndarray, candidate: np.ndarray, sample_rate: int) -> float:
    _, ref_mag = spectral_magnitude(reference, sample_rate)
    _, cand_mag = spectral_magnitude(candidate, sample_rate)
    ref_mag -= float(np.mean(ref_mag))
    cand_mag -= float(np.mean(cand_mag))
    return float(np.sqrt(np.mean((ref_mag - cand_mag) ** 2)))


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = remove_dc(a)
    b = remove_dc(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def band_energy(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> float:
    n = min(len(audio), sample_rate * 8)
    spectrum = np.fft.rfft(remove_dc(audio[:n]) * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(spectrum[mask]) ** 2))


def tone_features(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    low = band_energy(audio, sample_rate, 60.0, 250.0)
    mid = band_energy(audio, sample_rate, 250.0, 1800.0)
    high = band_energy(audio, sample_rate, 1800.0, 8000.0)
    total = low + mid + high + 1e-12

    freqs, mag = spectral_magnitude(audio, sample_rate)
    linear_mag = 10.0 ** (mag / 20.0)
    centroid = float(np.sum(freqs * linear_mag) / (np.sum(linear_mag) + 1e-12))

    return {
        "rms": rms(audio),
        "peak": float(np.max(np.abs(audio)) + 1e-12),
        "crest_factor": float((np.max(np.abs(audio)) + 1e-12) / (rms(audio) + 1e-12)),
        "spectral_centroid_hz": centroid,
        "low_energy_ratio": low / total,
        "mid_energy_ratio": mid / total,
        "high_energy_ratio": high / total,
    }


def detailed_tone_profile(audio: np.ndarray, sample_rate: int) -> dict:
    bands = [
        (50.0, 90.0),
        (90.0, 160.0),
        (160.0, 250.0),
        (250.0, 400.0),
        (400.0, 650.0),
        (650.0, 1000.0),
        (1000.0, 1500.0),
        (1500.0, 2200.0),
        (2200.0, 3200.0),
        (3200.0, 4800.0),
        (4800.0, 7000.0),
        (7000.0, 10000.0),
    ]
    clean = remove_dc(np.asarray(audio, dtype=np.float64))
    n = min(len(clean), sample_rate * 8)
    if n < 2:
        energies = np.zeros(len(bands), dtype=np.float64)
    else:
        spectrum = np.fft.rfft(clean[:n] * np.hanning(n))
        power = np.square(np.abs(spectrum))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        energies = np.asarray(
            [
                float(np.mean(power[(freqs >= low_hz) & (freqs <= high_hz)]))
                if np.any((freqs >= low_hz) & (freqs <= high_hz))
                else 0.0
                for low_hz, high_hz in bands
            ],
            dtype=np.float64,
        )
    energies /= float(np.sum(energies) + 1e-12)
    return {
        "bands_hz": [[float(low), float(high)] for low, high in bands],
        "energy_ratios": [float(value) for value in energies],
    }


def match_detailed_tone_profile(
    audio: np.ndarray,
    sample_rate: int,
    profile: dict,
    iterations: int = 2,
    max_step_db: float = 3.0,
) -> tuple[np.ndarray, dict]:
    bands = [(float(item[0]), float(item[1])) for item in profile.get("bands_hz", [])]
    target = np.asarray(profile.get("energy_ratios", []), dtype=np.float64)
    if not bands or len(bands) != len(target) or float(np.sum(target)) <= 1e-12:
        return audio, {"active": False}
    target /= float(np.sum(target) + 1e-12)
    output = remove_dc(np.asarray(audio, dtype=np.float64))

    def current_energy(value: np.ndarray) -> np.ndarray:
        energies = np.asarray(
            [band_energy(value, sample_rate, low_hz, high_hz) for low_hz, high_hz in bands],
            dtype=np.float64,
        )
        return energies / float(np.sum(energies) + 1e-12)

    before = current_energy(output)
    for _ in range(max(1, int(iterations))):
        current = current_energy(output)
        gain_db = np.clip(
            10.0 * np.log10((target + 1e-12) / (current + 1e-12)),
            -float(max_step_db),
            float(max_step_db),
        )
        centers = np.asarray([np.sqrt(low_hz * high_hz) for low_hz, high_hz in bands], dtype=np.float64)
        frequencies = np.concatenate([[20.0], centers, [sample_rate / 2.0]])
        gains = np.concatenate([[gain_db[0]], gain_db, [gain_db[-1]]])
        output = apply_frequency_gain_curve(output, sample_rate, frequencies, gains)
    after = current_energy(output)
    error_db = 10.0 * np.log10((after + 1e-12) / (target + 1e-12))
    return output, {
        "active": True,
        "before": [float(value) for value in before],
        "target": [float(value) for value in target],
        "after": [float(value) for value in after],
        "max_error_db": float(np.max(np.abs(error_db))),
    }


def causal_rms_envelope(audio: np.ndarray, sample_rate: int, smoothing_ms: float = 18.0) -> np.ndarray:
    clean = remove_dc(np.nan_to_num(np.asarray(audio, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    coefficient = float(np.exp(-1.0 / max(1.0, sample_rate * smoothing_ms / 1000.0)))
    power = lfilter([1.0 - coefficient], [1.0, -coefficient], np.square(clean))
    return np.sqrt(np.maximum(power, 1e-12))


def local_envelope_profile(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    envelope = causal_rms_envelope(audio, sample_rate)
    envelope_db = 20.0 * np.log10(envelope + 1e-12)
    active_floor = max(float(np.max(envelope_db) - 36.0), float(np.percentile(envelope_db, 5.0)))
    active = envelope_db[envelope_db >= active_floor]
    if len(active) < 32:
        active = envelope_db
    p10, p50, p90 = [float(np.percentile(active, value)) for value in (10.0, 50.0, 90.0)]
    return {
        "p10_db": p10,
        "p50_db": p50,
        "p90_db": p90,
        "spread_db": float(max(0.0, p90 - p10)),
    }


def reshape_local_envelope(
    audio: np.ndarray,
    sample_rate: int,
    target_spread_db: float,
    strength: float = 0.85,
    max_cut_db: float = 5.0,
    max_boost_db: float = 8.0,
) -> tuple[np.ndarray, dict]:
    output = remove_dc(np.asarray(audio, dtype=np.float64))
    before = local_envelope_profile(output, sample_rate)
    current_spread = max(0.25, float(before["spread_db"]))
    target_spread = float(np.clip(target_spread_db, 1.0, current_spread))
    ratio = float(np.clip(target_spread / current_spread, 0.08, 1.0))
    if ratio >= 0.985:
        return output, {"active": False, "before": before, "after": before, "target_spread_db": target_spread}

    envelope = causal_rms_envelope(output, sample_rate)
    envelope_db = 20.0 * np.log10(envelope + 1e-12)
    center = float(before["p50_db"])
    desired_db = center + ((envelope_db - center) * ratio)
    gain_db = np.clip(desired_db - envelope_db, -float(max_cut_db), float(max_boost_db))
    silence_floor = center - 24.0
    gain_db = np.where(envelope_db >= silence_floor, gain_db, np.minimum(gain_db, 0.0))
    gain_db *= float(np.clip(strength, 0.0, 1.0))
    gain_smoothing = float(np.exp(-1.0 / max(1.0, sample_rate * 12.0 / 1000.0)))
    gain_db = lfilter([1.0 - gain_smoothing], [1.0, -gain_smoothing], gain_db)
    original_rms = rms(output)
    output *= 10.0 ** (gain_db / 20.0)
    output *= original_rms / (rms(output) + 1e-12)
    after = local_envelope_profile(output, sample_rate)
    return remove_dc(output), {
        "active": True,
        "before": before,
        "after": after,
        "target_spread_db": target_spread,
        "ratio": ratio,
    }


def match_reference_local_envelope(
    audio: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    strength: float = 1.0,
    iterations: int = 3,
) -> tuple[np.ndarray, dict]:
    compare_len = min(len(audio), len(reference))
    if compare_len < 1024:
        return audio, {"active": False}
    output = remove_dc(np.asarray(audio, dtype=np.float64))
    before = local_envelope_profile(output[:compare_len], sample_rate)
    target_envelope = causal_rms_envelope(reference[:compare_len], sample_rate)
    matched = output.copy()
    for _ in range(max(1, int(iterations))):
        source_envelope = causal_rms_envelope(matched[:compare_len], sample_rate)
        gain_db = 20.0 * np.log10((target_envelope + 1e-12) / (source_envelope + 1e-12))
        gain_db = np.clip(gain_db, -7.0, 9.0) * float(np.clip(strength, 0.0, 1.0))
        gain_smoothing = float(np.exp(-1.0 / max(1.0, sample_rate * 4.0 / 1000.0)))
        gain_db = lfilter([1.0 - gain_smoothing], [1.0, -gain_smoothing], gain_db)
        matched[:compare_len] *= 10.0 ** (gain_db / 20.0)
        matched = match_reference_level(matched, reference, mode="rms")
    return remove_dc(matched), {
        "active": True,
        "target": local_envelope_profile(reference[:compare_len], sample_rate),
        "before": before,
        "after": local_envelope_profile(matched[:compare_len], sample_rate),
    }


def match_reference_tone_bands(
    audio: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    iterations: int = 3,
    max_step_db: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Converge low, mid, and high energy ratios to a close-mic reference."""
    output = remove_dc(np.asarray(audio, dtype=np.float64))
    target = tone_features(reference, sample_rate)
    before = tone_features(output, sample_rate)
    keys = ("low_energy_ratio", "mid_energy_ratio", "high_energy_ratio")
    detail_bands = [
        (50.0, 90.0),
        (90.0, 160.0),
        (160.0, 250.0),
        (250.0, 400.0),
        (400.0, 650.0),
        (650.0, 1000.0),
        (1000.0, 1500.0),
        (1500.0, 2200.0),
        (2200.0, 3200.0),
        (3200.0, 4800.0),
        (4800.0, 7000.0),
        (7000.0, 10000.0),
    ]

    def normalized_detail_energy(value: np.ndarray) -> np.ndarray:
        energies = np.asarray(
            [band_energy(value, sample_rate, low_hz, high_hz) for low_hz, high_hz in detail_bands],
            dtype=np.float64,
        )
        return energies / float(np.sum(energies) + 1e-12)

    target_detail = normalized_detail_energy(reference)
    for _ in range(max(1, int(iterations))):
        current_detail = normalized_detail_energy(output)
        detail_gain_db = np.clip(
            10.0 * np.log10((target_detail + 1e-12) / (current_detail + 1e-12)),
            -float(max_step_db),
            float(max_step_db),
        )
        centers = np.asarray([np.sqrt(low_hz * high_hz) for low_hz, high_hz in detail_bands], dtype=np.float64)
        frequencies = np.concatenate([[20.0], centers, [sample_rate / 2.0]])
        gains = np.concatenate([[detail_gain_db[0]], detail_gain_db, [detail_gain_db[-1]]])
        output = apply_frequency_gain_curve(output, sample_rate, frequencies, gains)

    after = tone_features(output, sample_rate)
    after_detail = normalized_detail_energy(output)
    detail_error_db = 10.0 * np.log10((after_detail + 1e-12) / (target_detail + 1e-12))
    return output, {
        "active": True,
        "iterations": int(max(1, iterations)),
        "target": {key: float(target[key]) for key in keys},
        "before": {key: float(before[key]) for key in keys},
        "after": {key: float(after[key]) for key in keys},
        "detail_bands_hz": [[float(low), float(high)] for low, high in detail_bands],
        "max_detail_error_db": float(np.max(np.abs(detail_error_db))),
    }


def installed_audio_stack() -> dict[str, str | bool]:
    modules = {
        "soundfile": "soundfile",
        "soxr": "soxr",
        "librosa": "librosa",
        "pyloudnorm": "pyloudnorm",
        "pedalboard": "pedalboard",
        "noisereduce": "noisereduce",
        "scikit-learn": "sklearn",
    }
    status: dict[str, str | bool] = {}
    for label, module_name in modules.items():
        try:
            module = __import__(module_name)
        except ImportError:
            status[label] = False
            continue
        status[label] = str(getattr(module, "__version__", "installed"))
    return status


def integrated_loudness_lufs(audio: np.ndarray, sample_rate: int) -> float | None:
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None

    clean = remove_dc(np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    if len(clean) < int(sample_rate * 0.45) or rms(clean) < 1e-9:
        return None
    try:
        return float(pyln.Meter(sample_rate).integrated_loudness(clean))
    except Exception:
        return None


def advanced_audio_descriptors(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """
    Optional descriptors from the advanced audio stack.

    These feed quality diagnostics only. They do not modify training audio.
    """
    descriptors: dict[str, float] = {}
    clean = remove_dc(np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    if len(clean) < 1024:
        return descriptors

    loudness = integrated_loudness_lufs(clean, sample_rate)
    if loudness is not None and np.isfinite(loudness):
        descriptors["integrated_loudness_lufs"] = float(loudness)

    try:
        import librosa
    except ImportError:
        return descriptors

    max_samples = min(len(clean), int(round(sample_rate * 12.0)))
    y = clean[:max_samples]
    if len(y) < 1024:
        return descriptors

    frame_length = min(4096, max(1024, 2 ** int(np.floor(np.log2(len(y))))))
    hop_length = max(256, frame_length // 4)
    try:
        descriptors["zero_crossing_rate"] = float(
            np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length))
        )
        descriptors["spectral_flatness"] = float(
            np.mean(librosa.feature.spectral_flatness(y=y, n_fft=frame_length, hop_length=hop_length))
        )
        descriptors["spectral_rolloff_85_hz"] = float(
            np.mean(
                librosa.feature.spectral_rolloff(
                    y=y,
                    sr=sample_rate,
                    n_fft=frame_length,
                    hop_length=hop_length,
                    roll_percent=0.85,
                )
            )
        )
        descriptors["spectral_bandwidth_hz"] = float(
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sample_rate, n_fft=frame_length, hop_length=hop_length))
        )
    except Exception:
        return descriptors

    return {key: value for key, value in descriptors.items() if np.isfinite(value)}


def reduce_noise_preview_audio(
    audio: np.ndarray,
    sample_rate: int,
    stationary: bool = False,
    prop_decrease: float = 0.55,
) -> np.ndarray:
    """Optional noisereduce preview. This is never applied to training implicitly."""
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise SystemExit(
            "The denoise preview command needs noisereduce.\n"
            "Install it with:\n"
            "  .venv/bin/python -m pip install -r requirements-audio-advanced.txt"
        ) from exc

    clean = remove_dc(np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    reduced = nr.reduce_noise(
        y=clean,
        sr=sample_rate,
        stationary=stationary,
        prop_decrease=float(np.clip(prop_decrease, 0.0, 1.0)),
    )
    return remove_dc(np.asarray(reduced, dtype=np.float64))


def pedalboard_preview_audio(audio: np.ndarray, sample_rate: int, preset: str) -> np.ndarray:
    """Explicit pedalboard preview. This is never applied to training implicitly."""
    try:
        from pedalboard import Compressor, Distortion, Gain, HighpassFilter, LowpassFilter, Pedalboard
    except ImportError as exc:
        raise SystemExit(
            "The pedalboard preview command needs pedalboard.\n"
            "Install it with:\n"
            "  .venv/bin/python -m pip install -r requirements-audio-advanced.txt"
        ) from exc

    if preset == "tighten":
        board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=70.0),
                LowpassFilter(cutoff_frequency_hz=8500.0),
                Compressor(threshold_db=-18.0, ratio=2.5, attack_ms=8.0, release_ms=80.0),
            ]
        )
    elif preset == "drive-check":
        board = Pedalboard(
            [
                HighpassFilter(cutoff_frequency_hz=80.0),
                Distortion(drive_db=12.0),
                LowpassFilter(cutoff_frequency_hz=7200.0),
                Gain(gain_db=-6.0),
            ]
        )
    else:
        board = Pedalboard([Gain(gain_db=0.0)])

    clean = remove_dc(np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    processed = board(clean, sample_rate)
    return remove_dc(np.asarray(processed, dtype=np.float64).reshape(-1))


def apply_profile_to_audio(audio: np.ndarray, sample_rate: int, profile: dict) -> np.ndarray:
    profile_rate = int(profile["sample_rate_hz"])
    audio = resample_if_needed(audio, sample_rate, profile_rate)

    nonlinear = profile["nonlinear"]
    cabinet = profile["cabinet"]
    driven = apply_dynamic_nonlinearity(
        audio,
        profile_rate,
        input_gain=float(nonlinear["input_gain"]),
        drive=float(nonlinear["drive"]),
        bias=float(nonlinear["bias"]),
        sag=float(nonlinear.get("sag", 0.0)),
        compression=float(nonlinear.get("compression", 0.0)),
    )
    ir = np.array(cabinet["impulse_response"], dtype=np.float64)
    filtered = fftconvolve(driven, ir, mode="full")[: len(driven)]
    output = filtered * float(profile["output_gain"])
    return normalize_peak(soft_limiter(output), peak=0.92)


def average_power_spectrum(audio: np.ndarray, fft_size: int = 8192, hop_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an average power spectrum without needing phase alignment."""
    if hop_size is None:
        hop_size = fft_size // 4

    audio = remove_dc(audio.astype(np.float64))
    if len(audio) < fft_size:
        audio = np.pad(audio, (0, fft_size - len(audio)))

    window = np.hanning(fft_size)
    power = np.zeros(fft_size // 2 + 1, dtype=np.float64)
    frame_count = 0

    for start in range(0, len(audio) - fft_size + 1, hop_size):
        frame = audio[start : start + fft_size] * window
        spectrum = np.fft.rfft(frame)
        power += np.abs(spectrum) ** 2
        frame_count += 1

    if frame_count == 0:
        frame = np.pad(audio, (0, max(0, fft_size - len(audio))))[:fft_size] * window
        spectrum = np.fft.rfft(frame)
        power += np.abs(spectrum) ** 2
        frame_count = 1

    return np.fft.rfftfreq(fft_size), power / frame_count


def representative_playing_excerpt(audio: np.ndarray, sample_rate: int, max_seconds: float = 14.0) -> np.ndarray:
    max_samples = max(1024, int(round(sample_rate * max_seconds)))
    if len(audio) <= max_samples:
        return audio

    block = max(1024, int(round(sample_rate * 0.35)))
    starts = list(range(0, max(1, len(audio) - block + 1), block))
    if not starts:
        return audio[:max_samples]

    scores = []
    for start in starts:
        chunk = audio[start : start + block]
        scores.append(float(np.sqrt(np.mean(np.square(remove_dc(chunk))) + 1e-12)))

    keep_count = max(1, min(len(starts), max_samples // block))
    loudest = np.argsort(np.asarray(scores))[-keep_count:]
    selected_starts = sorted(starts[int(index)] for index in loudest)
    excerpt = np.concatenate([audio[start : start + block] for start in selected_starts])
    return excerpt[:max_samples]


def live_pickup_metric_vector(
    centroid_hz: float,
    rolloff_hz: float,
    low_pct: float,
    body_pct: float,
    mid_pct: float,
    upper_pct: float,
    bite_pct: float,
    air_pct: float,
    resonant_hz: float,
    body_to_bite: float,
    rms_value_dbfs: float,
    p999_value_dbfs: float,
) -> np.ndarray:
    return np.array(
        [
            float(centroid_hz),
            float(rolloff_hz),
            float(low_pct),
            float(body_pct),
            float(mid_pct),
            float(upper_pct),
            float(bite_pct),
            float(air_pct),
            float(resonant_hz),
            float(np.clip(body_to_bite, 0.0, 20.0)),
            float(rms_value_dbfs),
            float(p999_value_dbfs),
        ],
        dtype=np.float64,
    )


def live_pickup_feature_vector_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    fft_size: int = 4096,
    max_seconds: float = 14.0,
) -> np.ndarray:
    excerpt = representative_playing_excerpt(audio, sample_rate, max_seconds=max_seconds)
    freqs_norm, power = average_power_spectrum(excerpt, fft_size=fft_size)
    freqs_hz = freqs_norm * sample_rate

    def band_pct(low_hz: float, high_hz: float) -> float:
        total_mask = (freqs_hz >= 80.0) & (freqs_hz <= 10000.0)
        band_mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
        total = float(np.sum(power[total_mask]) + 1e-12)
        return float(100.0 * np.sum(power[band_mask]) / total)

    usable_mask = (freqs_hz >= 80.0) & (freqs_hz <= 10000.0)
    usable_power = power[usable_mask]
    usable_freqs = freqs_hz[usable_mask]
    centroid_hz = 0.0
    rolloff_hz = 0.0
    if len(usable_power):
        total = float(np.sum(usable_power) + 1e-12)
        centroid_hz = float(np.sum(usable_freqs * usable_power) / total)
        cumulative = np.cumsum(usable_power)
        rolloff_index = min(int(np.searchsorted(cumulative, cumulative[-1] * 0.85)), len(usable_freqs) - 1)
        rolloff_hz = float(usable_freqs[rolloff_index])

    resonant_mask = (freqs_hz >= 700.0) & (freqs_hz <= 6500.0)
    resonant_hz = 0.0
    if np.any(resonant_mask):
        resonant_freqs = freqs_hz[resonant_mask]
        resonant_power = power[resonant_mask]
        resonant_hz = float(resonant_freqs[int(np.argmax(resonant_power))])

    low_pct = band_pct(80.0, 250.0)
    body_pct = band_pct(250.0, 750.0)
    mid_pct = band_pct(750.0, 2000.0)
    upper_pct = band_pct(1500.0, 2500.0)
    bite_pct = band_pct(2500.0, 5000.0)
    air_pct = band_pct(5000.0, 9000.0)
    rms_value_dbfs = 20.0 * np.log10(rms(excerpt) + 1e-12)
    p999_value_dbfs = 20.0 * np.log10(float(np.percentile(np.abs(excerpt), 99.9)) + 1e-12)
    return live_pickup_metric_vector(
        centroid_hz=centroid_hz,
        rolloff_hz=rolloff_hz,
        low_pct=low_pct,
        body_pct=body_pct,
        mid_pct=mid_pct,
        upper_pct=upper_pct,
        bite_pct=bite_pct,
        air_pct=air_pct,
        resonant_hz=resonant_hz,
        body_to_bite=body_pct / max(bite_pct, 1e-6),
        rms_value_dbfs=rms_value_dbfs,
        p999_value_dbfs=p999_value_dbfs,
    )


def smooth_gain_db(gain_db: np.ndarray, smoothing_bins: int) -> np.ndarray:
    smoothing_bins = max(3, int(smoothing_bins))
    if smoothing_bins % 2 == 0:
        smoothing_bins += 1

    if len(gain_db) < smoothing_bins:
        return gain_db

    kernel = np.hanning(smoothing_bins)
    kernel /= float(np.sum(kernel) + 1e-12)
    padded = np.pad(gain_db, smoothing_bins // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def apply_frequency_gain_curve(
    audio: np.ndarray,
    sample_rate: int,
    curve_freqs_hz: np.ndarray,
    gain_db: np.ndarray,
    phase_mode: str = "minimum",
) -> np.ndarray:
    """Apply a smooth tone curve without adding non-causal pre-ringing by default."""
    n = len(audio)
    if n < 2:
        return np.asarray(audio, dtype=np.float64)

    curve_freqs_hz = np.asarray(curve_freqs_hz, dtype=np.float64)
    gain_db = np.asarray(gain_db, dtype=np.float64)
    if len(curve_freqs_hz) < 2 or len(curve_freqs_hz) != len(gain_db):
        return np.asarray(audio, dtype=np.float64)

    if phase_mode not in {"minimum", "zero"}:
        raise ValueError("phase_mode must be minimum or zero")

    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    interpolated_gain_db = np.interp(freqs, curve_freqs_hz, gain_db, left=gain_db[0], right=gain_db[-1])
    gain = 10.0 ** (interpolated_gain_db / 20.0)
    clean = remove_dc(np.asarray(audio, dtype=np.float64))
    if phase_mode == "zero":
        spectrum = np.fft.rfft(clean)
        return np.fft.irfft(spectrum * gain, n=n).astype(np.float64)

    fir_samples = max(256, int(round(float(sample_rate) * TONE_FILTER_FIR_MS / 1000.0)))
    fir_samples = min(fir_samples, max(256, n))
    design_size = next_power_of_two(max(4096, fir_samples * 4))
    design_freqs = np.fft.rfftfreq(design_size, d=1.0 / sample_rate)
    design_gain_db = np.interp(
        design_freqs,
        curve_freqs_hz,
        gain_db,
        left=gain_db[0],
        right=gain_db[-1],
    )
    log_magnitude = np.log(np.maximum(10.0 ** (design_gain_db / 20.0), 1e-7))
    cepstrum = np.fft.irfft(log_magnitude, n=design_size)
    minimum_cepstrum = np.zeros_like(cepstrum)
    minimum_cepstrum[0] = cepstrum[0]
    midpoint = design_size // 2
    minimum_cepstrum[1:midpoint] = 2.0 * cepstrum[1:midpoint]
    minimum_cepstrum[midpoint] = cepstrum[midpoint]
    minimum_spectrum = np.exp(np.fft.rfft(minimum_cepstrum))
    impulse = np.fft.irfft(minimum_spectrum, n=design_size)[:fir_samples]
    taper_samples = max(16, fir_samples // 4)
    impulse[-taper_samples:] *= np.hanning(taper_samples * 2)[taper_samples:]
    if not np.all(np.isfinite(impulse)) or float(np.max(np.abs(impulse))) <= 1e-12:
        spectrum = np.fft.rfft(clean)
        return np.fft.irfft(spectrum * gain, n=n).astype(np.float64)
    return fftconvolve(clean, impulse, mode="full")[:n].astype(np.float64)


def match_reference_level(audio: np.ndarray, reference: np.ndarray, mode: str = "off") -> np.ndarray:
    if mode == "off":
        return audio
    if mode not in {"peak", "rms"}:
        raise SystemExit("--target-level-match must be off, peak, or rms.")

    compare_len = min(len(audio), len(reference))
    if compare_len < 1:
        return audio

    audio_ref = audio[:compare_len]
    target_ref = reference[:compare_len]
    if mode == "peak":
        source_level = float(np.max(np.abs(audio_ref)) + 1e-12)
        target_level = float(np.max(np.abs(target_ref)) + 1e-12)
    else:
        source_level = rms(audio_ref)
        target_level = rms(target_ref)

    gain = float(np.clip(target_level / source_level, 0.02, 8.0))
    return audio * gain


def apply_reference_spectral_imprint(
    audio: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    strength: float = 0.0,
    smoothing_bins: int = 91,
    max_gain_db: float = 10.0,
    fft_size: int = 8192,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.5))
    if strength <= 0.0:
        return audio

    compare_len = min(len(audio), len(reference))
    if compare_len < 1024:
        return audio

    fft_size = max(1024, int(fft_size))
    freqs_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    _, source_power = average_power_spectrum(audio[:compare_len], fft_size=fft_size)
    _, reference_power = average_power_spectrum(reference[:compare_len], fft_size=fft_size)

    gain_db = 10.0 * np.log10((reference_power + 1e-12) / (source_power + 1e-12))
    gain_db = smooth_gain_db(gain_db, smoothing_bins=smoothing_bins)
    gain_db = np.clip(gain_db, -float(max_gain_db), float(max_gain_db)) * strength
    return apply_frequency_gain_curve(audio, sample_rate, freqs_hz, gain_db)


def reference_spectral_gain_curve(
    source: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    fft_size: int = AMP_TONE_ANCHOR_FFT_SIZE,
    smoothing_bins: int = AMP_TONE_ANCHOR_SMOOTHING_BINS,
    max_gain_db: float = AMP_TONE_ANCHOR_MAX_GAIN_DB,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the amp/mic spectral curve that turns source tone into reference tone."""
    compare_len = min(len(source), len(reference))
    fft_size = max(1024, int(fft_size))
    freqs_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    if compare_len < 1024:
        return freqs_hz, np.zeros_like(freqs_hz, dtype=np.float64)

    _, source_power = average_power_spectrum(source[:compare_len], fft_size=fft_size)
    _, reference_power = average_power_spectrum(reference[:compare_len], fft_size=fft_size)
    gain_db = 10.0 * np.log10((reference_power + 1e-12) / (source_power + 1e-12))
    gain_db = smooth_gain_db(gain_db, smoothing_bins=smoothing_bins)
    gain_db = np.clip(gain_db, -float(max_gain_db), float(max_gain_db))
    return freqs_hz, gain_db.astype(np.float64)


def amp_dynamic_fingerprint(audio: np.ndarray) -> dict[str, float]:
    clean = remove_dc(np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    rms_value = rms(clean)
    peak_value = float(np.max(np.abs(clean)) + 1e-12)
    diff = np.diff(clean, prepend=clean[0])
    return {
        "rms": float(rms_value),
        "peak": float(peak_value),
        "crest_factor": float(peak_value / (rms_value + 1e-12)),
        "peak_over_rms_db": float(20.0 * np.log10((peak_value + 1e-12) / (rms_value + 1e-12))),
        "transient_rms_ratio": float(rms(diff) / (rms_value + 1e-12)),
    }


def median_amp_dynamic_fingerprint(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = ["crest_factor", "peak_over_rms_db", "transient_rms_ratio"]
    return {key: float(np.median([item[key] for item in items])) for key in keys}


def weighted_amp_dynamic_fingerprint(items: list[dict[str, float]], weights: np.ndarray) -> dict[str, float]:
    if not items:
        return {}
    keys = ["crest_factor", "peak_over_rms_db", "transient_rms_ratio"]
    return {
        key: float(np.sum(weights * np.array([item[key] for item in items], dtype=np.float64)))
        for key in keys
    }


def apply_peak_rms_amp_compression(
    audio: np.ndarray,
    target_crest_factor: float,
    ratio: float | None = None,
    threshold_rms: float | None = None,
) -> np.ndarray:
    """
    Pull DI-like peaks toward a mic/cab crest factor.

    The MLX waveform can still leave raw DI peaks poking through. This stage is
    intentionally target-driven: it only clamps when the render's peak/RMS shape
    is hotter than the learned mic behavior.
    """
    target_crest_factor = float(np.clip(target_crest_factor, 2.5, 18.0))
    output = remove_dc(np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    original_output = output.copy()
    original_rms = rms(output)
    if original_rms <= 1e-9:
        return output

    for _ in range(4):
        current_crest = crest_factor(output)
        if current_crest <= target_crest_factor * 1.04:
            break

        amount = current_crest / max(target_crest_factor, 1e-6)
        compressor_ratio = float(ratio if ratio is not None else np.clip(1.0 + (amount - 1.0) * 3.2, 2.0, 8.0))
        threshold = float(
            threshold_rms
            if threshold_rms is not None
            else np.clip(target_crest_factor * 0.42, 1.2, 3.0)
        )
        normalized_envelope = np.abs(output) / (rms(output) + 1e-12)
        over_threshold = normalized_envelope > threshold
        if not np.any(over_threshold):
            break

        gain = np.ones_like(output)
        gain[over_threshold] = (
            threshold + ((normalized_envelope[over_threshold] - threshold) / compressor_ratio)
        ) / (normalized_envelope[over_threshold] + 1e-12)
        output *= gain
        output *= original_rms / (rms(output) + 1e-12)

    original_crest = crest_factor(original_output)
    output_crest = crest_factor(output)
    if output_crest < target_crest_factor * 0.98 and original_crest > target_crest_factor:
        low = 0.0
        high = 1.0
        best = output
        best_error = abs(output_crest - target_crest_factor)
        for _ in range(18):
            mid = (low + high) / 2.0
            candidate = ((1.0 - mid) * original_output) + (mid * output)
            candidate *= original_rms / (rms(candidate) + 1e-12)
            candidate_crest = crest_factor(candidate)
            error = abs(candidate_crest - target_crest_factor)
            if error < best_error:
                best = candidate
                best_error = error
            if candidate_crest > target_crest_factor:
                low = mid
            else:
                high = mid
        output = best

    return remove_dc(output)


def source_match_feature_vector(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return amp_source_conditioning_features(audio, sample_rate).astype(np.float64)


def source_match_tokens(value: str | Path | None) -> set[str]:
    text = str(value or "").lower()
    for char in "/\\._-":
        text = text.replace(char, " ")
    stop = {
        "",
        "wav",
        "clean",
        "di",
        "amp",
        "mic",
        "target",
        "sm57",
        "take",
        "recordings",
    }
    return {token for token in text.split() if token not in stop and not token.isdigit()}


def source_hint_boosts(bank: list[dict], source_hint_path: Path | None) -> np.ndarray:
    boosts = np.ones(len(bank), dtype=np.float64)
    if source_hint_path is None:
        return boosts

    hint_path = Path(source_hint_path)
    hint_name = hint_path.name
    hint_stem = hint_path.stem
    hint_tokens = source_match_tokens(hint_stem)
    for index, item in enumerate(bank):
        item_di = Path(str(item.get("di", "")))
        item_text = " ".join(
            [
                str(item.get("take_name", "")),
                str(item.get("di", "")),
                str(item.get("target", "")),
            ]
        )
        item_tokens = source_match_tokens(item_text)
        if item_di.name == hint_name or item_di.stem == hint_stem:
            boosts[index] *= 40.0
            continue
        if hint_tokens and item_tokens:
            overlap = len(hint_tokens & item_tokens) / max(1, len(hint_tokens | item_tokens))
            boosts[index] *= 1.0 + (8.0 * overlap)
            if {"drop", "d"} <= hint_tokens and {"drop", "d"} <= item_tokens:
                boosts[index] *= 2.0
            if "drop" in hint_tokens and "drop" in item_tokens:
                boosts[index] *= 1.45
            if "maxon808" in hint_stem.lower() and "maxon808" in item_text.lower():
                boosts[index] *= 1.8
            if "les" in hint_tokens and "paul" in hint_tokens and "les" in item_tokens and "paul" in item_tokens:
                boosts[index] *= 1.8
    return boosts


def enforce_dominant_top_match(weights: np.ndarray, source_hint_path: Path | None, bank: list[dict]) -> np.ndarray:
    if len(weights) == 0:
        return weights

    top_index = int(np.argmax(weights))
    exact_index = None
    if source_hint_path is not None:
        hint_path = Path(source_hint_path)
        for index, item in enumerate(bank):
            item_di = Path(str(item.get("di", "")))
            quality_weight = float(item.get("quality_weight", 1.0))
            if (
                quality_weight >= AMP_QUALITY_MIN_WEIGHT
                and (item_di.name == hint_path.name or item_di.stem == hint_path.stem)
            ):
                exact_index = index
                break

    desired_top_weight = SOURCE_MATCH_EXACT_WEIGHT if exact_index is not None else SOURCE_MATCH_MIN_TOP_WEIGHT
    dominant_index = exact_index if exact_index is not None else top_index
    if weights[dominant_index] >= desired_top_weight:
        return weights

    remainder = np.array(weights, dtype=np.float64)
    current = float(remainder[dominant_index])
    remainder[dominant_index] = 0.0
    remainder_sum = float(np.sum(remainder))
    if remainder_sum <= 1e-12:
        weights[:] = 0.0
        weights[dominant_index] = 1.0
        return weights

    weights = remainder * ((1.0 - desired_top_weight) / remainder_sum)
    weights[dominant_index] = desired_top_weight
    if current <= 1e-12:
        weights /= float(np.sum(weights) + 1e-12)
    return weights


def source_matched_segment_profile(
    anchor: dict,
    source_di: np.ndarray,
    sample_rate: int,
    source_hint_path: Path | None = None,
    rig_fingerprint_value: str | None = None,
    mic_position: str | None = None,
) -> dict:
    """Select captured amp body and density from riff-sized training windows."""
    bank = list(anchor.get("segment_transfer_bank", []))
    if not bank:
        return {"enabled": False}

    feature_rows = np.asarray([item.get("source_features", []) for item in bank], dtype=np.float64)
    if feature_rows.ndim != 2 or feature_rows.shape[0] != len(bank) or feature_rows.shape[1] < 1:
        return {"enabled": False}
    query = source_match_feature_vector(source_di, sample_rate)
    if query.shape[0] != feature_rows.shape[1]:
        return {"enabled": False}

    quality_weights = np.asarray(
        [float(np.clip(item.get("quality_weight", 1.0), 0.0, 1.0)) for item in bank],
        dtype=np.float64,
    )
    active_quality = np.maximum(quality_weights, 1e-6)
    feature_mean = np.average(feature_rows, axis=0, weights=active_quality)
    feature_var = np.average(np.square(feature_rows - feature_mean.reshape(1, -1)), axis=0, weights=active_quality)
    feature_std = np.maximum(np.sqrt(feature_var), 1e-6)
    distances = np.sqrt(
        np.mean(
            np.square((feature_rows - feature_mean.reshape(1, -1)) / feature_std - (query - feature_mean) / feature_std),
            axis=1,
        )
    )
    finite = np.isfinite(distances)
    if not np.any(finite):
        distances = np.zeros(len(bank), dtype=np.float64)
    else:
        distances = np.where(finite, distances, float(np.max(distances[finite]) + 1.0))
    close_distance = float(np.percentile(distances, 12.0))
    temperature = max(0.08, close_distance * 0.45, 1e-6)
    weights = np.exp(-0.5 * np.square(distances / temperature))
    weights *= source_hint_boosts(bank, source_hint_path)
    weights *= quality_weights

    requested_rig = str(rig_fingerprint_value or "").strip()
    rig_matches = np.asarray(
        [bool(requested_rig and str(item.get("rig_fingerprint", "")) == requested_rig) for item in bank],
        dtype=bool,
    )
    if np.any(rig_matches):
        weights *= np.where(rig_matches, 1.0, 0.001)

    if str(mic_position or "").strip():
        requested_mic = mic_position_conditioning_features(mic_position)
        mic_rows = np.stack(
            [mic_position_conditioning_features(str(item.get("mic_position", ""))) for item in bank]
        )
        mic_distance = np.sqrt(np.mean(np.square(mic_rows - requested_mic.reshape(1, -1)), axis=1))
        weights *= np.exp(-2.5 * np.square(mic_distance))

    exact_matches = np.zeros(len(bank), dtype=bool)
    if source_hint_path is not None:
        hint_path = Path(source_hint_path)
        exact_matches = np.asarray(
            [
                Path(str(item.get("di", ""))).name == hint_path.name
                or Path(str(item.get("di", ""))).stem == hint_path.stem
                for item in bank
            ],
            dtype=bool,
        )
        exact_matches &= quality_weights >= AMP_QUALITY_MIN_WEIGHT
        if np.any(exact_matches):
            weights *= np.where(exact_matches, 1.0, 0.001)

    weights += 0.0002 * quality_weights * (np.where(rig_matches, 1.0, 0.001) if np.any(rig_matches) else 1.0)
    if float(np.sum(weights)) <= 1e-12:
        weights = quality_weights.copy()
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones(len(bank), dtype=np.float64)
    weights /= float(np.sum(weights) + 1e-12)

    if np.any(exact_matches):
        exact_indices = np.flatnonzero(exact_matches)
        dominant_index = int(exact_indices[np.argmin(distances[exact_indices])])
        desired_weight = SOURCE_MATCH_SEGMENT_EXACT_WEIGHT
    else:
        dominant_index = int(np.argmax(weights))
        desired_weight = SOURCE_MATCH_MIN_TOP_WEIGHT
    if weights[dominant_index] < desired_weight:
        remainder = weights.copy()
        remainder[dominant_index] = 0.0
        remainder_sum = float(np.sum(remainder))
        if remainder_sum <= 1e-12:
            weights[:] = 0.0
            weights[dominant_index] = 1.0
        else:
            weights = remainder * ((1.0 - desired_weight) / remainder_sum)
            weights[dominant_index] = desired_weight

    tone_profiles = [dict(item.get("target_detailed_tone_profile", {})) for item in bank]
    if not all(profile.get("bands_hz") and profile.get("energy_ratios") for profile in tone_profiles):
        return {"enabled": False}
    tone_rows = np.asarray([profile["energy_ratios"] for profile in tone_profiles], dtype=np.float64)
    blended_tone = np.sum(tone_rows * weights.reshape(-1, 1), axis=0)
    blended_tone /= float(np.sum(blended_tone) + 1e-12)

    envelope_profiles = [dict(item.get("target_local_envelope_profile", {})) for item in bank]
    envelope_keys = ("p10_db", "p50_db", "p90_db", "spread_db")
    envelope_profile = {}
    if all(all(key in profile for key in envelope_keys) for profile in envelope_profiles):
        envelope_profile = {
            key: float(np.sum(weights * np.asarray([profile[key] for profile in envelope_profiles])))
            for key in envelope_keys
        }
    dynamics = [dict(item.get("target_dynamic_fingerprint", {})) for item in bank]
    dynamic_profile = weighted_amp_dynamic_fingerprint(dynamics, weights) if all(dynamics) else {}
    top_indices = np.argsort(weights)[::-1][: min(5, len(bank))]
    return {
        "enabled": True,
        "target_detailed_tone_profile": {
            "bands_hz": tone_profiles[0]["bands_hz"],
            "energy_ratios": [float(value) for value in blended_tone],
        },
        "target_local_envelope_profile": envelope_profile,
        "target_dynamic_fingerprint": dynamic_profile,
        "top_segments": [
            {
                "take_name": str(bank[index].get("take_name", "")),
                "di": str(bank[index].get("di", "")),
                "start_seconds": float(bank[index].get("start_seconds", 0.0)),
                "duration_seconds": float(bank[index].get("duration_seconds", 0.0)),
                "weight": float(weights[index]),
                "distance": float(distances[index]),
            }
            for index in top_indices
        ],
    }


def source_matched_amp_transfer(
    anchor: dict,
    source_di: np.ndarray,
    sample_rate: int,
    source_hint_path: Path | None = None,
    rig_fingerprint_value: str | None = None,
    mic_position: str | None = None,
) -> dict:
    """
    Blend every recorded DI-to-amp transfer, favoring takes whose DI resembles
    the incoming DI. This keeps all recordings in the model while avoiding one
    smeared average curve for every pickup/tuning/guitar.
    """
    anchor_freqs = np.asarray(anchor.get("curve_freqs_hz", []), dtype=np.float64)
    default_gain_db = np.asarray(anchor.get("target_over_di_gain_db", []), dtype=np.float64)
    if len(anchor_freqs) < 2 or len(anchor_freqs) != len(default_gain_db):
        return {"enabled": False}

    bank = list(anchor.get("per_take_transfer_bank", []))
    if not bank:
        return {
            "enabled": True,
            "mode": "global_average",
            "curve_freqs_hz": anchor_freqs,
            "target_over_di_gain_db": default_gain_db,
            "target_dynamic_fingerprint": dict(anchor.get("target_dynamic_fingerprint", {})),
            "fft_size": int(anchor.get("fft_size", AMP_TONE_ANCHOR_FFT_SIZE)),
            "weights": [],
        }

    feature_rows = np.array([item.get("source_features", []) for item in bank], dtype=np.float64)
    gain_rows = np.array([item.get("target_over_di_gain_db", []) for item in bank], dtype=np.float64)
    if feature_rows.ndim != 2 or gain_rows.ndim != 2 or gain_rows.shape[1] != len(anchor_freqs):
        return {
            "enabled": True,
            "mode": "global_average",
            "curve_freqs_hz": anchor_freqs,
            "target_over_di_gain_db": default_gain_db,
            "target_dynamic_fingerprint": dict(anchor.get("target_dynamic_fingerprint", {})),
            "fft_size": int(anchor.get("fft_size", AMP_TONE_ANCHOR_FFT_SIZE)),
            "weights": [],
        }

    query = source_match_feature_vector(source_di, sample_rate)
    if query.shape[0] != feature_rows.shape[1]:
        return {
            "enabled": True,
            "mode": "global_average",
            "curve_freqs_hz": anchor_freqs,
            "target_over_di_gain_db": default_gain_db,
            "target_dynamic_fingerprint": dict(anchor.get("target_dynamic_fingerprint", {})),
            "fft_size": int(anchor.get("fft_size", AMP_TONE_ANCHOR_FFT_SIZE)),
            "weights": [],
        }

    feature_mean = np.asarray(anchor.get("source_feature_mean", []), dtype=np.float64)
    feature_std = np.asarray(anchor.get("source_feature_std", []), dtype=np.float64)
    if feature_mean.shape[0] != query.shape[0] or feature_std.shape[0] != query.shape[0]:
        feature_mean = np.mean(feature_rows, axis=0)
        feature_std = np.std(feature_rows, axis=0)
    feature_std = np.maximum(feature_std, 1e-6)

    query_norm = (query - feature_mean) / feature_std
    rows_norm = (feature_rows - feature_mean) / feature_std
    distances = np.sqrt(np.mean(np.square(rows_norm - query_norm.reshape(1, -1)), axis=1))
    finite = np.isfinite(distances)
    if not np.any(finite):
        distances = np.zeros(len(bank), dtype=np.float64)
        weights = np.ones(len(bank), dtype=np.float64)
    else:
        distances = np.where(finite, distances, float(np.max(distances[finite]) + 1.0))
        close_distance = float(np.percentile(distances, 20.0))
        temperature = max(0.12, close_distance * 0.55, 1e-6)
        weights = np.exp(-0.5 * np.square(distances / temperature))
        weights *= source_hint_boosts(bank, source_hint_path)
    quality_weights = np.asarray(
        [float(np.clip(item.get("quality_weight", 1.0), 0.0, 1.0)) for item in bank],
        dtype=np.float64,
    )
    weights *= quality_weights
    requested_rig = str(rig_fingerprint_value or "").strip()
    rig_matches = np.asarray(
        [bool(requested_rig and str(item.get("rig_fingerprint", "")) == requested_rig) for item in bank],
        dtype=bool,
    )
    if np.any(rig_matches):
        # The neural stage still learns from every accepted recording. Keep the
        # final amp/cab/mic tone local to the selected fixed rig so different
        # microphone positions cannot smear together into a room-like result.
        weights *= np.where(rig_matches, 1.0, 0.001)

    requested_mic = mic_position_conditioning_features(mic_position)
    mic_rows = []
    for item in bank:
        item_mic = str(item.get("mic_position", ""))
        mic_rows.append(mic_position_conditioning_features(item_mic))
    if str(mic_position or "").strip() and mic_rows:
        mic_matrix = np.stack(mic_rows)
        mic_distance = np.sqrt(np.mean(np.square(mic_matrix - requested_mic.reshape(1, -1)), axis=1))
        weights *= np.exp(-2.5 * np.square(mic_distance))

    # Tiny floor means every good recording contributes, without reviving excluded captures.
    floor_weights = quality_weights.copy()
    if np.any(rig_matches):
        floor_weights *= np.where(rig_matches, 1.0, 0.001)
    weights += 0.0015 * floor_weights
    if float(np.sum(weights)) <= 1e-12:
        weights = quality_weights.copy()
    if float(np.sum(weights)) <= 1e-12:
        weights = np.ones(len(bank), dtype=np.float64)
    weights /= float(np.sum(weights) + 1e-12)
    weights = enforce_dominant_top_match(weights, source_hint_path, bank)

    blended_gain_db = np.sum(gain_rows * weights.reshape(-1, 1), axis=0)
    blended_gain_db = smooth_gain_db(
        blended_gain_db,
        smoothing_bins=int(anchor.get("smoothing_bins", AMP_TONE_ANCHOR_SMOOTHING_BINS)),
    )
    max_gain_db = float(anchor.get("max_gain_db", AMP_TONE_ANCHOR_MAX_GAIN_DB))
    blended_gain_db = np.clip(blended_gain_db, -max_gain_db, max_gain_db)

    dynamics = [
        dict(item.get("target_dynamic_fingerprint", {}))
        for item in bank
        if item.get("target_dynamic_fingerprint")
    ]
    if len(dynamics) == len(bank):
        dynamic_target = weighted_amp_dynamic_fingerprint(dynamics, weights)
    else:
        dynamic_target = dict(anchor.get("target_dynamic_fingerprint", {}))

    envelope_profiles = [dict(item.get("target_local_envelope_profile", {})) for item in bank]
    envelope_keys = ("p10_db", "p50_db", "p90_db", "spread_db")
    if all(all(key in profile for key in envelope_keys) for profile in envelope_profiles):
        target_envelope_profile = {
            key: float(np.sum(weights * np.asarray([profile[key] for profile in envelope_profiles], dtype=np.float64)))
            for key in envelope_keys
        }
    else:
        target_envelope_profile = dict(anchor.get("target_local_envelope_profile", {}))

    tone_profiles = [dict(item.get("target_detailed_tone_profile", {})) for item in bank]
    if all(profile.get("bands_hz") and profile.get("energy_ratios") for profile in tone_profiles):
        tone_rows = np.asarray([profile["energy_ratios"] for profile in tone_profiles], dtype=np.float64)
        blended_tone = np.sum(tone_rows * weights.reshape(-1, 1), axis=0)
        blended_tone /= float(np.sum(blended_tone) + 1e-12)
        target_tone_profile = {
            "bands_hz": tone_profiles[0]["bands_hz"],
            "energy_ratios": [float(value) for value in blended_tone],
        }
    else:
        target_tone_profile = dict(anchor.get("target_detailed_tone_profile", {}))

    segment_match = source_matched_segment_profile(
        anchor,
        source_di,
        sample_rate,
        source_hint_path=source_hint_path,
        rig_fingerprint_value=rig_fingerprint_value,
        mic_position=mic_position,
    )
    if segment_match.get("enabled", False):
        target_tone_profile = dict(segment_match.get("target_detailed_tone_profile", target_tone_profile))
        if segment_match.get("target_local_envelope_profile"):
            target_envelope_profile = dict(segment_match["target_local_envelope_profile"])
        if segment_match.get("target_dynamic_fingerprint"):
            dynamic_target = dict(segment_match["target_dynamic_fingerprint"])

    top_indices = np.argsort(weights)[::-1][: min(5, len(bank))]
    return {
        "enabled": True,
        "mode": (
            "source_matched_all_recordings_segment_conditioned"
            if segment_match.get("enabled", False)
            else "source_matched_all_recordings"
        ),
        "rig_fingerprint": requested_rig or None,
        "curve_freqs_hz": anchor_freqs,
        "target_over_di_gain_db": blended_gain_db,
        "target_dynamic_fingerprint": dynamic_target,
        "target_local_envelope_profile": target_envelope_profile,
        "target_detailed_tone_profile": target_tone_profile,
        "fft_size": int(anchor.get("fft_size", AMP_TONE_ANCHOR_FFT_SIZE)),
        "weights": [float(value) for value in weights],
        "top_segments": list(segment_match.get("top_segments", [])),
        "top_matches": [
            {
                "take_name": str(bank[index].get("take_name", "")),
                "di": str(bank[index].get("di", "")),
                "target": str(bank[index].get("target", "")),
                "weight": float(weights[index]),
                "quality_weight": float(bank[index].get("quality_weight", 1.0)),
                "distance": float(distances[index]),
            }
            for index in top_indices
        ],
    }


def loudest_aligned_segment(
    di_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    max_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    max_samples = int(round(max_seconds * sample_rate))
    compare_len = min(len(di_audio), len(target_audio))
    di_audio = di_audio[:compare_len]
    target_audio = target_audio[:compare_len]
    if compare_len <= max_samples or max_samples < sample_rate:
        return di_audio, target_audio

    window = max_samples
    hop = max(sample_rate // 2, window // 8)
    best_start = 0
    best_rms = -1.0
    for start in range(0, compare_len - window + 1, hop):
        value = rms(di_audio[start : start + window])
        if value > best_rms:
            best_rms = value
            best_start = start
    return di_audio[best_start : best_start + window], target_audio[best_start : best_start + window]


def render_source_matched_hammerstein_layer(
    source_di: np.ndarray,
    sample_rate: int,
    matched_transfer: dict,
    max_layers: int = HAMMERSTEIN_LAYER_TOP_N,
    min_weight: float = HAMMERSTEIN_LAYER_MIN_WEIGHT,
    max_learn_seconds: float = HAMMERSTEIN_LAYER_MAX_SECONDS,
) -> tuple[np.ndarray | None, dict]:
    top_matches = list(matched_transfer.get("top_matches", []))
    usable_matches = [
        item
        for item in top_matches[:max_layers]
        if float(item.get("weight", 0.0)) >= min_weight
        and Path(str(item.get("di", ""))).exists()
        and Path(str(item.get("target", ""))).exists()
    ]
    if not usable_matches:
        return None, {"active": False, "reason": "no_usable_top_matches"}

    rendered_layers = []
    raw_weights = []
    diagnostics = []
    for item in usable_matches:
        try:
            di_rate, train_di = read_wav_float(Path(str(item["di"])))
            target_rate, train_target = read_wav_float(Path(str(item["target"])))
            train_di = resample_if_needed(train_di, di_rate, sample_rate)
            train_target = resample_if_needed(train_target, target_rate, sample_rate)
            train_di = normalize_peak(remove_dc(train_di), peak=0.95)
            train_target = normalize_peak(remove_dc(train_target), peak=0.95)
            train_di, train_target, _, _ = align_pair(
                train_di,
                train_target,
                max_lag_s=0.05,
                sample_rate=sample_rate,
            )
            train_di, train_target = loudest_aligned_segment(
                train_di,
                train_target,
                sample_rate=sample_rate,
                max_seconds=max_learn_seconds,
            )
            model, _, metrics = estimate_parallel_hammerstein_model(
                train_di,
                train_target,
                sample_rate=sample_rate,
                ir_ms=64.0,
                regularization=0.035,
            )
            _, layer = apply_parallel_hammerstein_model(source_di, sample_rate, model)
        except Exception as exc:
            diagnostics.append(
                {
                    "di": str(item.get("di", "")),
                    "weight": float(item.get("weight", 0.0)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        rendered_layers.append(remove_dc(layer[: len(source_di)]))
        raw_weights.append(float(item.get("weight", 0.0)))
        diagnostics.append(
            {
                "di": str(item.get("di", "")),
                "weight": float(item.get("weight", 0.0)),
                "spectral_error_db": float(metrics.get("spectral_error_db", 0.0)),
                "correlation": float(metrics.get("match_correlation", 0.0)),
            }
        )

    if not rendered_layers:
        return None, {"active": False, "reason": "all_layers_failed", "layers": diagnostics}

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights /= float(np.sum(weights) + 1e-12)
    layer = np.zeros_like(source_di, dtype=np.float64)
    for rendered, weight in zip(rendered_layers, weights):
        layer += rendered[: len(source_di)] * float(weight)

    return layer, {
        "active": True,
        "layer_count": int(len(rendered_layers)),
        "mix": float(HAMMERSTEIN_LAYER_MIX),
        "layers": diagnostics,
    }


def blend_hammerstein_amp_layer(
    base_audio: np.ndarray,
    source_di: np.ndarray,
    sample_rate: int,
    matched_transfer: dict,
    mix_override: float | None = None,
) -> tuple[np.ndarray, dict]:
    requested_mix = HAMMERSTEIN_LAYER_MIX if mix_override is None else float(mix_override)
    requested_mix = float(np.clip(requested_mix, 0.0, 0.62))
    if requested_mix <= 0.0:
        return base_audio, {
            "active": False,
            "reason": "disabled_for_dry_close_mic_render",
            "mix": 0.0,
        }

    layer, diagnostics = render_source_matched_hammerstein_layer(
        source_di=source_di,
        sample_rate=sample_rate,
        matched_transfer=matched_transfer,
    )
    if layer is None:
        return base_audio, diagnostics

    base = remove_dc(base_audio.astype(np.float64))
    layer = layer[: len(base)]
    layer *= rms(base) / (rms(layer) + 1e-12)
    top_weight = max([float(item.get("weight", 0.0)) for item in matched_transfer.get("top_matches", [])] or [0.0])
    mix = float(np.clip(requested_mix + (0.08 * top_weight), 0.0, 0.62))
    blended = ((1.0 - mix) * base) + (mix * layer)
    blended *= rms(base) / (rms(blended) + 1e-12)
    diagnostics["mix"] = mix
    return remove_dc(blended), diagnostics


def apply_source_matched_band_balance(
    audio: np.ndarray,
    source_di: np.ndarray,
    sample_rate: int,
    matched_transfer: dict,
    strength: float = 0.70,
) -> np.ndarray:
    anchor_freqs = np.asarray(matched_transfer.get("curve_freqs_hz", []), dtype=np.float64)
    target_gain_db = np.asarray(matched_transfer.get("target_over_di_gain_db", []), dtype=np.float64)
    if len(anchor_freqs) < 2 or len(anchor_freqs) != len(target_gain_db):
        return audio

    compare_len = min(len(audio), len(source_di))
    if compare_len < 1024:
        return audio

    fft_size = int(matched_transfer.get("fft_size", AMP_TONE_ANCHOR_FFT_SIZE))
    _, source_power = average_power_spectrum(source_di[:compare_len], fft_size=fft_size)
    actual_freqs, output_power = average_power_spectrum(audio[:compare_len], fft_size=fft_size)
    actual_gain_db = 10.0 * np.log10((output_power + 1e-12) / (source_power + 1e-12))
    actual_on_anchor = np.interp(anchor_freqs, actual_freqs, actual_gain_db, left=actual_gain_db[0], right=actual_gain_db[-1])
    missing = target_gain_db - actual_on_anchor

    bands = [
        (45.0, 85.0, 5.0, 1.00),
        (85.0, 150.0, 5.5, 1.00),
        (150.0, 350.0, 4.0, 0.90),
        (350.0, 900.0, 3.5, 0.70),
        (900.0, 1800.0, 3.0, 0.65),
        (1800.0, 3200.0, 3.0, 0.60),
        (3200.0, 6500.0, 4.5, 0.78),
        (6500.0, 12000.0, 5.0, 0.90),
    ]
    points: list[tuple[float, float]] = [(20.0, 0.0)]
    for low_hz, high_hz, max_db, band_strength in bands:
        mask = (anchor_freqs >= low_hz) & (anchor_freqs <= high_hz)
        if not np.any(mask):
            continue
        value = float(np.median(missing[mask]))
        value = float(np.clip(value, -max_db, max_db) * strength * band_strength)
        points.append((float(np.sqrt(low_hz * high_hz)), value))
    points.append((sample_rate / 2.0, 0.0))
    points = sorted(points, key=lambda item: item[0])
    freqs = np.array([item[0] for item in points], dtype=np.float64)
    gains = smooth_gain_db(np.array([item[1] for item in points], dtype=np.float64), smoothing_bins=3)
    return apply_frequency_gain_curve(audio, sample_rate, freqs, gains)


def match_local_amp_dynamic_behavior(
    audio: np.ndarray,
    reference: np.ndarray | None,
    sample_rate: int,
    anchor: dict | None = None,
    frame_ms: float = 250.0,
    hop_ms: float = 125.0,
) -> tuple[np.ndarray, dict]:
    if reference is None:
        return match_amp_dynamic_behavior(audio, reference=None, anchor=anchor)

    compare_len = min(len(audio), len(reference))
    if compare_len < sample_rate // 2:
        return match_amp_dynamic_behavior(audio, reference=reference, anchor=anchor)

    frame_samples = max(512, int(round(sample_rate * frame_ms / 1000.0)))
    hop_samples = max(128, int(round(sample_rate * hop_ms / 1000.0)))
    window = np.hanning(frame_samples)
    if float(np.max(window)) <= 0.0:
        window = np.ones(frame_samples, dtype=np.float64)

    padded_audio = np.pad(remove_dc(audio[:compare_len]), (frame_samples // 2, frame_samples // 2), mode="reflect")
    padded_ref = np.pad(remove_dc(reference[:compare_len]), (frame_samples // 2, frame_samples // 2), mode="reflect")
    output = np.zeros_like(padded_audio, dtype=np.float64)
    window_sum = np.zeros_like(padded_audio, dtype=np.float64)
    target_crests = []
    before_crests = []
    after_crests = []

    for start in range(0, len(padded_audio) - frame_samples + 1, hop_samples):
        frame = padded_audio[start : start + frame_samples]
        ref_frame = padded_ref[start : start + frame_samples]
        target_crest = crest_factor(ref_frame)
        before_crest = crest_factor(frame)
        processed = apply_peak_rms_amp_compression(frame, target_crest_factor=target_crest)
        processed *= rms(frame) / (rms(processed) + 1e-12)
        output[start : start + frame_samples] += processed * window
        window_sum[start : start + frame_samples] += window
        target_crests.append(target_crest)
        before_crests.append(before_crest)
        after_crests.append(crest_factor(processed))

    valid = window_sum > 1e-8
    output[valid] /= window_sum[valid]
    output = output[frame_samples // 2 : frame_samples // 2 + compare_len]
    output = match_reference_level(output, reference[:compare_len], mode="rms")
    output = np.concatenate([output, audio[compare_len:]]) if len(audio) > compare_len else output
    before = amp_dynamic_fingerprint(audio[:compare_len])
    after = amp_dynamic_fingerprint(output[:compare_len])
    target = amp_dynamic_fingerprint(reference[:compare_len])
    return remove_dc(output), {
        "active": True,
        "changed": True,
        "local": True,
        "target": target,
        "before": before,
        "after": after,
        "median_frame_target_crest": float(np.median(target_crests)),
        "median_frame_before_crest": float(np.median(before_crests)),
        "median_frame_after_crest": float(np.median(after_crests)),
    }


def match_amp_dynamic_behavior(
    audio: np.ndarray,
    reference: np.ndarray | None = None,
    anchor: dict | None = None,
) -> tuple[np.ndarray, dict]:
    reference_metrics = amp_dynamic_fingerprint(reference) if reference is not None else {}
    if not reference_metrics and anchor:
        reference_metrics = dict(anchor.get("target_dynamic_fingerprint", {}))
    if not reference_metrics:
        return audio, {"active": False}

    before = amp_dynamic_fingerprint(audio)
    target_crest = float(reference_metrics.get("crest_factor", before["crest_factor"]))
    target_peak_over_rms = float(reference_metrics.get("peak_over_rms_db", before["peak_over_rms_db"]))
    needs_compression = (
        before["crest_factor"] > target_crest * 1.08
        or before["peak_over_rms_db"] > target_peak_over_rms + 0.75
    )
    if not needs_compression:
        return audio, {
            "active": True,
            "changed": False,
            "target": reference_metrics,
            "before": before,
            "after": before,
        }

    compressed = apply_peak_rms_amp_compression(audio, target_crest_factor=target_crest)
    if reference is not None:
        compressed = match_reference_level(compressed, reference, mode="rms")
    else:
        compressed *= before["rms"] / (rms(compressed) + 1e-12)
    after = amp_dynamic_fingerprint(compressed)
    return compressed, {
        "active": True,
        "changed": True,
        "target": reference_metrics,
        "before": before,
        "after": after,
    }


def recording_pair_quality_report(
    di_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    di_path: Path | None = None,
    target_path: Path | None = None,
    lag: int = 0,
    polarity: int = 1,
) -> dict:
    compare_len = min(len(di_audio), len(target_audio))
    if compare_len < 1024:
        return {
            "quality_weight": 0.0,
            "quality_gate_excluded": True,
            "severity": 10,
            "issues": ["too little aligned audio for quality analysis"],
            "di": str(di_path or ""),
            "target": str(target_path or ""),
        }

    di_excerpt, target_excerpt = loudest_aligned_segment(
        di_audio[:compare_len],
        target_audio[:compare_len],
        sample_rate=sample_rate,
        max_seconds=18.0,
    )
    excerpt_len = min(len(di_excerpt), len(target_excerpt))
    di_excerpt = di_excerpt[:excerpt_len]
    target_excerpt = target_excerpt[:excerpt_len]
    di_gain_baseline = match_reference_level(di_excerpt, target_excerpt, mode="rms")

    baseline_spec = spectral_error_db(target_excerpt, di_gain_baseline, sample_rate)
    aligned_corr = abs(correlation(di_gain_baseline, target_excerpt))
    di_dynamic = amp_dynamic_fingerprint(di_excerpt)
    target_dynamic = amp_dynamic_fingerprint(target_excerpt)
    di_tone = tone_features(di_excerpt, sample_rate)
    target_tone = tone_features(target_excerpt, sample_rate)
    di_advanced = advanced_audio_descriptors(di_excerpt, sample_rate)
    target_advanced = advanced_audio_descriptors(target_excerpt, sample_rate)
    rms_delta_db = abs(rms_dbfs(target_excerpt) - rms_dbfs(di_excerpt))
    loudness_delta_db = None
    if "integrated_loudness_lufs" in di_advanced and "integrated_loudness_lufs" in target_advanced:
        loudness_delta_db = abs(
            float(target_advanced["integrated_loudness_lufs"]) - float(di_advanced["integrated_loudness_lufs"])
        )
    lag_ms = (abs(int(lag)) / max(1, sample_rate)) * 1000.0

    issues: list[str] = []
    severity = 0
    weight = 1.0

    if baseline_spec < AMP_QUALITY_EXCLUDE_SPECTRAL_ERROR_DB:
        issues.append("amp/mic target is too close to gain-matched DI")
        severity += 5
        weight = 0.0
    elif baseline_spec < AMP_QUALITY_WEAK_SPECTRAL_ERROR_DB:
        issues.append("weak amp/cab spectral difference")
        severity += 3
        weight *= 0.35

    if aligned_corr >= AMP_QUALITY_DI_LIKE_CORRELATION:
        issues.append("DI and mic waveforms are too correlated")
        severity += 5
        weight = 0.0
    elif aligned_corr >= AMP_QUALITY_PARTIAL_DI_LIKE_CORRELATION:
        issues.append("DI and mic waveforms are partly too correlated")
        severity += 2
        weight *= 0.55

    di_crest = float(di_dynamic.get("crest_factor", 0.0))
    target_crest = float(target_dynamic.get("crest_factor", 0.0))
    if di_crest > 0.0 and target_crest >= di_crest * 0.92:
        issues.append("mic dynamics are too DI-like / not compressed")
        severity += 3
        if aligned_corr >= 0.14:
            weight = 0.0
        else:
            weight *= 0.45

    if float(target_dynamic.get("transient_rms_ratio", 0.0)) < 0.065:
        issues.append("mic transient detail is unusually low")
        severity += 1
        weight *= 0.75

    if rms_delta_db > 8.0:
        issues.append("DI and mic RMS levels differ by more than 8 dB")
        severity += 1
        weight *= 0.80

    if loudness_delta_db is not None and loudness_delta_db > 10.0:
        issues.append("DI and mic integrated loudness differ by more than 10 LU")
        severity += 1
        weight *= 0.85

    if lag_ms > 35.0:
        issues.append("alignment lag is large")
        severity += 1
        weight *= 0.85

    if int(polarity) < 0:
        issues.append("polarity inverted after alignment")
        severity += 1
        weight *= 0.95

    if target_tone["high_energy_ratio"] > 0.82 or target_tone["spectral_centroid_hz"] > 4500.0:
        issues.append("bright/fizz energy is unusually high")
        severity += 1
        weight *= 0.70

    if float(target_advanced.get("spectral_flatness", 0.0)) > 0.12:
        issues.append("mic target has unusually noise-like spectral flatness")
        severity += 1
        weight *= 0.85

    if target_tone["low_energy_ratio"] > 0.78 or target_tone["low_energy_ratio"] < 0.025:
        issues.append("low-end/body balance is an outlier")
        severity += 1
        weight *= 0.75

    weight = float(np.clip(weight, 0.0, 1.0))
    return {
        "quality_weight": weight,
        "quality_gate_excluded": bool(weight < AMP_QUALITY_MIN_WEIGHT),
        "severity": int(severity),
        "issues": issues,
        "di": str(di_path or ""),
        "target": str(target_path or ""),
        "di_gain_baseline_spectral_error_db": float(baseline_spec),
        "di_target_correlation": float(aligned_corr),
        "di_dynamic": di_dynamic,
        "target_dynamic": target_dynamic,
        "di_tone_features": di_tone,
        "target_tone_features": target_tone,
        "di_advanced_audio_features": di_advanced,
        "target_advanced_audio_features": target_advanced,
        "loudness_delta_lu": float(loudness_delta_db) if loudness_delta_db is not None else None,
        "rms_delta_db": float(rms_delta_db),
        "alignment_lag_ms": float(lag_ms),
        "polarity": int(polarity),
    }


def apply_recording_quality_context(reports: list[dict]) -> None:
    active_reports = [report for report in reports if "target_tone_features" in report]
    if len(active_reports) < 4:
        return

    high_values = np.asarray(
        [float(report["target_tone_features"].get("high_energy_ratio", 0.0)) for report in active_reports],
        dtype=np.float64,
    )
    low_values = np.asarray(
        [float(report["target_tone_features"].get("low_energy_ratio", 0.0)) for report in active_reports],
        dtype=np.float64,
    )
    centroid_values = np.asarray(
        [float(report["target_tone_features"].get("spectral_centroid_hz", 0.0)) for report in active_reports],
        dtype=np.float64,
    )

    def robust_z(values: np.ndarray) -> np.ndarray:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(1e-6, mad * 1.4826)
        return (values - median) / scale

    high_z = robust_z(high_values)
    low_z = robust_z(low_values)
    centroid_z = robust_z(centroid_values)

    for report, high_score, low_score, centroid_score in zip(active_reports, high_z, low_z, centroid_z):
        weight = float(report.get("quality_weight", 1.0))
        severity = int(report.get("severity", 0))
        issues = list(report.get("issues", []))
        if high_score > 1.85 or centroid_score > 2.0:
            if "bright/fizz energy is a dataset outlier" not in issues:
                issues.append("bright/fizz energy is a dataset outlier")
            severity += 1
            weight *= 0.60
        if abs(low_score) > 1.85:
            if "low-end/body balance is a dataset outlier" not in issues:
                issues.append("low-end/body balance is a dataset outlier")
            severity += 1
            weight *= 0.70
        weight = float(np.clip(weight, 0.0, 1.0))
        report["quality_weight"] = weight
        report["quality_gate_excluded"] = bool(weight < AMP_QUALITY_MIN_WEIGHT)
        report["severity"] = severity
        report["issues"] = issues


def quality_report_for_pair_spec(spec: dict, sample_rate: int, context_radius: int = 256) -> dict:
    di_path = Path(spec["di_path"])
    target_path = Path(spec["target_path"])
    try:
        di_rate, di_audio = read_wav_float(di_path)
        target_rate, target_audio = read_wav_float(target_path)
        di_audio = resample_if_needed(di_audio, di_rate, sample_rate)
        target_audio = resample_if_needed(target_audio, target_rate, sample_rate)
        di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
        target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
        di_aligned, target_aligned, lag, polarity = align_pair(
            di_audio,
            target_audio,
            max_lag_s=0.05,
            sample_rate=sample_rate,
        )
        min_len = min(len(di_aligned), len(target_aligned))
        usable_end = min_len - int(context_radius)
        if usable_end < 1024:
            raise ValueError("not enough usable aligned audio")
        return recording_pair_quality_report(
            di_aligned[:min_len],
            target_aligned[:min_len],
            sample_rate=sample_rate,
            di_path=di_path,
            target_path=target_path,
            lag=lag,
            polarity=polarity,
        )
    except Exception as exc:
        return {
            "quality_weight": 0.0,
            "quality_gate_excluded": True,
            "severity": 10,
            "issues": [f"quality analysis failed: {type(exc).__name__}: {exc}"],
            "di": str(di_path),
            "target": str(target_path),
        }


def build_all_recordings_amp_tone_anchor(
    training_pairs: list[dict],
    sample_rate: int,
    strength: float = 1.0,
    fft_size: int = AMP_TONE_ANCHOR_FFT_SIZE,
    smoothing_bins: int = AMP_TONE_ANCHOR_SMOOTHING_BINS,
    max_gain_db: float = AMP_TONE_ANCHOR_MAX_GAIN_DB,
) -> dict:
    """
    Average the DI-to-amp/mic curve from every selected recording pair.

    The neural waveform model can learn compression and pick shape, but this
    anchor makes the saved model carry the all-recordings SM57 amp/cab color so
    final renders cannot collapse into gain-matched DI.
    """
    curves = []
    target_dynamic_items = []
    source_dynamic_items = []
    target_envelope_items = []
    source_envelope_items = []
    target_tone_profile_items = []
    source_tone_profile_items = []
    source_feature_rows = []
    transfer_bank = []
    segment_transfer_bank = []
    quality_weights = []
    freqs_hz = None
    for pair in training_pairs:
        quality_weight = float(np.clip(pair.get("quality_weight", 1.0), 0.0, 1.0))
        if quality_weight <= 0.0:
            continue
        curve_freqs, gain_db = reference_spectral_gain_curve(
            pair["di_aligned"],
            pair["target_aligned"],
            sample_rate=sample_rate,
            fft_size=fft_size,
            smoothing_bins=smoothing_bins,
            max_gain_db=max_gain_db,
        )
        freqs_hz = curve_freqs
        curves.append(gain_db)
        source_dynamic = amp_dynamic_fingerprint(pair["di_aligned"])
        target_dynamic = amp_dynamic_fingerprint(pair["target_aligned"])
        source_envelope = local_envelope_profile(pair["di_aligned"], sample_rate)
        target_envelope = local_envelope_profile(pair["target_aligned"], sample_rate)
        source_tone_profile = detailed_tone_profile(pair["di_aligned"], sample_rate)
        target_tone_profile = detailed_tone_profile(pair["target_aligned"], sample_rate)
        source_features = source_match_feature_vector(pair["di_aligned"], sample_rate)
        source_dynamic_items.append(source_dynamic)
        target_dynamic_items.append(target_dynamic)
        source_envelope_items.append(source_envelope)
        target_envelope_items.append(target_envelope)
        source_tone_profile_items.append(source_tone_profile)
        target_tone_profile_items.append(target_tone_profile)
        source_feature_rows.append(source_features)
        quality_weights.append(quality_weight)
        transfer_bank.append(
            {
                "index": int(pair.get("index", len(transfer_bank) + 1)),
                "take_name": str(pair.get("take_name", "")),
                "di": str(pair.get("di_path", "")),
                "target": str(pair.get("target_path", "")),
                "rig_fingerprint": str(pair.get("rig_fingerprint", "")),
                "mic_position": str(pair.get("mic_position", "")),
                "take_metadata": dict(pair.get("take_metadata", {})),
                "quality_weight": quality_weight,
                "quality_report": dict(pair.get("quality_report", {})),
                "source_features": [float(value) for value in source_features],
                "source_dynamic_fingerprint": source_dynamic,
                "target_dynamic_fingerprint": target_dynamic,
                "source_local_envelope_profile": source_envelope,
                "target_local_envelope_profile": target_envelope,
                "source_detailed_tone_profile": source_tone_profile,
                "target_detailed_tone_profile": target_tone_profile,
                "target_over_di_gain_db": [float(value) for value in gain_db],
            }
        )
        segment_samples = max(1024, int(round(sample_rate * AMP_TONE_SEGMENT_SECONDS)))
        segment_samples = min(segment_samples, len(pair["di_aligned"]), len(pair["target_aligned"]))
        hop_samples = max(512, int(round(sample_rate * AMP_TONE_SEGMENT_HOP_SECONDS)))
        last_start = max(0, min(len(pair["di_aligned"]), len(pair["target_aligned"])) - segment_samples)
        starts = list(range(0, last_start + 1, hop_samples))
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        for start in starts:
            di_segment = pair["di_aligned"][start : start + segment_samples]
            target_segment = pair["target_aligned"][start : start + segment_samples]
            if min(len(di_segment), len(target_segment)) < 1024:
                continue
            segment_transfer_bank.append(
                {
                    "take_name": str(pair.get("take_name", "")),
                    "di": str(pair.get("di_path", "")),
                    "target": str(pair.get("target_path", "")),
                    "rig_fingerprint": str(pair.get("rig_fingerprint", "")),
                    "mic_position": str(pair.get("mic_position", "")),
                    "quality_weight": quality_weight,
                    "start_seconds": float(start / sample_rate),
                    "duration_seconds": float(len(di_segment) / sample_rate),
                    "source_features": [
                        float(value) for value in source_match_feature_vector(di_segment, sample_rate)
                    ],
                    "source_detailed_tone_profile": detailed_tone_profile(di_segment, sample_rate),
                    "target_detailed_tone_profile": detailed_tone_profile(target_segment, sample_rate),
                    "target_local_envelope_profile": local_envelope_profile(target_segment, sample_rate),
                    "target_dynamic_fingerprint": amp_dynamic_fingerprint(target_segment),
                }
            )

    if not curves or freqs_hz is None:
        return {"enabled": False, "pair_count": 0}

    normalized_quality_weights = np.asarray(quality_weights, dtype=np.float64)
    normalized_quality_weights /= float(np.sum(normalized_quality_weights) + 1e-12)
    average_gain_db = np.average(np.stack(curves, axis=0), axis=0, weights=normalized_quality_weights)
    average_gain_db = smooth_gain_db(average_gain_db, smoothing_bins=smoothing_bins)
    average_gain_db = np.clip(average_gain_db, -float(max_gain_db), float(max_gain_db))
    source_feature_matrix = np.stack(source_feature_rows, axis=0)
    source_feature_mean = np.sum(source_feature_matrix * normalized_quality_weights.reshape(-1, 1), axis=0)
    source_feature_var = np.sum(
        np.square(source_feature_matrix - source_feature_mean.reshape(1, -1))
        * normalized_quality_weights.reshape(-1, 1),
        axis=0,
    )
    source_feature_std = np.maximum(np.sqrt(source_feature_var), 1e-6)
    return {
        "enabled": True,
        "source": "quality_weighted_selected_recording_pairs",
        "pair_count": int(len(curves)),
        "quality_weighted": True,
        "quality_weight_sum": float(np.sum(quality_weights)),
        "quality_min_weight": float(AMP_QUALITY_MIN_WEIGHT),
        "sample_rate_hz": int(sample_rate),
        "fft_size": int(fft_size),
        "smoothing_bins": int(smoothing_bins),
        "max_gain_db": float(max_gain_db),
        "default_strength": float(np.clip(strength, 0.0, 1.5)),
        "render_max_correction_db": float(max(6.0, max_gain_db)),
        "source_feature_names": list(AMP_SOURCE_CONDITIONING_FEATURES),
        "source_feature_mean": [float(value) for value in source_feature_mean],
        "source_feature_std": [float(value) for value in source_feature_std],
        "source_dynamic_fingerprint": weighted_amp_dynamic_fingerprint(source_dynamic_items, normalized_quality_weights),
        "target_dynamic_fingerprint": weighted_amp_dynamic_fingerprint(target_dynamic_items, normalized_quality_weights),
        "source_local_envelope_profile": {
            key: float(np.sum(normalized_quality_weights * np.asarray([item[key] for item in source_envelope_items])))
            for key in ("p10_db", "p50_db", "p90_db", "spread_db")
        },
        "target_local_envelope_profile": {
            key: float(np.sum(normalized_quality_weights * np.asarray([item[key] for item in target_envelope_items])))
            for key in ("p10_db", "p50_db", "p90_db", "spread_db")
        },
        "source_detailed_tone_profile": {
            "bands_hz": source_tone_profile_items[0]["bands_hz"],
            "energy_ratios": [
                float(value)
                for value in np.sum(
                    np.asarray([item["energy_ratios"] for item in source_tone_profile_items], dtype=np.float64)
                    * normalized_quality_weights.reshape(-1, 1),
                    axis=0,
                )
            ],
        },
        "target_detailed_tone_profile": {
            "bands_hz": target_tone_profile_items[0]["bands_hz"],
            "energy_ratios": [
                float(value)
                for value in np.sum(
                    np.asarray([item["energy_ratios"] for item in target_tone_profile_items], dtype=np.float64)
                    * normalized_quality_weights.reshape(-1, 1),
                    axis=0,
                )
            ],
        },
        "curve_freqs_hz": [float(value) for value in freqs_hz],
        "target_over_di_gain_db": [float(value) for value in average_gain_db],
        "per_take_transfer_bank": transfer_bank,
        "segment_seconds": float(AMP_TONE_SEGMENT_SECONDS),
        "segment_hop_seconds": float(AMP_TONE_SEGMENT_HOP_SECONDS),
        "segment_transfer_bank": segment_transfer_bank,
    }


def derive_amp_tone_anchor_from_model_metadata(metadata: dict) -> dict:
    model_rate = int(metadata.get("sample_rate_hz", 0))
    if model_rate <= 0:
        return {"enabled": False, "pair_count": 0}

    training_pairs = []
    for item in dict(metadata.get("training", {})).get("training_pairs", []):
        di_path = Path(str(item.get("di", "")))
        target_path = Path(str(item.get("target", "")))
        if not di_path.exists() or not target_path.exists():
            continue

        di_rate, di_audio = read_wav_float(di_path)
        target_rate, target_audio = read_wav_float(target_path)
        di_audio = resample_if_needed(di_audio, di_rate, model_rate)
        target_audio = resample_if_needed(target_audio, target_rate, model_rate)
        di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
        target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
        di_aligned, target_aligned, lag, polarity = align_pair(
            di_audio,
            target_audio,
            max_lag_s=0.05,
            sample_rate=model_rate,
        )
        min_len = min(len(di_aligned), len(target_aligned))
        if min_len < 1024:
            continue
        quality_report = dict(item.get("quality_report", {}))
        if not quality_report:
            quality_report = recording_pair_quality_report(
                di_aligned[:min_len],
                target_aligned[:min_len],
                sample_rate=model_rate,
                di_path=di_path,
                target_path=target_path,
                lag=lag,
                polarity=polarity,
            )
        quality_weight = float(item.get("quality_weight", quality_report.get("quality_weight", 1.0)))
        training_pairs.append(
            {
                "index": int(item.get("index", len(training_pairs) + 1)),
                "take_name": str(item.get("take_name", "")),
                "di_path": di_path,
                "target_path": target_path,
                "rig_fingerprint": str(item.get("rig_fingerprint", "")),
                "mic_position": str(item.get("mic_position", "")),
                "take_metadata": dict(item.get("take_metadata", {})),
                "di_aligned": di_aligned[:min_len],
                "target_aligned": target_aligned[:min_len],
                "quality_weight": quality_weight,
                "quality_report": quality_report,
            }
        )

    if not training_pairs:
        return {"enabled": False, "pair_count": 0}

    apply_recording_quality_context([pair["quality_report"] for pair in training_pairs])
    for pair in training_pairs:
        pair["quality_weight"] = float(pair["quality_report"].get("quality_weight", pair.get("quality_weight", 1.0)))

    return build_all_recordings_amp_tone_anchor(
        training_pairs,
        sample_rate=model_rate,
        strength=1.0,
        smoothing_bins=AMP_TONE_ANCHOR_SMOOTHING_BINS,
        max_gain_db=AMP_TONE_ANCHOR_MAX_GAIN_DB,
    )


def amp_tone_anchor_for_metadata(metadata: dict) -> dict:
    anchor = dict(metadata.get("amp_tone_anchor", {}))
    if anchor.get("enabled", False):
        training_pairs = list(dict(metadata.get("training", {})).get("training_pairs", []))
        pair_context = {}
        for item in training_pairs:
            path = Path(str(item.get("di", "")))
            context = {
                "rig_fingerprint": str(item.get("rig_fingerprint", "")),
                "mic_position": str(item.get("mic_position", "")),
                "take_metadata": dict(item.get("take_metadata", {})),
            }
            pair_context[str(path)] = context
            pair_context[path.name] = context
        enriched_bank = []
        for bank_item in list(anchor.get("per_take_transfer_bank", [])):
            enriched = dict(bank_item)
            path = Path(str(enriched.get("di", "")))
            context = pair_context.get(str(path)) or pair_context.get(path.name) or {}
            for key in ("rig_fingerprint", "mic_position", "take_metadata"):
                if not enriched.get(key) and context.get(key):
                    enriched[key] = context[key]
            enriched_bank.append(enriched)
        if enriched_bank:
            anchor["per_take_transfer_bank"] = enriched_bank
        return anchor
    return derive_amp_tone_anchor_from_model_metadata(metadata)


def apply_all_recordings_amp_tone_anchor(
    audio: np.ndarray,
    source_di: np.ndarray,
    sample_rate: int,
    metadata: dict,
    strength: float | None = None,
    source_hint_path: Path | None = None,
    rig_fingerprint_value: str | None = None,
    mic_position: str | None = None,
) -> np.ndarray:
    anchor = amp_tone_anchor_for_metadata(metadata)
    if not anchor.get("enabled", False):
        return audio

    matched_transfer = source_matched_amp_transfer(
        anchor,
        source_di,
        sample_rate,
        source_hint_path=source_hint_path,
        rig_fingerprint_value=rig_fingerprint_value,
        mic_position=mic_position,
    )
    anchor_freqs = np.asarray(matched_transfer.get("curve_freqs_hz", []), dtype=np.float64)
    target_gain_db = np.asarray(matched_transfer.get("target_over_di_gain_db", []), dtype=np.float64)
    if len(anchor_freqs) < 2 or len(anchor_freqs) != len(target_gain_db):
        return audio

    correction_strength = float(anchor.get("default_strength", 1.0) if strength is None else strength)
    correction_strength = float(np.clip(correction_strength, 0.0, 1.5))
    if correction_strength <= 0.0:
        return audio

    compare_len = min(len(audio), len(source_di))
    if compare_len < 1024:
        return audio

    fft_size = int(anchor.get("fft_size", AMP_TONE_ANCHOR_FFT_SIZE))
    smoothing_bins = int(anchor.get("smoothing_bins", AMP_TONE_ANCHOR_SMOOTHING_BINS))
    _, source_power = average_power_spectrum(source_di[:compare_len], fft_size=fft_size)
    actual_freqs, output_power = average_power_spectrum(audio[:compare_len], fft_size=fft_size)
    actual_gain_db = 10.0 * np.log10((output_power + 1e-12) / (source_power + 1e-12))
    actual_gain_db = smooth_gain_db(actual_gain_db, smoothing_bins=smoothing_bins)
    actual_on_anchor_bins = np.interp(
        anchor_freqs,
        actual_freqs,
        actual_gain_db,
        left=actual_gain_db[0],
        right=actual_gain_db[-1],
    )

    missing_amp_curve_db = target_gain_db - actual_on_anchor_bins
    max_correction_db = float(anchor.get("render_max_correction_db", AMP_TONE_ANCHOR_MAX_GAIN_DB))
    missing_amp_curve_db = smooth_gain_db(missing_amp_curve_db, smoothing_bins=smoothing_bins)
    correction_db = np.clip(missing_amp_curve_db, -max_correction_db, max_correction_db) * correction_strength
    return apply_frequency_gain_curve(audio, sample_rate, anchor_freqs, correction_db)


def amp_tone_guard_metrics(
    source_di: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    min_improvement_db: float = AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
    min_movement_db: float = AMP_TONE_GUARD_MIN_MOVEMENT_DB,
) -> dict:
    compare_len = min(len(source_di), len(candidate), len(reference))
    if compare_len < 1024:
        return {
            "passes": True,
            "reason": "too_short_to_score",
            "compare_seconds": float(compare_len / max(1, sample_rate)),
        }

    source = source_di[:compare_len]
    output = candidate[:compare_len]
    target = reference[:compare_len]
    di_baseline = match_reference_level(source, target, mode="rms")
    output_matched = match_reference_level(output, target, mode="rms")
    di_error = spectral_error_db(target, di_baseline, sample_rate)
    output_error = spectral_error_db(target, output_matched, sample_rate)
    output_vs_di = spectral_error_db(di_baseline, output_matched, sample_rate)
    reference_dynamics = amp_dynamic_fingerprint(target)
    output_dynamics = amp_dynamic_fingerprint(output_matched)
    source_dynamics = amp_dynamic_fingerprint(di_baseline)
    required_improvement = max(float(min_improvement_db), di_error * 0.18)
    required_movement = max(float(min_movement_db), di_error * 0.22)
    improvement = di_error - output_error
    dynamics_passes = (
        output_dynamics["crest_factor"] <= reference_dynamics["crest_factor"] * AMP_DYNAMICS_GUARD_MAX_CREST_RATIO
        and output_dynamics["peak_over_rms_db"]
        <= reference_dynamics["peak_over_rms_db"] + AMP_DYNAMICS_GUARD_MAX_PEAK_OVER_RMS_DB_DELTA
    )
    spectral_passes = improvement >= required_improvement and output_vs_di >= required_movement
    passes = spectral_passes and dynamics_passes
    reason = "ok"
    if not spectral_passes:
        reason = "render_too_close_to_gain_matched_di"
    elif not dynamics_passes:
        reason = "render_dynamics_still_di_like"
    return {
        "passes": bool(passes),
        "reason": reason,
        "di_gain_baseline_spectral_error_db": float(di_error),
        "render_spectral_error_db": float(output_error),
        "render_improvement_over_di_db": float(improvement),
        "render_vs_di_spectral_distance_db": float(output_vs_di),
        "required_improvement_db": float(required_improvement),
        "required_movement_db": float(required_movement),
        "spectral_passes": bool(spectral_passes),
        "dynamics_passes": bool(dynamics_passes),
        "source_dynamics": source_dynamics,
        "reference_dynamics": reference_dynamics,
        "render_dynamics": output_dynamics,
        "compare_seconds": float(compare_len / max(1, sample_rate)),
    }


def heldout_amp_model_metrics(
    source_di: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_spectral_error_db: float,
    min_correlation: float,
    max_level_error_db: float,
    min_improvement_db: float = AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
    min_movement_db: float = AMP_TONE_GUARD_MIN_MOVEMENT_DB,
) -> dict:
    compare_len = min(len(source_di), len(reference), len(candidate))
    if compare_len < 1024:
        return {"passes": False, "reason": "heldout_audio_too_short", "compare_samples": compare_len}
    source = source_di[:compare_len]
    target = reference[:compare_len]
    output = candidate[:compare_len]
    output_level_matched = match_reference_level(output, target, mode="rms")
    level_error_db = abs(rms_dbfs(output) - rms_dbfs(target))
    spectral_error = spectral_error_db(target, output_level_matched, sample_rate)
    match_corr = correlation(target, output_level_matched)
    guard = amp_tone_guard_metrics(
        source,
        output,
        target,
        sample_rate,
        min_improvement_db=min_improvement_db,
        min_movement_db=min_movement_db,
    )
    failures = []
    if spectral_error > float(max_spectral_error_db):
        failures.append(f"spectral_error={spectral_error:.2f}dB")
    if match_corr < float(min_correlation):
        failures.append(f"correlation={match_corr:.3f}")
    if level_error_db > float(max_level_error_db):
        failures.append(f"level_error={level_error_db:.2f}dB")
    if not guard.get("passes", False):
        failures.append(str(guard.get("reason", "amp_tone_guard_failed")))
    return {
        "passes": not failures,
        "failures": failures,
        "match_correlation": float(match_corr),
        "spectral_error_db": float(spectral_error),
        "level_error_db": float(level_error_db),
        "amp_tone_guard": guard,
        "compare_samples": int(compare_len),
        "compare_seconds": float(compare_len / max(1, sample_rate)),
    }


def aggregate_heldout_amp_metrics(items: list[dict]) -> dict:
    if not items:
        return {
            "pair_count": 0,
            "pass_rate": 0.0,
            "mean_match_correlation": 0.0,
            "mean_spectral_error_db": float("inf"),
            "max_spectral_error_db": float("inf"),
            "mean_level_error_db": float("inf"),
        }
    return {
        "pair_count": len(items),
        "pass_rate": float(np.mean([bool(item.get("passes", False)) for item in items])),
        "mean_match_correlation": float(np.mean([item["match_correlation"] for item in items])),
        "mean_spectral_error_db": float(np.mean([item["spectral_error_db"] for item in items])),
        "max_spectral_error_db": float(np.max([item["spectral_error_db"] for item in items])),
        "mean_level_error_db": float(np.mean([item["level_error_db"] for item in items])),
    }


def amp_model_promotion_decision(
    candidate: dict,
    existing: dict | None,
    max_mean_spectral_error_db: float,
    min_mean_correlation: float,
    min_pass_rate: float,
    min_existing_improvement_db: float,
    max_pair_regression_db: float,
    candidate_pairs: list[dict],
    existing_pairs: list[dict] | None = None,
) -> dict:
    failures = []
    if float(candidate["mean_spectral_error_db"]) > float(max_mean_spectral_error_db):
        failures.append(
            f"mean spectral error {candidate['mean_spectral_error_db']:.2f} dB > {max_mean_spectral_error_db:.2f} dB"
        )
    if float(candidate["mean_match_correlation"]) < float(min_mean_correlation):
        failures.append(
            f"mean correlation {candidate['mean_match_correlation']:.3f} < {min_mean_correlation:.3f}"
        )
    if float(candidate["pass_rate"]) < float(min_pass_rate):
        failures.append(f"held-out pass rate {candidate['pass_rate']:.1%} < {min_pass_rate:.1%}")

    comparison = None
    if existing is not None:
        improvement = float(existing["mean_spectral_error_db"] - candidate["mean_spectral_error_db"])
        comparison = {"mean_spectral_improvement_db": improvement}
        if improvement < float(min_existing_improvement_db):
            failures.append(
                f"existing-model improvement {improvement:.2f} dB < {min_existing_improvement_db:.2f} dB"
            )
        if existing_pairs:
            existing_by_index = {int(item["index"]): item for item in existing_pairs}
            regressions = []
            for item in candidate_pairs:
                previous = existing_by_index.get(int(item["index"]))
                if previous is None:
                    continue
                regressions.append(float(item["spectral_error_db"] - previous["spectral_error_db"]))
            worst_regression = max(regressions) if regressions else 0.0
            comparison["worst_pair_spectral_regression_db"] = float(worst_regression)
            if worst_regression > float(max_pair_regression_db):
                failures.append(
                    f"worst held-out pair regressed {worst_regression:.2f} dB > {max_pair_regression_db:.2f} dB"
                )
    return {
        "accepted": not failures,
        "failures": failures,
        "candidate": candidate,
        "existing": existing,
        "comparison": comparison,
    }


def enforce_amp_tone_regression_guard(
    output: np.ndarray,
    source_di: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    repair_strength: float,
    smoothing_bins: int,
    max_gain_db: float,
    min_improvement_db: float = AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
    min_movement_db: float = AMP_TONE_GUARD_MIN_MOVEMENT_DB,
) -> tuple[np.ndarray, dict]:
    metrics = amp_tone_guard_metrics(
        source_di,
        output,
        reference,
        sample_rate,
        min_improvement_db=min_improvement_db,
        min_movement_db=min_movement_db,
    )
    if metrics.get("passes", False):
        return output, metrics

    best_output = output
    best_metrics = metrics
    repair_strengths = [
        max(0.75, float(repair_strength)),
        max(1.0, float(repair_strength)),
        max(1.15, float(repair_strength)),
    ]
    for strength in repair_strengths:
        repaired = apply_reference_spectral_imprint(
            audio=output,
            reference=reference,
            sample_rate=sample_rate,
            strength=strength,
            smoothing_bins=smoothing_bins,
            max_gain_db=max(max_gain_db, AMP_TONE_ANCHOR_MAX_GAIN_DB),
        )
        repaired, _ = match_amp_dynamic_behavior(repaired, reference=reference)
        repaired = match_reference_level(repaired, reference, mode="rms")
        candidate_metrics = amp_tone_guard_metrics(
            source_di,
            repaired,
            reference,
            sample_rate,
            min_improvement_db=min_improvement_db,
            min_movement_db=min_movement_db,
        )
        if candidate_metrics["render_spectral_error_db"] < best_metrics["render_spectral_error_db"]:
            best_output = repaired
            best_metrics = candidate_metrics
        if candidate_metrics.get("passes", False):
            candidate_metrics["repaired"] = True
            candidate_metrics["repair_strength"] = float(strength)
            return repaired, candidate_metrics

    details = (
        "Amp-tone regression guard failed: render is still too close to gain-matched DI "
        f"(DI baseline spec={best_metrics.get('di_gain_baseline_spectral_error_db', 0.0):.2f} dB, "
        f"render spec={best_metrics.get('render_spectral_error_db', 0.0):.2f} dB, "
        f"movement={best_metrics.get('render_vs_di_spectral_distance_db', 0.0):.2f} dB)."
    )
    raise SystemExit(details)


def stft_audio(audio: np.ndarray, fft_size: int, hop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return complex STFT frames and the Hann window used to make them."""
    if fft_size < 256:
        raise ValueError("fft_size must be at least 256")
    if hop_size < 1 or hop_size >= fft_size:
        raise ValueError("hop_size must be between 1 and fft_size - 1")

    audio = remove_dc(audio.astype(np.float64))
    pad = fft_size // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    frame_count = 1 + max(0, (len(padded) - fft_size) // hop_size)
    window = np.hanning(fft_size)
    frames = np.zeros((frame_count, fft_size // 2 + 1), dtype=np.complex128)

    for frame_index in range(frame_count):
        start = frame_index * hop_size
        frame = padded[start : start + fft_size]
        if len(frame) < fft_size:
            frame = np.pad(frame, (0, fft_size - len(frame)))
        frames[frame_index] = np.fft.rfft(frame * window)

    return frames, window


def istft_audio(frames: np.ndarray, window: np.ndarray, hop_size: int, output_length: int) -> np.ndarray:
    """Overlap-add inverse STFT using the original output length."""
    fft_size = (frames.shape[1] - 1) * 2
    pad = fft_size // 2
    total_length = (len(frames) - 1) * hop_size + fft_size
    audio = np.zeros(total_length, dtype=np.float64)
    window_sum = np.zeros(total_length, dtype=np.float64)

    for frame_index, spectrum in enumerate(frames):
        start = frame_index * hop_size
        frame = np.fft.irfft(spectrum, n=fft_size) * window
        audio[start : start + fft_size] += frame
        window_sum[start : start + fft_size] += window**2

    valid = window_sum > 1e-8
    audio[valid] /= window_sum[valid]
    return audio[pad : pad + output_length]


def smooth_gain_frames(gain_db: np.ndarray, smoothing_bins: int) -> np.ndarray:
    if smoothing_bins <= 1:
        return gain_db

    smoothed = np.zeros_like(gain_db)
    for frame_index in range(gain_db.shape[0]):
        smoothed[frame_index] = smooth_gain_db(gain_db[frame_index], smoothing_bins=smoothing_bins)
    return smoothed


def render_amp_style_source(
    audio: np.ndarray,
    sample_rate: int,
    amp_style: str,
    input_gain: float,
    drive: float,
    bias: float,
    sag: float,
    compression: float,
    drive_boost: float,
) -> np.ndarray:
    """Create the source that will be tone-matched to the real mic target."""
    style = amp_style.lower()
    if style == "mic-layer":
        shaped = fft_tone_filter(
            audio,
            sample_rate,
            points=[
                (20.0, -36.0),
                (70.0, -14.0),
                (110.0, -5.0),
                (220.0, -1.5),
                (650.0, 0.0),
                (1400.0, 0.8),
                (2800.0, 0.4),
                (5200.0, -8.0),
                (8000.0, -22.0),
                (sample_rate / 2.0, -54.0),
            ],
        )
        return normalize_basis_signal(shaped)

    if style == "neutral":
        return apply_dynamic_nonlinearity(
            audio,
            sample_rate,
            input_gain=input_gain * drive_boost,
            drive=drive,
            bias=bias,
            sag=sag,
            compression=compression,
        )

    if style in {"high-gain", "6505"}:
        low_tightness = -12.0 if style == "6505" else -8.0
        upper_mid_push = 4.0 if style == "6505" else 2.5
        pre = fft_tone_filter(
            audio,
            sample_rate,
            points=[
                (20.0, -30.0),
                (70.0, low_tightness),
                (110.0, -4.5),
                (220.0, -0.5),
                (520.0, 1.8),
                (950.0, 1.0),
                (1700.0, upper_mid_push),
                (3200.0, 2.5),
                (6200.0, -7.0),
                (9800.0, -22.0),
                (sample_rate / 2.0, -48.0),
            ],
        )
        first_stage = apply_dynamic_nonlinearity(
            pre,
            sample_rate,
            input_gain=max(4.0, input_gain * 3.8 * drive_boost),
            drive=max(7.5, drive * 1.25),
            bias=-0.035 if style == "6505" else -0.015,
            sag=max(0.18, sag),
            compression=max(0.36, compression),
        )
        post = fft_tone_filter(
            first_stage,
            sample_rate,
            points=[
                (20.0, -42.0),
                (80.0, -14.0),
                (120.0, -5.0),
                (180.0, 0.0),
                (430.0, 1.2),
                (900.0, 0.4),
                (1600.0, 2.2),
                (2800.0, 3.8 if style == "6505" else 2.4),
                (4700.0, -1.0),
                (6400.0, -12.0),
                (9000.0, -28.0),
                (sample_rate / 2.0, -54.0),
            ],
        )
        second_stage = apply_dynamic_nonlinearity(
            post,
            sample_rate,
            input_gain=1.55 if style == "6505" else 1.25,
            drive=4.4 if style == "6505" else 3.4,
            bias=0.025,
            sag=max(0.20, sag * 0.75),
            compression=max(0.42, compression),
        )
        return normalize_peak(soft_limiter(second_stage), peak=0.90)

    raise ValueError(f"Unsupported amp style: {amp_style}")


def render_spectral_tone_match(
    di_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    profile: dict | None = None,
    fft_size: int = 8192,
    smoothing_bins: int = 91,
    amp_style: str = "neutral",
    drive_boost: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """
    Render an obvious amp/mic-colored audition from DI using target spectral shape.

    This is a robust fallback when the deeper phase-sensitive capture has weak
    validation on a real mic take.
    """
    di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
    target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
    di_aligned, target_aligned, lag, polarity = align_pair(
        di_audio,
        target_audio,
        max_lag_s=0.05,
        sample_rate=sample_rate,
    )

    if profile:
        nonlinear = profile["nonlinear"]
        input_gain = max(2.0, float(nonlinear["input_gain"]))
        drive = max(4.0, float(nonlinear["drive"]))
        bias = float(nonlinear["bias"])
        sag = max(0.20, float(nonlinear.get("sag", 0.0)))
        compression = max(0.20, float(nonlinear.get("compression", 0.0)))
    else:
        input_gain = 2.4
        drive = 5.2
        bias = 0.0
        sag = 0.28
        compression = 0.30

    driven = render_amp_style_source(
        di_aligned,
        sample_rate,
        amp_style=amp_style,
        input_gain=input_gain,
        drive=drive,
        bias=bias,
        sag=sag,
        compression=compression,
        drive_boost=drive_boost,
    )
    freqs_norm, source_power = average_power_spectrum(driven, fft_size=fft_size)
    _, target_power = average_power_spectrum(target_aligned, fft_size=fft_size)
    curve_freqs_hz = freqs_norm * sample_rate

    gain_db = 10.0 * np.log10((target_power + 1e-10) / (source_power + 1e-10))
    gain_db = smooth_gain_db(gain_db, smoothing_bins=smoothing_bins)
    gain_db = np.clip(gain_db, -30.0, 24.0)

    matched = apply_frequency_gain_curve(driven, sample_rate, curve_freqs_hz, gain_db)
    matched *= rms(target_aligned) / (rms(matched) + 1e-12)
    matched = normalize_for_audition(soft_limiter(matched), peak=0.86)

    min_len = min(len(target_aligned), len(matched))
    metrics = {
        "alignment_lag_samples": int(lag),
        "target_polarity_after_alignment": int(polarity),
        "render_mode": "spectral_tone_match_audition",
        "amp_style": amp_style,
        "drive_boost": float(drive_boost),
        "fft_size": int(fft_size),
        "smoothing_bins": int(smoothing_bins),
        "match_correlation": correlation(target_aligned[:min_len], matched[:min_len]),
        "spectral_error_db": spectral_error_db(target_aligned[:min_len], matched[:min_len], sample_rate),
        "audition_peak_dbfs": peak_dbfs(matched),
        "note": "Audible tone-match fallback; not a saved reusable nonlinear profile by itself.",
    }
    return matched, metrics


def normalize_basis_signal(audio: np.ndarray) -> np.ndarray:
    audio = remove_dc(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0))
    peak = float(np.max(np.abs(audio)) + 1e-12)
    return audio / peak


def build_hammerstein_basis(audio: np.ndarray, basis_names: list[str] | None = None) -> list[tuple[str, np.ndarray]]:
    """
    Build nonlinear basis signals for a mic-learned amp/cab model.

    A parallel Hammerstein model lets the target mic recording determine how
    each nonlinear version of the DI is filtered and mixed.
    """
    x = normalize_basis_signal(audio)
    all_basis = {
        "linear": x,
        "soft_drive_2": apply_saturation(x, drive=2.0, bias=0.0),
        "soft_drive_5": apply_saturation(x, drive=5.0, bias=0.0),
        "soft_drive_9": apply_saturation(x, drive=9.0, bias=0.0),
        "asym_drive_neg": apply_saturation(x, drive=6.5, bias=-0.045),
        "asym_drive_pos": apply_saturation(x, drive=6.5, bias=0.045),
        "even_harmonic": normalize_basis_signal(x * np.abs(x)),
        "odd_cubic": normalize_basis_signal(x**3),
    }

    selected_names = basis_names or list(all_basis)
    return [(name, normalize_basis_signal(all_basis[name])) for name in selected_names]


def estimate_parallel_hammerstein_model(
    di_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    ir_ms: float = 96.0,
    regularization: float = 0.02,
) -> tuple[dict, np.ndarray, dict]:
    """
    Learn nonlinear filters directly from a clean DI and SM57 target recording.

    This is more target-driven than the synthetic 6505 audition mode: each basis
    branch is estimated from the mic recording with regularized deconvolution.
    """
    di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
    target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
    di_aligned, target_aligned, lag, polarity = align_pair(
        di_audio,
        target_audio,
        max_lag_s=0.05,
        sample_rate=sample_rate,
    )

    ir_samples = max(512, int(round(sample_rate * ir_ms / 1000.0)))
    basis = build_hammerstein_basis(di_aligned)
    nfft = next_power_of_two(len(di_aligned) + ir_samples - 1)
    basis_fft = [np.fft.rfft(signal, nfft) for _, signal in basis]
    target_fft = np.fft.rfft(target_aligned, nfft)

    denom = np.zeros_like(target_fft.real)
    for spectrum in basis_fft:
        denom += np.abs(spectrum) ** 2

    reg = regularization * float(np.max(denom) + 1e-12)
    window = np.hanning(ir_samples * 2)[ir_samples:]
    filters = []
    reconstructed = np.zeros_like(target_aligned)

    for (name, signal), spectrum in zip(basis, basis_fft):
        transfer = target_fft * np.conj(spectrum) / (denom + reg)
        impulse = np.fft.irfft(transfer, nfft)[:ir_samples]
        impulse = remove_dc(impulse * window)
        filters.append((name, impulse.astype(np.float64)))
        reconstructed += fftconvolve(signal, impulse, mode="full")[: len(target_aligned)]

    output_gain = estimate_gain(reconstructed, target_aligned)
    reconstructed *= output_gain
    reconstructed = normalize_for_audition(soft_limiter(reconstructed), peak=0.86)

    min_len = min(len(target_aligned), len(reconstructed))
    metrics = {
        "match_rmse": rms(target_aligned[:min_len] - reconstructed[:min_len]),
        "match_correlation": correlation(target_aligned[:min_len], reconstructed[:min_len]),
        "spectral_error_db": spectral_error_db(target_aligned[:min_len], reconstructed[:min_len], sample_rate),
        "audition_peak_dbfs": peak_dbfs(reconstructed),
    }
    model = {
        "model_version": "parallel_hammerstein_mic_capture_1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rate_hz": int(sample_rate),
        "alignment_lag_samples": int(lag),
        "target_polarity_after_alignment": int(polarity),
        "ir_ms": float(ir_ms),
        "ir_samples": int(ir_samples),
        "regularization": float(regularization),
        "basis_names": [name for name, _ in filters],
        "filters": {
            name: [round(float(value), 9) for value in impulse]
            for name, impulse in filters
        },
        "output_gain": float(output_gain),
        "validation": metrics,
        "portfolio_note": "Mic-learned nonlinear DI-to-SM57 prototype using a parallel Hammerstein DSP model.",
    }
    return model, reconstructed, metrics


def apply_parallel_hammerstein_model(audio: np.ndarray, sample_rate: int, model: dict) -> tuple[int, np.ndarray]:
    model_rate = int(model["sample_rate_hz"])
    audio = resample_if_needed(audio, sample_rate, model_rate)
    audio = normalize_peak(remove_dc(audio), peak=0.95)
    basis = build_hammerstein_basis(audio, basis_names=list(model["basis_names"]))
    output = np.zeros_like(audio)

    for name, signal in basis:
        impulse = np.array(model["filters"][name], dtype=np.float64)
        output += fftconvolve(signal, impulse, mode="full")[: len(audio)]

    output *= float(model["output_gain"])
    return model_rate, normalize_for_audition(soft_limiter(output), peak=0.86)


def require_mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise SystemExit(
            "The optional MLX layer needs Apple's 'mlx' package.\n"
            "Install it inside this project environment only with:\n"
            "  python3 -m pip install -r requirements-mlx.txt\n\n"
            "The regular DSP commands do not need MLX."
        ) from exc

    try:
        probe = mx.array([0.0])
        mx.eval(probe)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            raise SystemExit(
                "MLX is installed, but this process cannot access the Mac's Metal device.\n"
                "Run the same command directly in your normal Terminal window from this project venv.\n"
                "The DSP-only commands still work here."
            ) from exc
        raise

    return mx


def build_context_features(
    di_audio: np.ndarray,
    base_audio: np.ndarray,
    sample_indices: np.ndarray,
    context_radius: int,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    """Build local sample-window features for neural residual tone matching."""
    if context_radius < 1:
        raise ValueError("context_radius must be at least 1")

    di_pad = np.pad(di_audio.astype(np.float32), context_radius, mode="reflect")
    base_pad = np.pad(base_audio.astype(np.float32), context_radius, mode="reflect")

    columns = []
    for offset in range(context_radius * 2 + 1):
        columns.append(base_pad[sample_indices + offset])
    for offset in range(context_radius * 2 + 1):
        columns.append(di_pad[sample_indices + offset])

    center = context_radius
    di_center = di_pad[sample_indices + center]
    base_center = base_pad[sample_indices + center]
    columns.extend(
        [
            np.abs(di_center),
            np.abs(base_center),
            di_center * base_center,
        ]
    )

    features = np.stack(columns, axis=1).astype(np.float32)
    if feature_mean is not None and feature_std is not None:
        features = (features - feature_mean) / np.maximum(feature_std, 1e-6)
    return features


def build_amp_window_features(
    di_audio: np.ndarray,
    sample_indices: np.ndarray,
    context_radius: int,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
    conditioning_features: np.ndarray | None = None,
    input_scale: float | None = None,
    prepared_audio: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build nonlinear waveform features for a direct neural amp model.

    Unlike spectrum-only matching, this gives MLX local waveform shape so it can
    learn clipping, compression, and pick-attack behavior from the mic target.
    """
    if context_radius < 1:
        raise ValueError("context_radius must be at least 1")

    if prepared_audio is not None:
        x = np.asarray(prepared_audio, dtype=np.float32)
    elif input_scale is None:
        x = normalize_basis_signal(di_audio).astype(np.float32)
    else:
        safe_scale = max(float(input_scale), 1e-6)
        x = np.clip(remove_dc(di_audio.astype(np.float64)) / safe_scale, -4.0, 4.0).astype(np.float32)
    window_columns = []
    safe_direct = bool(
        len(sample_indices)
        and int(np.min(sample_indices)) >= context_radius
        and int(np.max(sample_indices)) < len(x) - context_radius
    )
    if safe_direct:
        for offset in range(-context_radius, context_radius + 1):
            window_columns.append(x[sample_indices + offset])
    else:
        padded = np.pad(x, context_radius, mode="reflect")
        for offset in range(context_radius * 2 + 1):
            window_columns.append(padded[sample_indices + offset])

    window = np.stack(window_columns, axis=1).astype(np.float32)
    center = window[:, context_radius]
    features = np.concatenate(
        [
            window,
            np.abs(window),
            np.square(window),
            center[:, None],
            np.abs(center)[:, None],
            np.square(center)[:, None],
            np.power(center, 3)[:, None],
        ],
        axis=1,
    ).astype(np.float32)

    if conditioning_features is not None and len(conditioning_features):
        conditioning = np.asarray(conditioning_features, dtype=np.float32).reshape(1, -1)
        conditioning = np.repeat(conditioning, features.shape[0], axis=0)
        features = np.concatenate([features, conditioning], axis=1).astype(np.float32)

    if feature_mean is not None and feature_std is not None:
        features = (features - feature_mean) / np.maximum(feature_std, 1e-6)
    return features


AMP_SOURCE_CONDITIONING_FEATURES = [
    "rms",
    "peak",
    "crest_factor",
    "spectral_centroid_ratio",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "transient_rms_ratio",
    "p95_abs",
    "p99_abs",
]

AMP_MIC_POSITION_FEATURES = [
    "mic_grille_touch",
    "mic_close",
    "mic_distance_norm",
    "mic_center",
    "mic_edge",
    "mic_on_axis",
    "mic_off_axis",
]


def amp_source_conditioning_features(
    di_audio: np.ndarray,
    sample_rate: int,
    input_scale: float | None = None,
) -> np.ndarray:
    """Summarize the DI source so the amp model can adapt across guitars/pickups."""
    centered = remove_dc(di_audio.astype(np.float64))
    if input_scale is None:
        audio = normalize_basis_signal(centered)
    else:
        audio = np.clip(centered / max(float(input_scale), 1e-6), -4.0, 4.0)
    abs_audio = np.abs(audio)
    rms_value = rms(audio)
    peak_value = float(np.max(abs_audio) + 1e-12)
    crest = float(np.clip(peak_value / (rms_value + 1e-12), 0.0, 32.0))
    features = tone_features(audio, sample_rate)
    diff = np.diff(audio, prepend=audio[0])
    transient_ratio = float(np.clip(rms(diff) / (rms_value + 1e-12), 0.0, 32.0))

    return np.array(
        [
            rms_value,
            peak_value,
            crest,
            float(features["spectral_centroid_hz"] / max(1.0, sample_rate / 2.0)),
            float(features["low_energy_ratio"]),
            float(features["mid_energy_ratio"]),
            float(features["high_energy_ratio"]),
            transient_ratio,
            float(np.percentile(abs_audio, 95.0)),
            float(np.percentile(abs_audio, 99.0)),
        ],
        dtype=np.float32,
    )


def mic_position_conditioning_features(mic_position: str | None) -> np.ndarray:
    """Encode coarse SM57 placement notes as numeric conditioning features."""
    text = str(mic_position or "").strip().lower()
    grille_touch = any(token in text for token in ("grille", "grill", "touch"))
    close = grille_touch or "close" in text or "directly" in text or "front" in text
    one_cm = any(token in text for token in ("1 cm", "1cm", "centimeter", "centimetre"))
    center = any(token in text for token in ("center", "centre", "directly in front"))
    edge = "edge" in text or "cap edge" in text
    off_axis = any(token in text for token in ("off-axis", "off axis", "angled", "angle"))
    on_axis = not off_axis and (center or "on-axis" in text or "on axis" in text or close)

    if grille_touch:
        distance_norm = 0.0
    elif one_cm:
        distance_norm = 0.10
    elif close:
        distance_norm = 0.18
    elif text:
        distance_norm = 0.45
    else:
        distance_norm = 0.25

    return np.array(
        [
            float(grille_touch),
            float(close),
            float(distance_norm),
            float(center),
            float(edge),
            float(on_axis),
            float(off_axis),
        ],
        dtype=np.float32,
    )


def amp_conditioning_features(
    di_audio: np.ndarray,
    sample_rate: int,
    conditioning_mode: str,
    mic_position: str | None = None,
    include_mic_position: bool = False,
    input_scale: float | None = None,
    rig_fingerprint_value: str | None = None,
    known_rig_fingerprints: list[str] | None = None,
) -> np.ndarray:
    parts = []
    if conditioning_mode == "source-stats":
        parts.append(amp_source_conditioning_features(di_audio, sample_rate, input_scale=input_scale))
        if include_mic_position:
            parts.append(mic_position_conditioning_features(mic_position))
    known_rigs = list(known_rig_fingerprints or [])
    if known_rigs:
        one_hot = np.zeros(len(known_rigs), dtype=np.float32)
        requested = str(rig_fingerprint_value or "")
        if requested in known_rigs:
            one_hot[known_rigs.index(requested)] = 1.0
        parts.append(one_hot)
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def init_mlx_mlp_params(mx, input_dim: int, hidden_dim: int, seed: int, output_dim: int = 1) -> dict:
    rng = np.random.default_rng(seed)

    def weight(in_dim: int, out_dim: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(1, in_dim))
        return rng.normal(0.0, scale, (in_dim, out_dim)).astype(np.float32)

    return {
        "w1": mx.array(weight(input_dim, hidden_dim)),
        "b1": mx.zeros((hidden_dim,), dtype=mx.float32),
        "w2": mx.array(weight(hidden_dim, hidden_dim)),
        "b2": mx.zeros((hidden_dim,), dtype=mx.float32),
        "w3": mx.array(weight(hidden_dim, output_dim)),
        "b3": mx.zeros((output_dim,), dtype=mx.float32),
    }


def mlx_mlp_forward(mx, params: dict, x):
    hidden_1 = mx.tanh((x @ params["w1"]) + params["b1"])
    hidden_2 = mx.tanh((hidden_1 @ params["w2"]) + params["b2"])
    return (hidden_2 @ params["w3"]) + params["b3"]


def mlx_mse_loss(mx, params: dict, x, y):
    prediction = mlx_mlp_forward(mx, params, x)
    return mx.mean(mx.square(prediction - y))


def mlx_huber_mean(mx, error, delta: float):
    absolute = mx.abs(error)
    quadratic = mx.minimum(absolute, float(delta))
    linear = absolute - quadratic
    return mx.mean((0.5 * mx.square(quadratic)) + (float(delta) * linear))


def mlx_amp_detail_loss_from_prediction(
    mx,
    prediction,
    y,
    transient_weight: float,
    highfreq_weight: float,
    envelope_weight: float,
    robust_delta: float = 0.08,
):
    waveform_loss = mlx_huber_mean(mx, prediction - y, robust_delta)
    envelope_loss = mlx_huber_mean(mx, mx.abs(prediction) - mx.abs(y), robust_delta)

    transient_loss = mlx_huber_mean(
        mx,
        (prediction[1:] - prediction[:-1]) - (y[1:] - y[:-1]),
        robust_delta,
    )
    highfreq_loss = mlx_huber_mean(
        mx,
        (prediction[1:] - 0.97 * prediction[:-1]) - (y[1:] - 0.97 * y[:-1]),
        robust_delta,
    )

    if y.shape[0] > 8:
        transient_loss = transient_loss + 0.50 * mlx_huber_mean(
            mx,
            (prediction[4:] - prediction[:-4]) - (y[4:] - y[:-4]),
            robust_delta,
        )
    if y.shape[0] > 32:
        transient_loss = transient_loss + 0.25 * mlx_huber_mean(
            mx,
            (prediction[16:] - prediction[:-16]) - (y[16:] - y[:-16]),
            robust_delta,
        )

    return (
        waveform_loss
        + transient_weight * transient_loss
        + highfreq_weight * highfreq_loss
        + envelope_weight * envelope_loss
    )


def mlx_amp_detail_loss(
    mx,
    params: dict,
    x,
    y,
    transient_weight: float,
    highfreq_weight: float,
    envelope_weight: float,
    robust_delta: float = 0.08,
):
    """
    Detail-aware amp loss for contiguous waveform chunks.

    The waveform term learns the core tone, the derivative terms preserve pick
    attack and mids, the pre-emphasis term pushes upper-frequency detail, and
    the envelope term helps match compression/sustain behavior.
    """
    prediction = mlx_mlp_forward(mx, params, x)
    return mlx_amp_detail_loss_from_prediction(
        mx,
        prediction,
        y,
        transient_weight=transient_weight,
        highfreq_weight=highfreq_weight,
        envelope_weight=envelope_weight,
        robust_delta=robust_delta,
    )


def mlx_amp_detail_spectral_loss(
    mx,
    params: dict,
    x,
    y,
    transient_weight: float,
    highfreq_weight: float,
    envelope_weight: float,
    esr_weight: float,
    spectral_weight: float,
    robust_delta: float = 0.08,
):
    """
    Amp-focused detail loss.

    ESR normalizes waveform error against the target energy, while the log-FFT
    terms make missing cab/SM57 spectral shape visible to the optimizer.
    """
    prediction = mlx_mlp_forward(mx, params, x)
    detail_loss = mlx_amp_detail_loss_from_prediction(
        mx,
        prediction,
        y,
        transient_weight=transient_weight,
        highfreq_weight=highfreq_weight,
        envelope_weight=envelope_weight,
        robust_delta=robust_delta,
    )

    error_energy = mx.mean(mx.square(prediction - y))
    target_energy = mx.mean(mx.square(y)) + 1e-7
    esr_loss = mx.minimum(error_energy / target_energy, mx.array(25.0))

    pred_1d = prediction.reshape((-1,))
    target_1d = y.reshape((-1,))
    chunk_len = int(y.shape[0])
    fft_losses = []
    for fft_size in (256, 512, 1024, 2048):
        if chunk_len < fft_size:
            continue
        window = mx.array(np.hanning(fft_size).astype(np.float32))
        pred_frame = pred_1d[:fft_size] * window
        target_frame = target_1d[:fft_size] * window
        pred_mag = mx.abs(mx.fft.rfft(pred_frame)) + 1e-5
        target_mag = mx.abs(mx.fft.rfft(target_frame)) + 1e-5
        log_delta = mx.clip(mx.log(pred_mag) - mx.log(target_mag), -6.0, 6.0)
        log_mag_loss = mlx_huber_mean(mx, log_delta, max(0.25, robust_delta * 4.0))
        spectral_convergence = mx.minimum(
            mx.mean(mx.abs(pred_mag - target_mag)) / (mx.mean(mx.abs(target_mag)) + 1e-5),
            mx.array(20.0),
        )
        fft_losses.append(log_mag_loss + 0.35 * spectral_convergence)

    if fft_losses:
        spectral_loss = fft_losses[0]
        for item in fft_losses[1:]:
            spectral_loss = spectral_loss + item
        spectral_loss = spectral_loss / len(fft_losses)
    else:
        spectral_loss = error_energy

    return detail_loss + (esr_weight * esr_loss) + (spectral_weight * spectral_loss)


def mlx_tree_zeros_like(mx, params: dict) -> dict:
    return {key: mx.zeros_like(value) for key, value in params.items()}


def mlx_clip_gradient_tree(mx, grads: dict, max_norm: float) -> tuple[dict, float]:
    if max_norm <= 0.0:
        return grads, 0.0
    total = None
    for value in grads.values():
        item = mx.sum(mx.square(value))
        total = item if total is None else total + item
    norm = mx.sqrt(total + 1e-12)
    scale = mx.minimum(mx.array(1.0), float(max_norm) / (norm + 1e-12))
    clipped = {key: value * scale for key, value in grads.items()}
    mx.eval(norm)
    return clipped, float(norm.item())


def mlx_adam_update(
    mx,
    params: dict,
    grads: dict,
    first_moment: dict,
    second_moment: dict,
    step: int,
    learning_rate: float,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[dict, dict, dict]:
    next_params = {}
    next_first = {}
    next_second = {}

    for key, value in params.items():
        grad = grads[key]
        m = (beta_1 * first_moment[key]) + ((1.0 - beta_1) * grad)
        v = (beta_2 * second_moment[key]) + ((1.0 - beta_2) * mx.square(grad))
        m_hat = m / (1.0 - (beta_1**step))
        v_hat = v / (1.0 - (beta_2**step))
        next_params[key] = value - learning_rate * m_hat / (mx.sqrt(v_hat) + epsilon)
        next_first[key] = m
        next_second[key] = v

    return next_params, next_first, next_second


def load_mlx_residual_model(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item()))
    if metadata.get("model_version") != MLX_MODEL_VERSION:
        raise SystemExit(f"Unsupported MLX model version in {path}: {metadata.get('model_version')}")

    params = {
        "w1": data["w1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "w2": data["w2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
        "w3": data["w3"].astype(np.float32),
        "b3": data["b3"].astype(np.float32),
    }
    return metadata, params


def save_mlx_residual_model(path: Path, metadata: dict, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    np.savez(path, metadata=json.dumps(metadata, indent=2), **arrays)


def load_mlx_spectral_model(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item()))
    if metadata.get("model_version") != MLX_SPECTRAL_MODEL_VERSION:
        raise SystemExit(f"Unsupported MLX spectral model version in {path}: {metadata.get('model_version')}")

    params = {
        "w1": data["w1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "w2": data["w2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
        "w3": data["w3"].astype(np.float32),
        "b3": data["b3"].astype(np.float32),
    }
    return metadata, params


def save_mlx_spectral_model(path: Path, metadata: dict, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    np.savez(path, metadata=json.dumps(metadata, indent=2), **arrays)


def load_mlx_amp_model(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"].item()))
    version = str(metadata.get("model_version", ""))
    if version != MLX_AMP_MODEL_VERSION and version not in MLX_AMP_LEGACY_MODEL_VERSIONS:
        raise SystemExit(f"Unsupported MLX amp model version in {path}: {metadata.get('model_version')}")

    params = {
        "w1": data["w1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "w2": data["w2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
        "w3": data["w3"].astype(np.float32),
        "b3": data["b3"].astype(np.float32),
    }
    return metadata, params


def save_mlx_amp_model(path: Path, metadata: dict, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    np.savez(path, metadata=json.dumps(metadata, indent=2), **arrays)


def predict_mlx_residual(
    di_audio: np.ndarray,
    base_audio: np.ndarray,
    metadata: dict,
    params: dict,
    chunk_samples: int = 65536,
) -> np.ndarray:
    mx = require_mlx()
    params_mx = {key: mx.array(value.astype(np.float32)) for key, value in params.items()}

    context_radius = int(metadata["context_radius"])
    feature_mean = np.array(metadata["feature_mean"], dtype=np.float32)
    feature_std = np.array(metadata["feature_std"], dtype=np.float32)
    residual_scale = float(metadata["residual_scale"])
    residual_mix = float(metadata.get("residual_mix", 1.0))
    residual = np.zeros(len(base_audio), dtype=np.float32)

    for start in range(0, len(base_audio), chunk_samples):
        end = min(len(base_audio), start + chunk_samples)
        indices = np.arange(start, end, dtype=np.int64)
        features = build_context_features(
            di_audio,
            base_audio,
            indices,
            context_radius=context_radius,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        prediction = mlx_mlp_forward(mx, params_mx, mx.array(features))
        mx.eval(prediction)
        residual[start:end] = np.array(prediction).reshape(-1).astype(np.float32)

    return residual.astype(np.float64) * residual_scale * residual_mix


def apply_cabinet_guard_filter(
    audio: np.ndarray,
    sample_rate: int,
    lowpass_hz: float = 6500.0,
    highpass_hz: float = 75.0,
    presence_db: float = 2.0,
    air_db: float = 0.0,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    lowpass_hz = min(max(lowpass_hz, 1500.0), nyquist)
    highpass_hz = min(max(highpass_hz, 20.0), lowpass_hz * 0.75)
    presence_db = float(np.clip(presence_db, -6.0, 8.0))
    air_db = float(np.clip(air_db, -6.0, 6.0))
    points = [
        (20.0, -48.0),
        (highpass_hz * 0.70, -18.0),
        (highpass_hz, -3.0),
        (140.0, 0.0),
        (600.0, 0.0),
        (1400.0, presence_db * 0.25),
        (2400.0, presence_db * 0.85),
        (3600.0, presence_db),
        (4800.0, presence_db * 0.55),
        (lowpass_hz, -4.0 + air_db),
        (min(nyquist, lowpass_hz * 1.35), -14.0 + air_db),
        (nyquist, -50.0),
    ]
    points = sorted(points, key=lambda point: point[0])
    return fft_tone_filter(
        audio,
        sample_rate,
        points=points,
    )


def predict_mlx_amp_audio(
    di_audio: np.ndarray,
    metadata: dict,
    params: dict,
    chunk_samples: int = 65536,
    mic_position: str | None = None,
    rig_fingerprint_value: str | None = None,
) -> np.ndarray:
    mx = require_mlx()
    params_mx = {key: mx.array(value.astype(np.float32)) for key, value in params.items()}
    context_radius = int(metadata["context_radius"])
    feature_mean = np.array(metadata["feature_mean"], dtype=np.float32)
    feature_std = np.array(metadata["feature_std"], dtype=np.float32)
    conditioning_mode = str(metadata.get("conditioning_mode", "none"))
    preserve_input_level = bool(metadata.get("preserve_input_level", False))
    input_scale = float(metadata.get("input_scale", 1.0)) if preserve_input_level else None
    known_rig_fingerprints = list(metadata.get("known_rig_fingerprints", []))
    render_mic_position = (
        str(mic_position)
        if mic_position is not None
        else str(metadata.get("default_mic_position", ""))
    )
    conditioning_features = amp_conditioning_features(
        di_audio,
        int(metadata["sample_rate_hz"]),
        conditioning_mode=conditioning_mode,
        mic_position=render_mic_position,
        include_mic_position=bool(metadata.get("mic_position_conditioning", False)),
        input_scale=input_scale,
        rig_fingerprint_value=rig_fingerprint_value,
        known_rig_fingerprints=known_rig_fingerprints,
    )
    target_scale = float(metadata["target_scale"])
    output = np.zeros(len(di_audio), dtype=np.float32)
    if input_scale is None:
        prepared_audio = normalize_basis_signal(di_audio).astype(np.float32)
    else:
        prepared_audio = np.clip(remove_dc(di_audio) / max(float(input_scale), 1e-6), -4.0, 4.0).astype(np.float32)

    for start in range(0, len(di_audio), chunk_samples):
        end = min(len(di_audio), start + chunk_samples)
        indices = np.arange(start, end, dtype=np.int64)
        features = build_amp_window_features(
            di_audio,
            indices,
            context_radius=context_radius,
            feature_mean=feature_mean,
            feature_std=feature_std,
            conditioning_features=conditioning_features,
            input_scale=input_scale,
            prepared_audio=prepared_audio,
        )
        prediction = mlx_mlp_forward(mx, params_mx, mx.array(features))
        mx.eval(prediction)
        output[start:end] = np.array(prediction).reshape(-1).astype(np.float32)

    return output.astype(np.float64) * target_scale


def render_mlx_amp_model(
    di_audio: np.ndarray,
    sample_rate: int,
    model_path: Path,
    chunk_samples: int = 65536,
    cab_lowpass_hz: float | None = None,
    cab_highpass_hz: float | None = None,
    cab_presence_db: float | None = None,
    cab_air_db: float | None = None,
    render_sample_rate: int | None = None,
    model_input_trim_db: float = 0.0,
    render_limiter: str = "soft",
    output_peak_dbfs: float = -1.31,
    mic_position: str | None = None,
    source_hint_path: Path | None = None,
    rig_fingerprint_value: str | None = None,
    cabinet_profile: Path | None = None,
    cabinet_mix: float = 1.0,
    inferred_ir_mix: float = DEFAULT_INFERRED_IR_MIX,
) -> tuple[int, np.ndarray, dict]:
    metadata, params = load_mlx_amp_model(model_path)
    model_rate = int(metadata["sample_rate_hz"])
    output_rate = int(render_sample_rate or model_rate)
    if output_rate <= 0:
        raise SystemExit("--render-sample-rate must be greater than zero.")
    if render_limiter not in {"soft", "off"}:
        raise SystemExit("--render-limiter must be soft or off.")

    audio = resample_if_needed(di_audio, sample_rate, model_rate)
    preserve_input_level = bool(metadata.get("preserve_input_level", False))
    audio = remove_dc(audio)
    if not preserve_input_level:
        audio = normalize_peak(audio, peak=0.95)
    audio *= db_to_linear(model_input_trim_db)
    selected_rig = str(rig_fingerprint_value or "")
    if not selected_rig and source_hint_path is not None:
        source_resolved = Path(source_hint_path).resolve()
        for pair in dict(metadata.get("training", {})).get("training_pairs", []):
            pair_path = Path(str(pair.get("di", "")))
            if pair_path.exists() and pair_path.resolve() == source_resolved:
                selected_rig = str(pair.get("rig_fingerprint", ""))
                break
    if not selected_rig:
        selected_rig = str(metadata.get("default_rig_fingerprint", ""))
    known_rigs = list(metadata.get("known_rig_fingerprints", []))
    if known_rigs and selected_rig not in known_rigs:
        raise SystemExit(
            f"Rig fingerprint {selected_rig or '<none>'} is not available in this model. "
            f"Choose one of: {', '.join(known_rigs)}"
        )
    metadata["last_selected_rig_fingerprint"] = selected_rig or None
    render_mic_position = (
        str(mic_position)
        if mic_position is not None
        else str(metadata.get("default_mic_position", ""))
    )
    predicted = predict_mlx_amp_audio(
        di_audio=audio,
        metadata=metadata,
        params=params,
        chunk_samples=chunk_samples,
        mic_position=render_mic_position,
        rig_fingerprint_value=selected_rig,
    )
    amp_anchor = amp_tone_anchor_for_metadata(metadata)
    if amp_anchor.get("enabled", False):
        metadata["amp_tone_anchor"] = amp_anchor
    matched_transfer = (
        source_matched_amp_transfer(
            amp_anchor,
            audio,
            model_rate,
            source_hint_path=source_hint_path,
            rig_fingerprint_value=selected_rig,
            mic_position=render_mic_position,
        )
        if amp_anchor.get("enabled", False)
        else {"enabled": False}
    )
    if matched_transfer.get("enabled", False):
        metadata["last_source_matched_transfer"] = {
            "mode": str(matched_transfer.get("mode", "")),
            "top_matches": list(matched_transfer.get("top_matches", [])),
            "top_segments": list(matched_transfer.get("top_segments", [])),
        }
    predicted = apply_cabinet_guard_filter(
        predicted,
        model_rate,
        lowpass_hz=float(cab_lowpass_hz if cab_lowpass_hz is not None else metadata.get("cab_lowpass_hz", 6500.0)),
        highpass_hz=float(cab_highpass_hz if cab_highpass_hz is not None else metadata.get("cab_highpass_hz", 75.0)),
        presence_db=float(cab_presence_db if cab_presence_db is not None else metadata.get("cab_presence_db", 2.0)),
        air_db=float(cab_air_db if cab_air_db is not None else metadata.get("cab_air_db", 0.0)),
    )
    predicted = apply_all_recordings_amp_tone_anchor(
        audio=predicted,
        source_di=audio,
        sample_rate=model_rate,
        metadata=metadata,
        source_hint_path=source_hint_path,
        rig_fingerprint_value=selected_rig,
        mic_position=render_mic_position,
    )
    if matched_transfer.get("enabled", False):
        predicted, layer_diagnostics = blend_hammerstein_amp_layer(
            base_audio=predicted,
            source_di=audio,
            sample_rate=model_rate,
            matched_transfer=matched_transfer,
            mix_override=float(inferred_ir_mix),
        )
        predicted = apply_source_matched_band_balance(
            audio=predicted,
            source_di=audio,
            sample_rate=model_rate,
            matched_transfer=matched_transfer,
            strength=0.78,
        )
        metadata["last_hammerstein_layer"] = layer_diagnostics
    predicted, _ = match_amp_dynamic_behavior(
        predicted,
        reference=None,
        anchor=matched_transfer if matched_transfer.get("enabled", False) else amp_anchor,
    )
    if matched_transfer.get("enabled", False):
        predicted = apply_source_matched_band_balance(
            audio=predicted,
            source_di=audio,
            sample_rate=model_rate,
            matched_transfer=matched_transfer,
            strength=0.52,
        )
        envelope_profile = dict(matched_transfer.get("target_local_envelope_profile", {}))
        if float(envelope_profile.get("spread_db", 0.0)) > 0.0:
            initial_envelope = local_envelope_profile(predicted, model_rate)
            envelope_diagnostics = {}
            for _ in range(2):
                predicted, envelope_diagnostics = reshape_local_envelope(
                    predicted,
                    model_rate,
                    target_spread_db=float(envelope_profile["spread_db"]),
                    strength=0.92,
                )
            predicted, _ = match_amp_dynamic_behavior(predicted, reference=None, anchor=matched_transfer)
            envelope_diagnostics["before"] = initial_envelope
            envelope_diagnostics["after"] = local_envelope_profile(predicted, model_rate)
            metadata["last_local_envelope_match"] = envelope_diagnostics
        tone_profile = dict(matched_transfer.get("target_detailed_tone_profile", {}))
        if tone_profile.get("bands_hz") and tone_profile.get("energy_ratios"):
            predicted, tone_diagnostics = match_detailed_tone_profile(
                predicted,
                model_rate,
                tone_profile,
                iterations=3,
                max_step_db=3.0,
            )
            metadata["last_detailed_tone_match"] = tone_diagnostics
            if float(envelope_profile.get("spread_db", 0.0)) > 0.0:
                predicted, final_envelope = reshape_local_envelope(
                    predicted,
                    model_rate,
                    target_spread_db=float(envelope_profile["spread_db"]),
                    strength=0.55,
                )
                predicted, _ = match_amp_dynamic_behavior(predicted, reference=None, anchor=matched_transfer)
                if metadata.get("last_local_envelope_match"):
                    metadata["last_local_envelope_match"]["after"] = local_envelope_profile(
                        predicted,
                        model_rate,
                    )
                    metadata["last_local_envelope_match"]["final_tone_pass_envelope"] = final_envelope
    selected_cabinet_profile = cabinet_profile
    if selected_cabinet_profile is None and metadata.get("hybrid_cabinet_profile"):
        stored_profile = Path(str(metadata["hybrid_cabinet_profile"]))
        if stored_profile.exists():
            selected_cabinet_profile = stored_profile
    if selected_cabinet_profile is not None:
        from cabinet_variant_workflow import apply_cabinet_variant_audio

        predicted, cabinet_metadata = apply_cabinet_variant_audio(
            predicted,
            model_rate,
            Path(selected_cabinet_profile),
            mix=float(cabinet_mix),
        )
        metadata["last_hybrid_cabinet"] = {
            "profile": str(selected_cabinet_profile),
            "mix": float(cabinet_mix),
            "name": str(cabinet_metadata.get("name", "")),
        }
    output_peak = float(np.clip(db_to_linear(output_peak_dbfs), 0.02, 0.98))
    limited = soft_limiter(predicted) if render_limiter == "soft" else remove_dc(predicted)
    if preserve_input_level:
        peak = float(np.max(np.abs(limited)) + 1e-12)
        output = limited * min(1.0, output_peak / peak)
    else:
        output = normalize_for_audition(limited, peak=output_peak)
    output = resample_if_needed(output, model_rate, output_rate)
    output = normalize_for_audition(output, peak=output_peak)
    return output_rate, output, metadata


def predict_mlx_spectral_gain(
    di_frames: np.ndarray,
    metadata: dict,
    params: dict,
    batch_frames: int = 128,
) -> np.ndarray:
    mx = require_mlx()
    params_mx = {key: mx.array(value.astype(np.float32)) for key, value in params.items()}
    feature_mean = np.array(metadata["feature_mean"], dtype=np.float32)
    feature_std = np.array(metadata["feature_std"], dtype=np.float32)
    gain_scale_db = float(metadata["gain_scale_db"])
    max_gain_db = float(metadata["max_gain_db"])

    di_db = 20.0 * np.log10(np.abs(di_frames).astype(np.float32) + 1e-7)
    features = (di_db - feature_mean) / np.maximum(feature_std, 1e-6)
    predicted_gain = np.zeros_like(features, dtype=np.float32)

    for start in range(0, len(features), batch_frames):
        end = min(len(features), start + batch_frames)
        prediction = mlx_mlp_forward(mx, params_mx, mx.array(features[start:end]))
        mx.eval(prediction)
        predicted_gain[start:end] = np.array(prediction, dtype=np.float32)

    predicted_gain_db = predicted_gain.astype(np.float64) * gain_scale_db
    predicted_gain_db = np.clip(predicted_gain_db, -max_gain_db, max_gain_db)
    predicted_gain_db = smooth_gain_frames(
        predicted_gain_db,
        smoothing_bins=int(metadata.get("output_smoothing_bins", 1)),
    )
    return predicted_gain_db


def render_mlx_spectral_bridge(
    di_audio: np.ndarray,
    sample_rate: int,
    model_path: Path,
    batch_frames: int = 128,
) -> tuple[int, np.ndarray, dict]:
    metadata, params = load_mlx_spectral_model(model_path)
    model_rate = int(metadata["sample_rate_hz"])
    audio = resample_if_needed(di_audio, sample_rate, model_rate)
    audio = normalize_peak(remove_dc(audio), peak=0.95)
    fft_size = int(metadata["fft_size"])
    hop_size = int(metadata["hop_size"])
    di_frames, window = stft_audio(audio, fft_size=fft_size, hop_size=hop_size)
    gain_db = predict_mlx_spectral_gain(
        di_frames=di_frames,
        metadata=metadata,
        params=params,
        batch_frames=batch_frames,
    )
    output_frames = di_frames * (10.0 ** (gain_db / 20.0))
    output = istft_audio(output_frames, window=window, hop_size=hop_size, output_length=len(audio))
    output = normalize_for_audition(soft_limiter(output), peak=0.86)
    return model_rate, output, metadata


def render_mlx_enhanced_audio(
    di_audio: np.ndarray,
    sample_rate: int,
    profile: dict,
    model_path: Path,
    chunk_samples: int = 65536,
) -> tuple[int, np.ndarray, dict]:
    metadata, params = load_mlx_residual_model(model_path)
    profile_rate = int(profile["sample_rate_hz"])
    model_rate = int(metadata["sample_rate_hz"])
    if model_rate != profile_rate:
        raise SystemExit(
            f"MLX model sample rate ({model_rate}) does not match profile sample rate ({profile_rate})."
        )

    di_audio = resample_if_needed(di_audio, sample_rate, profile_rate)
    base_audio = apply_profile_to_audio(di_audio, profile_rate, profile)
    base_audio *= float(metadata.get("base_gain", 1.0))
    residual = predict_mlx_residual(
        di_audio=di_audio,
        base_audio=base_audio,
        metadata=metadata,
        params=params,
        chunk_samples=chunk_samples,
    )
    enhanced = normalize_peak(soft_limiter(base_audio + residual), peak=0.92)
    return profile_rate, enhanced, metadata


def capture_tone_profile(
    di_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    config: CaptureConfig,
    di_source_name: str,
    target_source_name: str,
    hardware_context: dict | None = None,
) -> CaptureResult:
    """Capture a reusable tone profile from aligned DI and amp target audio."""
    di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
    target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
    di_aligned, target_aligned, lag, target_polarity = align_pair(
        di_audio,
        target_audio,
        max_lag_s=0.05,
        sample_rate=sample_rate,
    )

    ir_samples = max(128, int(round((config.ir_ms / 1000.0) * sample_rate)))
    input_gain = float(np.clip(rms(target_aligned) / (rms(di_aligned) + 1e-12), 0.35, 8.0))

    drive_values = np.array([0.85, 1.15, 1.55, 2.1, 2.8, 3.8, 5.1, 6.8, 9.0], dtype=np.float64)
    bias_values = np.array([-0.055, 0.0, 0.055], dtype=np.float64) if config.search_bias else np.array([0.0])

    best: dict | None = None
    best_dynamic: dict | None = None

    def evaluate_candidate(drive: float, bias: float, sag: float, compression: float) -> None:
        nonlocal best, best_dynamic
        nonlinear_source = apply_dynamic_nonlinearity(
            di_aligned,
            sample_rate,
            input_gain=input_gain,
            drive=drive,
            bias=bias,
            sag=sag,
            compression=compression,
        )
        impulse = estimate_ir_regularized(
            nonlinear_source,
            target_aligned,
            ir_samples=ir_samples,
            regularization=config.regularization,
        )
        reconstructed = fftconvolve(nonlinear_source, impulse, mode="full")[: len(target_aligned)]
        output_gain = estimate_gain(reconstructed, target_aligned)
        reconstructed *= output_gain

        time_error = rms(target_aligned - reconstructed)
        spec_error = spectral_error_db(target_aligned, reconstructed, sample_rate)
        score = time_error + 0.015 * spec_error

        candidate = {
            "score": score,
            "drive": float(drive),
            "bias": float(bias),
            "sag": float(sag),
            "compression": float(compression),
            "impulse": impulse,
            "output_gain": float(output_gain),
            "reconstructed": reconstructed,
            "time_error": float(time_error),
            "spectral_error_db": float(spec_error),
            "selection_note": "lowest_error_candidate",
        }

        if best is None or score < best["score"]:
            best = candidate

        if (sag > 0.0 or compression > 0.0) and (best_dynamic is None or score < best_dynamic["score"]):
            best_dynamic = candidate | {"selection_note": "dynamic_candidate_within_tolerance"}

    for drive in drive_values:
        for bias in bias_values:
            evaluate_candidate(float(drive), float(bias), sag=0.0, compression=0.0)

    assert best is not None
    refined_drives = sorted({float(best["drive"]), float(min(9.0, best["drive"] * 1.35))})
    refined_biases = sorted({float(best["bias"]), 0.0})
    for drive in refined_drives:
        for bias in refined_biases:
            for sag in [0.28, 0.48]:
                for compression in [0.42, 0.65]:
                    evaluate_candidate(drive, bias, sag=sag, compression=compression)

    assert best is not None
    if best_dynamic is not None and best_dynamic["score"] <= best["score"] * 1.18:
        best = best_dynamic

    reconstructed = normalize_peak(best["reconstructed"], peak=0.92)
    match_rmse = rms(target_aligned - reconstructed)
    match_corr = correlation(target_aligned, reconstructed)
    match_spec = spectral_error_db(target_aligned, reconstructed, sample_rate)

    profile = {
        "profile_version": PROFILE_VERSION,
        "name": config.profile_name,
        "instrument": config.instrument,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rate_hz": int(sample_rate),
        "capture_sources": {
            "di": di_source_name,
            "target": target_source_name,
            "alignment_lag_samples": int(lag),
            "target_polarity_after_alignment": int(target_polarity),
        },
        "hardware_context": hardware_context or {
            "mode": "file_capture",
            "note": "No audio interface or DI box metadata was supplied for this capture.",
        },
        "nonlinear": {
            "model": "dynamic_asymmetric_tanh",
            "input_gain": float(input_gain),
            "drive": float(best["drive"]),
            "bias": float(best["bias"]),
            "sag": float(best["sag"]),
            "compression": float(best["compression"]),
            "envelope_follower": {
                "attack_ms": 4.0,
                "release_ms": 85.0,
            },
            "selection_note": str(best["selection_note"]),
        },
        "cabinet": {
            "model": "regularized_deconvolution_fir",
            "ir_samples": int(len(best["impulse"])),
            "ir_ms": float(len(best["impulse"]) * 1000.0 / sample_rate),
            "regularization": float(config.regularization),
            "impulse_response": [round(float(value), 9) for value in best["impulse"]],
        },
        "output_gain": float(best["output_gain"]),
        "tone_features": {
            "di": tone_features(di_aligned, sample_rate),
            "target": tone_features(target_aligned, sample_rate),
            "profiled": tone_features(reconstructed, sample_rate),
        },
        "validation": {
            "match_rmse": float(match_rmse),
            "match_correlation": float(match_corr),
            "spectral_error_db": float(match_spec),
        },
        "portfolio_note": "Prototype tone capture/profile system, not a commercial amp modeling product.",
    }

    return CaptureResult(
        profile=profile,
        reconstructed=reconstructed,
        aligned_di=di_aligned,
        aligned_target=target_aligned,
        match_rmse=float(match_rmse),
        match_correlation=float(match_corr),
        spectral_error_db=float(match_spec),
    )


def save_profile(path: Path, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_profile(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fft_tone_filter(audio: np.ndarray, sample_rate: int, points: list[tuple[float, float]]) -> np.ndarray:
    """Apply an interpolated minimum-phase frequency response in dB."""
    point_freqs = np.array([point[0] for point in points], dtype=np.float64)
    point_db = np.array([point[1] for point in points], dtype=np.float64)
    return apply_frequency_gain_curve(audio, sample_rate, point_freqs, point_db, phase_mode="minimum")


def synthesize_plucked_riff(
    sample_rate: int,
    duration_s: float,
    instrument: str,
    seed: int,
    variant: int = 0,
) -> np.ndarray:
    """Create a synthetic DI-style guitar or bass riff for the demo."""
    rng = np.random.default_rng(seed)
    sample_count = int(round(sample_rate * duration_s))
    audio = np.zeros(sample_count, dtype=np.float64)

    if instrument == "bass":
        base_notes = [41.2, 55.0, 61.7, 73.4, 82.4, 73.4, 61.7, 55.0]
        note_gap_s = 0.58
        decay_s = 0.82
        harmonic_count = 8
    else:
        base_notes = [82.4, 110.0, 146.8, 196.0, 246.9, 196.0, 146.8, 110.0]
        note_gap_s = 0.42
        decay_s = 0.58
        harmonic_count = 12

    if variant:
        base_notes = base_notes[2:] + base_notes[:2]

    for note_index, frequency in enumerate(base_notes * 3):
        start_s = 0.18 + note_index * note_gap_s
        if start_s >= duration_s:
            break

        start = int(round(start_s * sample_rate))
        length = min(sample_count - start, int(round(sample_rate * (note_gap_s + decay_s))))
        if length <= 0:
            continue

        local_t = np.arange(length, dtype=np.float64) / sample_rate
        envelope = np.exp(-local_t / decay_s)
        pick = rng.normal(0.0, 0.05 if instrument == "guitar" else 0.035, length) * np.exp(-local_t / 0.018)
        note = pick

        for harmonic in range(1, harmonic_count + 1):
            harmonic_amp = 1.0 / (harmonic ** (1.08 if instrument == "guitar" else 1.25))
            detune = 1.0 + rng.normal(0.0, 0.00045)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            note += harmonic_amp * np.sin(2.0 * np.pi * frequency * harmonic * detune * local_t + phase)

        note *= envelope
        audio[start : start + length] += note

    audio = fft_tone_filter(
        audio,
        sample_rate,
        points=[
            (20.0, -10.0),
            (70.0, -2.0 if instrument == "bass" else -5.0),
            (180.0, 0.0),
            (1200.0, -1.0),
            (5500.0, -4.0 if instrument == "bass" else 1.5),
            (12000.0, -18.0),
            (sample_rate / 2.0, -30.0),
        ],
    )
    return normalize_peak(audio, peak=0.72)


def make_cabinet_ir(sample_rate: int, instrument: str) -> np.ndarray:
    """Create a demo cabinet-like impulse response."""
    ir_len = int(round(sample_rate * (0.040 if instrument == "guitar" else 0.055)))
    t = np.arange(ir_len, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(88 if instrument == "guitar" else 144)

    if instrument == "bass":
        resonances = [(95.0, 1.0), (180.0, 0.45), (720.0, 0.16), (1800.0, 0.08)]
        decay = np.exp(-t / 0.017)
    else:
        resonances = [(115.0, 0.35), (420.0, 0.50), (1250.0, 0.38), (3100.0, 0.25)]
        decay = np.exp(-t / 0.011)

    impulse = np.zeros(ir_len, dtype=np.float64)
    impulse[0] = 1.0
    for freq, amp in resonances:
        impulse += amp * np.sin(2.0 * np.pi * freq * t + rng.uniform(0.0, np.pi)) * decay

    for delay_ms, amp in [(1.2, -0.22), (2.7, 0.17), (5.4, -0.10), (9.5, 0.06)]:
        index = int(round(delay_ms * sample_rate / 1000.0))
        if index < ir_len:
            impulse[index] += amp

    impulse *= np.hanning(ir_len * 2)[ir_len:]
    impulse = fft_tone_filter(
        impulse,
        sample_rate,
        points=[
            (20.0, -8.0 if instrument == "guitar" else -1.5),
            (80.0, -3.0 if instrument == "guitar" else 2.5),
            (250.0, 0.0),
            (900.0, 1.5),
            (3500.0, 0.5 if instrument == "guitar" else -6.0),
            (6500.0, -14.0),
            (sample_rate / 2.0, -42.0),
        ],
    )
    impulse /= float(np.max(np.abs(impulse)) + 1e-12)
    return impulse * 0.75


def create_demo_amp_target(di_audio: np.ndarray, sample_rate: int, instrument: str) -> np.ndarray:
    """Create a synthetic processed amp/cab target for demo capture."""
    if instrument == "bass":
        pre = fft_tone_filter(
            di_audio,
            sample_rate,
            points=[(20.0, -6.0), (80.0, 2.5), (250.0, 1.0), (900.0, -1.5), (3500.0, -4.0), (sample_rate / 2.0, -35.0)],
        )
        saturated = apply_dynamic_nonlinearity(
            pre,
            sample_rate,
            input_gain=2.1,
            drive=2.6,
            bias=-0.025,
            sag=0.42,
            compression=0.34,
        )
    else:
        pre = fft_tone_filter(
            di_audio,
            sample_rate,
            points=[(20.0, -18.0), (110.0, -4.0), (600.0, 1.5), (1600.0, 3.0), (4500.0, 1.0), (sample_rate / 2.0, -30.0)],
        )
        saturated = apply_dynamic_nonlinearity(
            pre,
            sample_rate,
            input_gain=3.2,
            drive=5.4,
            bias=0.035,
            sag=0.24,
            compression=0.16,
        )

    cabinet_ir = make_cabinet_ir(sample_rate, instrument)
    cabinet = fftconvolve(saturated, cabinet_ir, mode="full")[: len(saturated)]
    room_hint = fftconvolve(cabinet, np.array([1.0, 0.0, 0.0, 0.08, -0.04]), mode="full")[: len(cabinet)]
    return normalize_peak(room_hint, peak=0.78)


def peak_dbfs(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(float(np.max(np.abs(audio))) + 1e-12))


def rms_dbfs(audio: np.ndarray) -> float:
    return float(20.0 * np.log10(rms(audio) + 1e-12))


def crest_factor(audio: np.ndarray) -> float:
    return float((np.max(np.abs(audio)) + 1e-12) / (rms(audio) + 1e-12))


def peak_over_rms_db(peak: float, rms_value: float) -> float:
    return float(peak - rms_value)


def level_profile_settings(level_profile: str) -> dict[str, float | str]:
    if level_profile == "light":
        return {
            "name": "light",
            "usable_floor_dbfs": -24.0,
            "ideal_floor_dbfs": -16.0,
            "ideal_ceiling_dbfs": -8.0,
            "hot_ceiling_dbfs": -6.0,
            "balance_tolerance_db": 6.0,
            "check_target": "DI/mic -16 to -8 dBFS",
        }
    if level_profile == "dynamic":
        return {
            "name": "dynamic",
            "usable_floor_dbfs": -26.0,
            "ideal_floor_dbfs": -20.0,
            "ideal_ceiling_dbfs": -12.0,
            "hot_ceiling_dbfs": -5.0,
            "balance_tolerance_db": 7.0,
            "check_target": "DI/mic -20 to -12 dBFS",
        }
    if level_profile == "aggressive":
        return {
            "name": "aggressive",
            "usable_floor_dbfs": -28.0,
            "ideal_floor_dbfs": -22.0,
            "ideal_ceiling_dbfs": -14.0,
            "hot_ceiling_dbfs": -3.0,
            "balance_tolerance_db": 8.0,
            "check_target": "DI -22 to -14 dBFS, mic -18 to -10 dBFS",
        }
    if level_profile == "extreme":
        return {
            "name": "extreme",
            "usable_floor_dbfs": -32.0,
            "ideal_floor_dbfs": -26.0,
            "ideal_ceiling_dbfs": -16.0,
            "hot_ceiling_dbfs": -2.0,
            "balance_tolerance_db": 10.0,
            "check_target": "DI -26 to -16 dBFS, mic -20 to -10 dBFS",
        }
    return {
        "name": "normal",
        "usable_floor_dbfs": -24.0,
        "ideal_floor_dbfs": -18.0,
        "ideal_ceiling_dbfs": -10.0,
        "hot_ceiling_dbfs": -6.0,
        "balance_tolerance_db": 6.0,
        "check_target": "DI/mic -18 to -10 dBFS",
    }


def level_channel_ranges(level_profile: str = "normal") -> dict[str, tuple[float, float]]:
    settings = level_profile_settings(level_profile)
    di_ideal = (
        float(settings["ideal_floor_dbfs"]),
        float(settings["ideal_ceiling_dbfs"]),
    )
    mic_ideal = di_ideal

    if level_profile == "aggressive":
        mic_ideal = (-18.0, -10.0)
    elif level_profile == "extreme":
        mic_ideal = (-20.0, -10.0)

    return {
        "usable": (
            float(settings["usable_floor_dbfs"]),
            float(settings["hot_ceiling_dbfs"]),
        ),
        "di_ideal": di_ideal,
        "mic_ideal": mic_ideal,
    }


def classify_peak_level(peak: float, level_profile: str = "normal") -> tuple[str, str]:
    settings = level_profile_settings(level_profile)
    usable_floor = float(settings["usable_floor_dbfs"])
    hot_ceiling = float(settings["hot_ceiling_dbfs"])

    if peak >= -0.5:
        return "clipping risk", "Back the preamp/interface gain down and retake."
    if peak > hot_ceiling:
        return "hot", "Usable if clean, but lower gain before a main model take."
    if level_profile in {"dynamic", "aggressive", "extreme"} and peak > -6.0:
        return "transient hot", "Hard-pick transient captured cleanly; watch for clipping, not knob position."
    if peak >= usable_floor:
        return "good", "Good training range."
    if peak >= -36.0:
        return "quiet", "Usable if clean, but raise gain if you can."
    if peak >= -90.0:
        return "too quiet", "Retake for the main model; the model may learn noise."
    return "silent", "Check the channel, cable, DI routing, and interface input."


def classify_level_match(peak_delta_db: float, level_profile: str = "normal") -> tuple[str, str]:
    tolerance = float(level_profile_settings(level_profile)["balance_tolerance_db"])
    if abs(peak_delta_db) <= tolerance:
        return "matched enough", "Good channel balance for capture training."
    if peak_delta_db > 10.0:
        return "DI much hotter", "Turn channel 1 DI gain down before the next main take."
    if peak_delta_db > tolerance:
        return "DI hotter", "Turn channel 1 DI gain down a little."
    if peak_delta_db < -10.0:
        return "mic much hotter", "Turn channel 2 mic gain down or raise channel 1 DI gain."
    return "mic hotter", "Turn channel 2 mic gain down a little or raise channel 1 DI gain."


def classify_playing_dynamics(peak_over_rms: float) -> tuple[str, str]:
    if peak_over_rms >= 22.0:
        return "very transient", "Normal for hard-picked guitar/palm mutes; set level checks with extra headroom."
    if peak_over_rms >= 17.0:
        return "dynamic", "Healthy guitar dynamics; leave a few dB of peak headroom."
    return "controlled", "Moderate peak-to-RMS movement."


def percentile_peak_dbfs(audio: np.ndarray, percentile: float = 99.9) -> float:
    return float(20.0 * np.log10(float(np.percentile(np.abs(audio), percentile)) + 1e-12))


def hot_sample_percent(audio: np.ndarray, threshold_dbfs: float = -6.0) -> float:
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    return float(100.0 * np.mean(np.abs(audio) >= threshold))


def build_level_report(di_audio: np.ndarray, target_audio: np.ndarray, level_profile: str = "normal") -> dict:
    settings = level_profile_settings(level_profile)
    di_peak = peak_dbfs(di_audio)
    target_peak = peak_dbfs(target_audio)
    di_rms = rms_dbfs(di_audio)
    target_rms = rms_dbfs(target_audio)
    di_p999 = percentile_peak_dbfs(di_audio, 99.9)
    target_p999 = percentile_peak_dbfs(target_audio, 99.9)
    di_hot_percent = hot_sample_percent(di_audio, -6.0)
    target_hot_percent = hot_sample_percent(target_audio, -6.0)
    di_status, di_advice = classify_peak_level(di_peak, level_profile)
    target_status, target_advice = classify_peak_level(target_peak, level_profile)
    peak_delta = di_peak - target_peak
    rms_delta = di_rms - target_rms
    match_status, balance_advice = classify_level_match(peak_delta, level_profile)
    di_peak_over_rms = peak_over_rms_db(di_peak, di_rms)
    target_peak_over_rms = peak_over_rms_db(target_peak, target_rms)
    dynamics_status, dynamics_advice = classify_playing_dynamics(di_peak_over_rms)
    trainable_statuses = {"good", "hot", "transient hot", "quiet"}
    trainable = di_status in trainable_statuses and target_status in trainable_statuses
    preferred_statuses = {"good"} if level_profile == "normal" else {"good", "transient hot"}
    preferred = (
        di_status in preferred_statuses
        and target_status in preferred_statuses
        and abs(peak_delta) <= float(settings["balance_tolerance_db"])
    )
    return {
        "level_profile": level_profile,
        "clean_di_peak_dbfs": di_peak,
        "amp_mic_target_peak_dbfs": target_peak,
        "clean_di_rms": rms(di_audio),
        "amp_mic_target_rms": rms(target_audio),
        "clean_di_rms_dbfs": di_rms,
        "amp_mic_target_rms_dbfs": target_rms,
        "clean_di_peak_p999_dbfs": di_p999,
        "amp_mic_target_peak_p999_dbfs": target_p999,
        "clean_di_hot_sample_percent": di_hot_percent,
        "amp_mic_target_hot_sample_percent": target_hot_percent,
        "clean_di_crest_factor": crest_factor(di_audio),
        "amp_mic_target_crest_factor": crest_factor(target_audio),
        "clean_di_peak_over_rms_db": di_peak_over_rms,
        "amp_mic_target_peak_over_rms_db": target_peak_over_rms,
        "clean_di_headroom_db": -di_peak,
        "amp_mic_target_headroom_db": -target_peak,
        "clean_di_status": di_status,
        "amp_mic_target_status": target_status,
        "clean_di_advice": di_advice,
        "amp_mic_target_advice": target_advice,
        "peak_delta_db": peak_delta,
        "rms_delta_db": rms_delta,
        "level_match_status": match_status,
        "level_match_advice": balance_advice,
        "playing_dynamics_status": dynamics_status,
        "playing_dynamics_advice": dynamics_advice,
        "preferred_for_training": preferred,
        "usable_for_training": trainable,
        "target_peak_range_dbfs": str(settings["check_target"]),
        "target_balance_range_db": f"DI and mic peaks within +/-{settings['balance_tolerance_db']:.0f} dB preferred",
    }


def print_level_report(report: dict) -> None:
    profile_label = report.get("level_profile", "normal")
    if profile_label != "normal":
        print(f"Level profile: {profile_label}")
    print(
        "Clean DI: "
        f"peak {report['clean_di_peak_dbfs']:.1f} dBFS "
        f"rms {report['clean_di_rms_dbfs']:.1f} dBFS "
        f"crest {report['clean_di_crest_factor']:.1f}x "
        f"headroom {report['clean_di_headroom_db']:.1f} dB "
        f"({report['clean_di_status']})"
    )
    print(
        "Amp/mic target: "
        f"peak {report['amp_mic_target_peak_dbfs']:.1f} dBFS "
        f"rms {report['amp_mic_target_rms_dbfs']:.1f} dBFS "
        f"crest {report['amp_mic_target_crest_factor']:.1f}x "
        f"headroom {report['amp_mic_target_headroom_db']:.1f} dB "
        f"({report['amp_mic_target_status']})"
    )
    print(
        "Input balance: "
        f"peak delta DI-mic {report['peak_delta_db']:+.1f} dB, "
        f"rms delta DI-mic {report['rms_delta_db']:+.1f} dB "
        f"({report['level_match_status']})"
    )
    print(
        "Playing dynamics: "
        f"DI peak-over-RMS {report['clean_di_peak_over_rms_db']:.1f} dB, "
        f"mic peak-over-RMS {report['amp_mic_target_peak_over_rms_db']:.1f} dB "
        f"({report['playing_dynamics_status']})"
    )
    print(
        "Peak distribution: "
        f"DI p99.9 {report['clean_di_peak_p999_dbfs']:.1f} dBFS, "
        f"mic p99.9 {report['amp_mic_target_peak_p999_dbfs']:.1f} dBFS, "
        f"DI hot samples {report['clean_di_hot_sample_percent']:.4f}%"
    )
    if report["preferred_for_training"]:
        print("Level verdict: excellent for training.")
    elif report["usable_for_training"]:
        print("Level verdict: usable for training, but not ideal.")
    else:
        print("Level verdict: retake before adding this to the main model.")

    if report["level_match_status"] != "matched enough":
        print(f"Balance advice: {report['level_match_advice']}")
    if report["playing_dynamics_status"] == "very transient":
        print(f"Dynamics advice: {report['playing_dynamics_advice']}")
    if report["clean_di_status"] != "good":
        print(f"DI advice: {report['clean_di_advice']}")
    if report["amp_mic_target_status"] != "good":
        print(f"Mic advice: {report['amp_mic_target_advice']}")


def parse_audio_device(device: str | None) -> str | int | None:
    if device is None or device.strip() == "":
        return None

    try:
        return int(device)
    except ValueError:
        return device


def require_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "The real audio-interface layer needs the optional 'sounddevice' package.\n"
            "Install it inside this project environment with:\n"
            "  python3 -m pip install -r requirements-live.txt"
        ) from exc

    return sd


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise SystemExit(
            "The live graph view needs the optional 'matplotlib' package.\n"
            "Install it inside this project environment with:\n"
            "  python3 -m pip install -r requirements-live.txt"
        ) from exc

    return plt, FuncAnimation


def require_pyqtgraph():
    try:
        import pyqtgraph as pg
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:
        raise SystemExit(
            "The high-performance Qt live scope needs pyqtgraph and PySide6.\n"
            "Install them inside this project environment with:\n"
            "  python3 -m pip install -r requirements-live.txt"
        ) from exc

    return pg, QtCore, QtWidgets


def build_di_box_config(args: argparse.Namespace) -> DIBoxConfig:
    return DIBoxConfig(
        name=args.di_box,
        box_type=args.di_box_type,
        pad_db=args.pad_db,
        ground_lift=args.ground_lift,
        phantom_power_to_di=args.phantom_to_di,
        thru_to_amp=not args.no_thru_to_amp,
        mic_name=args.mic,
        amp_name=args.amp,
        cabinet_name=args.cabinet,
        notes=args.hardware_notes,
    )


def build_take_metadata(args: argparse.Namespace) -> TakeMetadata:
    return TakeMetadata(
        profile_family=getattr(args, "profile_family", ""),
        guitar=getattr(args, "guitar", ""),
        tuning=getattr(args, "tuning", ""),
        pickup=getattr(args, "pickup", ""),
        pickup_mode=getattr(args, "pickup_mode", ""),
        guitar_volume=getattr(args, "guitar_volume", ""),
        guitar_tone=getattr(args, "guitar_tone", ""),
        amp_channel=getattr(args, "amp_channel", ""),
        boost_pedal=getattr(args, "boost_pedal", ""),
        mic_position=getattr(args, "mic_position", ""),
        performance=getattr(args, "performance", ""),
        notes=getattr(args, "take_notes", ""),
    )


def build_audio_interface_config(args: argparse.Namespace) -> AudioInterfaceConfig:
    return AudioInterfaceConfig(
        device=parse_audio_device(args.device),
        sample_rate=args.sample_rate,
        duration_s=args.duration_s,
        input_channels=args.input_channels,
        di_channel=args.di_channel,
        target_channel=args.target_channel,
    )


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def apply_system_on_defaults(args: argparse.Namespace) -> None:
    for name, value in SYSTEM_ON_SCOPE_DEFAULTS.items():
        if not hasattr(args, name):
            setattr(args, name, value)

    if not getattr(args, "feature_log_enabled", True):
        args.feature_log = None


def prepare_system_on_workspace(feature_log: Path | None) -> list[Path]:
    paths = list(SYSTEM_ON_WORKSPACE_DIRS)
    if feature_log is not None:
        paths.append(feature_log.parent)

    prepared: list[Path] = []
    for path in paths:
        if path in prepared:
            continue
        path.mkdir(parents=True, exist_ok=True)
        prepared.append(path)
    return prepared


def build_hardware_context(
    take_name: str,
    interface: AudioInterfaceConfig,
    di_box: DIBoxConfig,
    take_metadata: TakeMetadata | None = None,
    di_path: Path | None = None,
    target_path: Path | None = None,
) -> dict:
    channel_map = {
        "clean_di_channel": interface.di_channel,
        "amp_mic_target_channel": interface.target_channel,
        "numbering": "1-based interface input numbers",
    }

    context = {
        "mode": "two_channel_interface_capture",
        "take_name": take_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio_interface": {
            "device": "system_default_input" if interface.device is None else interface.device,
            "sample_rate_hz": interface.sample_rate,
            "duration_s": interface.duration_s,
            "input_channels_recorded": interface.input_channels,
            "channel_map": channel_map,
        },
        "di_box": asdict(di_box),
        "take_metadata": asdict(take_metadata or TakeMetadata()),
        "routing": [
            "Instrument -> DI box 1/4 inch input",
            f"DI box XLR output -> audio interface channel {interface.di_channel} for clean DI",
            "DI box 1/4 inch THRU -> amplifier input" if di_box.thru_to_amp else "DI THRU output not used",
            f"{di_box.mic_name} on {di_box.cabinet_name} -> audio interface channel {interface.target_channel} for target tone",
        ],
        "gain_staging": {
            "target_peak_dbfs": "-12 to -6 dBFS while tracking",
            "clip_warning": "Retake if either channel clips or hits 0 dBFS.",
            "pad_note": "Use the DI pad if the clean DI channel is too hot.",
        },
        "safety_notes": [
            "Do not connect an amplifier speaker output directly to an audio interface input.",
            "A Shure SM57 does not need phantom power.",
            "A passive DI box normally does not need phantom power.",
            "Record the clean DI and mic target at the same time from the same performance.",
        ],
    }

    if di_path is not None or target_path is not None:
        context["recorded_files"] = {
            "clean_di_wav": str(di_path) if di_path else None,
            "amp_mic_target_wav": str(target_path) if target_path else None,
        }

    return context


def save_hardware_context(path: Path, context: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, indent=2), encoding="utf-8")


def load_hardware_context(path: Path | None) -> dict | None:
    if path is None:
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def append_recording_to_dataset(
    dataset_path: Path,
    recording: dict,
    profile_path: Path | None = None,
    reconstructed_path: Path | None = None,
) -> None:
    context = recording["hardware_context"]
    if dataset_path.exists():
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        dataset = {
            "dataset_version": "1.0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": "Tone-capture DI/amp-mic take manifest.",
            "takes": [],
        }

    entry = {
        "take_name": context["take_name"],
        "clean_di_wav": str(recording["di_path"]),
        "amp_mic_target_wav": str(recording["target_path"]),
        "hardware_manifest": str(recording["manifest_path"]),
        "take_metadata": context.get("take_metadata", {}),
        "recording_levels": context.get("recording_levels", {}),
        "usable_for_training": bool(
            context.get("recording_levels", {}).get("usable_for_training", False)
        ),
        "preferred_for_training": bool(
            context.get("recording_levels", {}).get("preferred_for_training", False)
        ),
    }
    if profile_path is not None:
        entry["profile_json"] = str(profile_path)
    if reconstructed_path is not None:
        entry["reconstructed_wav"] = str(reconstructed_path)

    dataset["takes"] = [
        item for item in dataset.get("takes", []) if item.get("take_name") != entry["take_name"]
    ]
    dataset["takes"].append(entry)
    dataset["updated_at"] = datetime.now().isoformat(timespec="seconds")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    print(f"Updated dataset manifest: {dataset_path}")
    if not entry["usable_for_training"]:
        print("Dataset note: this take is marked as not recommended for main-model training.")


def recording_take_name_from_clean_di(path: Path) -> str | None:
    name = path.name
    if not name.endswith(RECORDING_CLEAN_DI_SUFFIX):
        return None
    return name[: -len(RECORDING_CLEAN_DI_SUFFIX)]


def matching_amp_target_path(di_path: Path) -> Path | None:
    take_name = recording_take_name_from_clean_di(di_path)
    if not take_name:
        return None
    return di_path.with_name(f"{take_name}{RECORDING_AMP_TARGET_SUFFIX}")


def matching_hardware_manifest_path(di_path: Path) -> Path | None:
    take_name = recording_take_name_from_clean_di(di_path)
    if not take_name:
        return None
    return di_path.with_name(f"{take_name}{RECORDING_HARDWARE_MANIFEST_SUFFIX}")


def discover_recording_pair_specs(
    recordings_dir: Path,
    include_level_tests: bool = False,
    usable_only: bool = False,
) -> list[dict]:
    if not recordings_dir.exists():
        raise SystemExit(f"Recordings directory does not exist: {recordings_dir}")

    pair_specs = []
    for di_path in sorted(recordings_dir.glob(f"*{RECORDING_CLEAN_DI_SUFFIX}")):
        take_name = recording_take_name_from_clean_di(di_path)
        if not take_name:
            continue
        if not include_level_tests and take_name.startswith("level_test"):
            continue

        target_path = matching_amp_target_path(di_path)
        if target_path is None or not target_path.exists():
            continue

        take_metadata = {}
        recording_levels = {}
        manifest = {}
        manifest_path = matching_hardware_manifest_path(di_path)
        if manifest_path is not None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            take_metadata = dict(manifest.get("take_metadata", {}))
            recording_levels = dict(manifest.get("recording_levels", {}))

        if usable_only and not bool(recording_levels.get("usable_for_training", False)):
            continue

        rig_identity = rig_identity_from_manifest(manifest or {"take_metadata": take_metadata})
        pair_specs.append(
            {
                "di_path": di_path,
                "target_path": target_path,
                "take_name": take_name,
                "take_metadata": take_metadata,
                "recording_levels": recording_levels,
                "hardware_manifest": manifest,
                "hardware_manifest_path": manifest_path,
                "rig_identity": rig_identity,
                "rig_fingerprint": rig_fingerprint(rig_identity),
                "source": "recordings_dir",
            }
        )

    if not pair_specs:
        raise SystemExit(f"No paired DI/amp-mic recordings found in {recordings_dir}.")

    return pair_specs


def enrich_pair_spec_rig_identity(spec: dict) -> dict:
    enriched = dict(spec)
    identity = dict(enriched.get("rig_identity", {}))
    manifest = dict(enriched.get("hardware_manifest", {}))
    manifest_path_value = enriched.get("hardware_manifest_path")
    if not manifest and manifest_path_value:
        manifest_path = Path(manifest_path_value)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        manifest_path = matching_hardware_manifest_path(Path(enriched["di_path"]))
        if manifest_path is not None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            enriched["hardware_manifest_path"] = manifest_path
    if not identity:
        fallback_manifest = manifest or {"take_metadata": dict(enriched.get("take_metadata", {}))}
        identity = rig_identity_from_manifest(fallback_manifest)
    if manifest:
        merged_take_metadata = dict(manifest.get("take_metadata", {}))
        for key, value in dict(enriched.get("take_metadata", {})).items():
            if not isinstance(value, str) or value.strip():
                merged_take_metadata[key] = value
        merged_recording_levels = dict(manifest.get("recording_levels", {}))
        merged_recording_levels.update(dict(enriched.get("recording_levels", {})))
        enriched["take_metadata"] = merged_take_metadata
        enriched["recording_levels"] = merged_recording_levels
    enriched["hardware_manifest"] = manifest
    enriched["rig_identity"] = identity
    enriched["rig_fingerprint"] = str(enriched.get("rig_fingerprint") or rig_fingerprint(identity))
    return enriched


def select_pair_specs_by_rig_policy(
    pair_specs: list[dict],
    policy: str,
    input_path: Path | None = None,
    requested_fingerprint: str = "",
) -> tuple[list[dict], dict]:
    enriched = [enrich_pair_spec_rig_identity(spec) for spec in pair_specs]
    groups = Counter(str(spec["rig_fingerprint"]) for spec in enriched)
    identities = {}
    for spec in enriched:
        identities[str(spec["rig_fingerprint"])] = dict(spec["rig_identity"])

    selected_fingerprint = str(requested_fingerprint or "").strip()
    if selected_fingerprint and selected_fingerprint not in groups:
        available = ", ".join(f"{name}:{count}" for name, count in groups.most_common())
        raise SystemExit(f"Unknown --rig-fingerprint {selected_fingerprint}. Available groups: {available}")

    if policy == "strict" and len(groups) > 1:
        details = ", ".join(f"{name}:{count}" for name, count in groups.most_common())
        raise SystemExit(
            "Multiple fixed rigs were found. Use --rig-policy conditioned to retain all takes with explicit "
            f"rig conditioning, or choose one --rig-fingerprint. Groups: {details}"
        )

    if policy == "largest" and not selected_fingerprint:
        selected_fingerprint = groups.most_common(1)[0][0]
    elif policy == "match-input" and not selected_fingerprint:
        resolved_input = Path(input_path).resolve() if input_path else None
        for spec in enriched:
            if resolved_input is not None and Path(spec["di_path"]).resolve() == resolved_input:
                selected_fingerprint = str(spec["rig_fingerprint"])
                break
        if not selected_fingerprint:
            selected_fingerprint = str(enriched[0]["rig_fingerprint"])

    if selected_fingerprint:
        selected = [spec for spec in enriched if str(spec["rig_fingerprint"]) == selected_fingerprint]
    else:
        selected = enriched
    report = {
        "policy": str(policy),
        "group_counts": dict(groups),
        "group_identities": identities,
        "selected_fingerprint": selected_fingerprint or None,
        "conditioned": bool(policy == "conditioned" and len(groups) > 1),
        "selected_pair_count": len(selected),
        "total_pair_count": len(enriched),
    }
    return selected, report


def infer_live_pickup_reference_label(take_name: str, take_metadata: dict) -> str:
    text_parts = [
        take_name,
        str(take_metadata.get("guitar", "")),
        str(take_metadata.get("pickup", "")),
        str(take_metadata.get("pickup_mode", "")),
        str(take_metadata.get("boost_pedal", "")),
        str(take_metadata.get("notes", "")),
    ]
    text = " ".join(text_parts).lower().replace("-", "_")

    pickup = ""
    if "middle" in text or "neck+bridge" in text or "neck_bridge" in text:
        pickup = "middle"
    elif "neck" in text:
        pickup = "neck"
    elif "bridge" in text:
        pickup = "bridge"
    elif "hot rails" in text or "hot_rails" in text:
        pickup = "bridge"

    modes = []
    if "blower" in text:
        modes.append("blower")
    if "split" in text:
        modes.append("split")
    if "nazgul" in text or "nazg" in text:
        modes.append("nazgul_sentient")
    if "pegasus" in text:
        modes.append("pegasus_sentient")
    if "mahogany" in text or "maple" in text:
        modes.append("mahogany_maple")
    if "swamp_ash" in text or "swamp ash" in text or "richlite" in text or "rich_lite" in text:
        modes.append("swamp_ash_richlite")
    if "les_paul_custom_axcess" in text or "les paul custom axcess" in text:
        modes.append("les_paul_custom_axcess")
    if "maxon" in text or "808" in text or "boost" in text or "drive" in text:
        modes.append("boost")

    if pickup and modes:
        return f"{pickup}/{'/'.join(dict.fromkeys(modes))}"
    if pickup:
        return pickup
    if modes:
        return "/".join(dict.fromkeys(modes))
    return ""


def build_live_pickup_reference_library(
    recordings_dir: Path,
    fft_size: int = 4096,
    max_seconds: float = 14.0,
    include_level_tests: bool = False,
) -> dict:
    if not recordings_dir.exists():
        return {"entries": [], "labels": [], "label_counts": {}}

    try:
        pair_specs = discover_recording_pair_specs(
            recordings_dir=recordings_dir,
            include_level_tests=include_level_tests,
            usable_only=False,
        )
    except SystemExit:
        return {"entries": [], "labels": [], "label_counts": {}}

    entries = []
    for spec in pair_specs:
        label = infer_live_pickup_reference_label(str(spec.get("take_name", "")), dict(spec.get("take_metadata", {})))
        if not label:
            continue

        try:
            di_rate, di_audio = read_wav_float(Path(spec["di_path"]))
            mic_rate, mic_audio = read_wav_float(Path(spec["target_path"]))
            di_vector = live_pickup_feature_vector_from_audio(
                di_audio,
                di_rate,
                fft_size=fft_size,
                max_seconds=max_seconds,
            )
            mic_vector = live_pickup_feature_vector_from_audio(
                mic_audio,
                mic_rate,
                fft_size=fft_size,
                max_seconds=max_seconds,
            )
        except Exception as exc:
            entries.append(
                {
                    "label": "error",
                    "take_name": str(spec.get("take_name", "")),
                    "error": str(exc),
                }
            )
            continue

        entries.append(
            {
                "label": label,
                "take_name": str(spec.get("take_name", "")),
                "di_vector": di_vector,
                "mic_vector": mic_vector,
                "di_path": str(spec["di_path"]),
                "target_path": str(spec["target_path"]),
            }
        )

    valid_entries = [entry for entry in entries if isinstance(entry.get("di_vector"), np.ndarray)]
    label_counts: dict[str, int] = {}
    for entry in valid_entries:
        label = str(entry["label"])
        label_counts[label] = label_counts.get(label, 0) + 1

    sklearn_models = build_live_pickup_sklearn_models(valid_entries)
    return {
        "entries": valid_entries,
        "labels": sorted(label_counts),
        "label_counts": label_counts,
        "errors": [entry for entry in entries if entry.get("label") == "error"],
        "sklearn": sklearn_models,
    }


def live_pickup_classifier_vector(entry: dict, mode: str) -> np.ndarray | None:
    di_vector = entry.get("di_vector")
    mic_vector = entry.get("mic_vector")
    if mode == "di" and isinstance(di_vector, np.ndarray):
        return np.asarray(di_vector, dtype=np.float64)
    if mode == "mic" and isinstance(mic_vector, np.ndarray):
        return np.asarray(mic_vector, dtype=np.float64)
    if mode == "both" and isinstance(di_vector, np.ndarray) and isinstance(mic_vector, np.ndarray):
        return np.concatenate(
            [
                np.asarray(di_vector, dtype=np.float64),
                np.asarray(mic_vector, dtype=np.float64),
            ]
        )
    return None


def build_live_pickup_sklearn_models(entries: list[dict]) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"enabled": False, "reason": "scikit-learn not installed", "models": {}}

    models = {}
    for mode in ("both", "di", "mic"):
        vectors = []
        labels = []
        for entry in entries:
            label = str(entry.get("label", ""))
            vector = live_pickup_classifier_vector(entry, mode)
            if not label or vector is None or not np.all(np.isfinite(vector)):
                continue
            vectors.append(vector)
            labels.append(label)

        unique_labels = sorted(set(labels))
        if len(unique_labels) < 2 or len(vectors) < 4:
            continue

        try:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.75,
                    class_weight="balanced",
                    max_iter=1200,
                    random_state=6505,
                ),
            )
            model.fit(np.asarray(vectors, dtype=np.float64), np.asarray(labels))
        except Exception:
            continue

        models[mode] = {
            "model": model,
            "labels": unique_labels,
            "training_count": len(vectors),
        }

    return {
        "enabled": bool(models),
        "reason": "" if models else "not enough labeled reference recordings",
        "models": models,
    }


def classify_live_pickup_sklearn(
    library: dict,
    di_vector: np.ndarray,
    mic_vector: np.ndarray,
    di_active: bool,
    mic_active: bool,
) -> dict[str, float | bool | str]:
    sklearn_state = library.get("sklearn", {})
    models = dict(sklearn_state.get("models", {})) if isinstance(sklearn_state, dict) else {}
    if not models:
        return {"active": False, "reliable": False, "label": "classifier unavailable", "confidence": 0.0}

    if di_active and mic_active and "both" in models:
        mode = "both"
        vector = np.concatenate([di_vector, mic_vector])
    elif mic_active and "mic" in models:
        mode = "mic"
        vector = mic_vector
    elif di_active and "di" in models:
        mode = "di"
        vector = di_vector
    else:
        return {"active": True, "reliable": False, "label": "classifier no active route", "confidence": 0.0}

    route = dict(models[mode])
    model = route.get("model")
    if model is None:
        return {"active": True, "reliable": False, "label": "classifier missing", "confidence": 0.0}

    try:
        probabilities = np.asarray(model.predict_proba(np.asarray([vector], dtype=np.float64))[0], dtype=np.float64)
        classes = list(getattr(model, "classes_", []))
    except Exception:
        return {"active": True, "reliable": False, "label": "classifier failed", "confidence": 0.0}

    if len(probabilities) == 0 or len(classes) != len(probabilities):
        return {"active": True, "reliable": False, "label": "classifier invalid", "confidence": 0.0}

    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    second_probability = float(probabilities[int(order[1])]) if len(order) > 1 else 0.0
    confidence = float(probabilities[best_index])
    margin = float(confidence - second_probability)
    reliable = bool(confidence >= 0.62 and margin >= 0.16)
    return {
        "active": True,
        "reliable": reliable,
        "label": str(classes[best_index]),
        "confidence": confidence,
        "margin": margin,
        "method": f"sklearn-{mode}",
        "training_count": float(route.get("training_count", 0)),
    }


def classify_live_pickup_reference(
    library: dict,
    di_vector: np.ndarray,
    mic_vector: np.ndarray,
    di_active: bool,
    mic_active: bool,
    amp_weight: float = 2.75,
    min_margin: float = 0.18,
    max_distance: float = 3.2,
) -> dict[str, float | bool | str]:
    entries = list(library.get("entries", []))
    if not entries:
        return {"active": False, "reliable": False, "label": "no recording refs", "confidence": 0.0}
    if not di_active and not mic_active:
        return {"active": False, "reliable": False, "label": "waiting for signal", "confidence": 0.0}

    label_distances: dict[str, list[float]] = {}
    for entry in entries:
        label = str(entry["label"])
        weighted_distance = 0.0
        total_weight = 0.0
        if di_active and isinstance(entry.get("di_vector"), np.ndarray):
            diff = (di_vector - entry["di_vector"]) / LIVE_PICKUP_FEATURE_SCALES
            weighted_distance += float(np.sqrt(np.mean(np.square(diff))))
            total_weight += 1.0
        if mic_active and isinstance(entry.get("mic_vector"), np.ndarray):
            diff = (mic_vector - entry["mic_vector"]) / LIVE_PICKUP_FEATURE_SCALES
            weighted_distance += float(amp_weight) * float(np.sqrt(np.mean(np.square(diff))))
            total_weight += float(amp_weight)
        if total_weight <= 0.0:
            continue
        label_distances.setdefault(label, []).append(weighted_distance / total_weight)

    if not label_distances:
        return {"active": True, "reliable": False, "label": "no comparable refs", "confidence": 0.0}

    ranked = sorted(
        (
            (label, float(np.min(distances)), len(distances))
            for label, distances in label_distances.items()
            if distances
        ),
        key=lambda item: item[1],
    )
    best_label, best_distance, best_count = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else best_distance + max(min_margin, 1.0)
    margin = float(second_distance - best_distance)
    confidence = float(np.clip(margin / max(second_distance, 1e-6), 0.0, 1.0))
    close_match = best_distance <= min(0.12, max_distance * 0.20)
    reliable = bool(best_distance <= max_distance and (close_match or len(ranked) == 1 or margin >= min_margin))
    classifier = classify_live_pickup_sklearn(library, di_vector, mic_vector, di_active, mic_active)
    if bool(classifier.get("reliable", False)) and (
        str(classifier.get("label", "")) == best_label or not reliable or float(classifier.get("confidence", 0.0)) >= 0.74
    ):
        return {
            "active": True,
            "reliable": True,
            "label": str(classifier.get("label", best_label)),
            "confidence": float(classifier.get("confidence", 0.0)),
            "distance": best_distance,
            "second_distance": float(second_distance),
            "margin": float(classifier.get("margin", margin)),
            "reference_count": float(best_count),
            "method": str(classifier.get("method", "sklearn")),
            "nearest_label": best_label,
            "nearest_confidence": confidence,
            "classifier_training_count": float(classifier.get("training_count", 0)),
        }
    return {
        "active": True,
        "reliable": reliable,
        "label": best_label,
        "confidence": confidence,
        "distance": best_distance,
        "second_distance": float(second_distance),
        "margin": margin,
        "reference_count": float(best_count),
        "method": "nearest-reference",
        "classifier_label": str(classifier.get("label", "")),
        "classifier_confidence": float(classifier.get("confidence", 0.0)),
        "classifier_reliable": bool(classifier.get("reliable", False)),
    }


def latest_clean_di_from_pairs(pair_specs: list[dict]) -> Path:
    return max((Path(spec["di_path"]) for spec in pair_specs), key=lambda path: path.stat().st_mtime)


def dataset_take_is_inactive(item: dict) -> bool:
    cleanup_status = item.get("cleanup_status", {})
    return bool(
        item.get("inactive_for_training", False)
        or item.get("archived", False)
        or cleanup_status.get("action") in {"archived", "deleted"}
    )


def dataset_selected_take_entries(
    dataset_path: Path,
    profile_family: str = "",
    include_takes: list[str] | None = None,
    exclude_takes: list[str] | None = None,
    usable_only: bool = True,
    preferred_only: bool = False,
    dataset: dict | None = None,
) -> list[dict]:
    if dataset is None:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    include_set = set(include_takes or [])
    exclude_set = set(exclude_takes or [])
    selected = []

    for item in dataset.get("takes", []):
        take_name = str(item.get("take_name", ""))
        metadata = item.get("take_metadata", {})
        item_family = str(metadata.get("profile_family", ""))

        if dataset_take_is_inactive(item):
            continue
        if profile_family and item_family != profile_family:
            continue
        if include_set and take_name not in include_set:
            continue
        if take_name in exclude_set:
            continue
        if preferred_only and not bool(item.get("preferred_for_training", False)):
            continue
        if usable_only and not bool(item.get("usable_for_training", False)):
            continue

        di_path = item.get("clean_di_wav")
        target_path = item.get("amp_mic_target_wav")
        if not di_path or not target_path:
            continue
        selected.append(item)

    return selected


def refresh_dataset_take_quality(dataset_path: Path, dataset: dict, profile_family: str = "") -> int:
    refreshed = 0
    for item in dataset_family_active_entries(dataset, profile_family=profile_family):
        di_value = item.get("clean_di_wav")
        target_value = item.get("amp_mic_target_wav")
        if not di_value or not target_value:
            continue

        di_path = resolve_dataset_file_path(dataset_path, str(di_value))
        target_path = resolve_dataset_file_path(dataset_path, str(target_value))
        if not di_path.exists() or not target_path.exists():
            print(f"Quality refresh skipped missing pair: {item.get('take_name', 'unnamed_take')}")
            continue

        level_profile = str(item.get("recording_levels", {}).get("level_profile", "normal"))
        _, di_audio = read_wav_float(di_path)
        _, target_audio = read_wav_float(target_path)
        report = build_level_report(di_audio, target_audio, level_profile=level_profile)
        item["recording_levels"] = report
        item["usable_for_training"] = bool(report["usable_for_training"])
        item["preferred_for_training"] = bool(report["preferred_for_training"])
        refreshed += 1

    print(f"Quality-refreshed active takes: {refreshed}")
    return refreshed


def dataset_training_pairs(
    dataset_path: Path,
    profile_family: str = "",
    include_takes: list[str] | None = None,
    exclude_takes: list[str] | None = None,
    usable_only: bool = True,
    preferred_only: bool = False,
) -> list[tuple[Path, Path]]:
    entries = dataset_selected_take_entries(
        dataset_path=dataset_path,
        profile_family=profile_family,
        include_takes=include_takes,
        exclude_takes=exclude_takes,
        usable_only=usable_only,
        preferred_only=preferred_only,
    )
    pairs = [(Path(item["clean_di_wav"]), Path(item["amp_mic_target_wav"])) for item in entries]

    if not pairs:
        raise SystemExit(f"No training pairs matched dataset filters in {dataset_path}.")

    return pairs


def dataset_family_active_entries(dataset: dict, profile_family: str = "") -> list[dict]:
    entries = []
    for item in dataset.get("takes", []):
        if dataset_take_is_inactive(item):
            continue
        metadata = item.get("take_metadata", {})
        item_family = str(metadata.get("profile_family", ""))
        if profile_family and item_family != profile_family:
            continue
        entries.append(item)
    return entries


def resolve_dataset_file_path(dataset_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    project_candidate = dataset_path.resolve().parent.parent / path
    if project_candidate.exists():
        return project_candidate

    return cwd_candidate


def display_dataset_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def dataset_take_file_plan(dataset_path: Path, item: dict) -> list[tuple[str, Path]]:
    planned = []
    for key in DATASET_TAKE_PATH_KEYS:
        value = item.get(key)
        if not value:
            continue
        planned.append((key, resolve_dataset_file_path(dataset_path, str(value))))
    return planned


def cleanup_unused_dataset_takes(
    dataset_path: Path,
    profile_family: str = "",
    include_takes: list[str] | None = None,
    exclude_takes: list[str] | None = None,
    usable_only: bool = True,
    preferred_only: bool = False,
    archive_dir: Path = Path("archived_unused_takes"),
    mode: str = "archive",
    apply_changes: bool = False,
    confirm_delete: bool = False,
) -> dict:
    if mode not in {"archive", "delete"}:
        raise SystemExit("--cleanup-mode must be archive or delete.")
    if mode == "delete" and apply_changes and not confirm_delete:
        raise SystemExit("Permanent delete requires --confirm-delete-unused.")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    quality_refresh_count = 0
    if preferred_only:
        quality_refresh_count = refresh_dataset_take_quality(
            dataset_path=dataset_path,
            dataset=dataset,
            profile_family=profile_family,
        )
    selected_entries = dataset_selected_take_entries(
        dataset_path=dataset_path,
        profile_family=profile_family,
        include_takes=include_takes,
        exclude_takes=exclude_takes,
        usable_only=usable_only,
        preferred_only=preferred_only,
        dataset=dataset,
    )
    selected_names = {str(item.get("take_name", "")) for item in selected_entries}
    active_entries = dataset_family_active_entries(dataset, profile_family=profile_family)
    unused_entries = [
        item for item in active_entries if str(item.get("take_name", "")) not in selected_names
    ]

    summary = {
        "selected_count": len(selected_entries),
        "unused_count": len(unused_entries),
        "mode": mode,
        "apply": apply_changes,
        "quality_refresh_count": quality_refresh_count,
        "unused_takes": [],
    }
    timestamp = datetime.now().isoformat(timespec="seconds")
    timestamp_slug = timestamp.replace(":", "-")

    if not unused_entries:
        print("No unused active takes matched the cleanup filters.")
        print(f"Training-selected takes: {len(selected_entries)}")
        if apply_changes and quality_refresh_count:
            dataset["updated_at"] = timestamp
            dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
            print(f"Updated dataset quality flags: {dataset_path}")
        return summary

    if mode == "archive":
        archive_resolved = archive_dir.expanduser().resolve()
        research_resolved = TONE_RESEARCH_MOUNT.resolve()
        if archive_resolved == research_resolved or research_resolved in archive_resolved.parents:
            if not research_resolved.exists():
                raise SystemExit(f"Tone research volume is not mounted: {research_resolved}")
            research_device = research_resolved.stat().st_dev
            incoming_bytes = 0
            for item in unused_entries:
                for _, source in dataset_take_file_plan(dataset_path, item):
                    if source.exists() and source.stat().st_dev != research_device:
                        incoming_bytes += source.stat().st_size
            current_bytes = shutil.disk_usage(research_resolved).used
            projected_bytes = current_bytes + incoming_bytes
            print(
                "Tone research cap: "
                f"current={current_bytes / 1024**3:.2f} GiB "
                f"incoming={incoming_bytes / 1024**3:.2f} GiB "
                f"projected={projected_bytes / 1024**3:.2f} GiB / "
                f"{TONE_RESEARCH_WORKING_CAP_BYTES / 1024**3:.2f} GiB"
            )
            if projected_bytes > TONE_RESEARCH_WORKING_CAP_BYTES:
                raise SystemExit(
                    "Refusing archive cleanup because it would exceed the 5 GiB tone research working cap."
                )

    print(f"Training-selected takes: {len(selected_entries)}")
    print(f"Unused active takes: {len(unused_entries)}")
    print(f"Cleanup mode: {mode}")
    if not apply_changes:
        print("Dry run only. Add --apply to archive unused takes.")
        if mode == "delete":
            print("Permanent delete also requires --confirm-delete-unused.")

    for item in unused_entries:
        take_name = str(item.get("take_name", "unnamed_take"))
        file_plan = dataset_take_file_plan(dataset_path, item)
        take_summary = {"take_name": take_name, "files": []}
        print(f"\nUnused take: {take_name}")

        for key, source in file_plan:
            exists = source.exists()
            action_label = "would"
            target_path = None

            if apply_changes and exists and mode == "archive":
                destination_dir = archive_dir / dataset_path.stem / take_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / source.name
                if destination.exists():
                    destination = destination.with_name(
                        f"{destination.stem}_{timestamp_slug}{destination.suffix}"
                    )
                shutil.move(str(source), str(destination))
                target_path = destination
                item[key] = display_dataset_path(destination)
                action_label = "archived"
            elif apply_changes and exists and mode == "delete":
                source.unlink()
                action_label = "deleted"
            elif mode == "archive":
                target_path = archive_dir / dataset_path.stem / take_name / source.name

            if mode == "archive":
                if target_path is not None:
                    print(
                        f"  {action_label} archive {key}: "
                        f"{display_dataset_path(source)} -> {display_dataset_path(target_path)}"
                    )
                else:
                    print(f"  missing {key}: {display_dataset_path(source)}")
            else:
                print(f"  {action_label} delete {key}: {display_dataset_path(source)}")

            take_summary["files"].append(
                {
                    "key": key,
                    "source": display_dataset_path(source),
                    "target": display_dataset_path(target_path) if target_path else "",
                    "exists": exists,
                    "action": action_label,
                }
            )

        summary["unused_takes"].append(take_summary)

        if apply_changes:
            item["inactive_for_training"] = True
            item["archived"] = mode == "archive"
            item["usable_for_training"] = False
            item["preferred_for_training"] = False
            item["cleanup_status"] = {
                "action": "archived" if mode == "archive" else "deleted",
                "timestamp": timestamp,
                "reason": "not selected by current training filters",
                "profile_family": profile_family,
                "selected_training_takes": sorted(selected_names),
            }

    if apply_changes:
        dataset["updated_at"] = timestamp
        dataset_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
        print(f"\nUpdated dataset manifest: {dataset_path}")

    return summary


def validate_interface_channels(interface: AudioInterfaceConfig) -> None:
    if interface.di_channel < 1 or interface.target_channel < 1:
        raise SystemExit("Interface channels are 1-based. Use channel 1, 2, 3, etc.")

    if interface.di_channel == interface.target_channel:
        raise SystemExit("The clean DI and amp/mic target need two different interface channels.")

    required_channels = max(interface.di_channel, interface.target_channel)
    if interface.input_channels < required_channels:
        raise SystemExit(
            f"--input-channels must be at least {required_channels} for the selected channel map."
        )

    if interface.duration_s <= 0:
        raise SystemExit("--duration-s must be greater than zero.")

    if interface.sample_rate <= 0:
        raise SystemExit("--sample-rate must be greater than zero.")


def record_interface_take(
    take_name: str,
    output_dir: Path,
    interface: AudioInterfaceConfig,
    di_box: DIBoxConfig,
    take_metadata: TakeMetadata | None = None,
    level_profile: str = "normal",
) -> dict:
    validate_interface_channels(interface)
    sd = require_sounddevice()

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(interface.duration_s * interface.sample_rate))
    device_label = "system default input" if interface.device is None else interface.device

    print("Hardware routing:")
    print("  Guitar/bass -> DI input")
    print(f"  DI XLR out -> interface channel {interface.di_channel} clean DI")
    print(f"  DI THRU -> amp -> cab -> {di_box.mic_name} -> interface channel {interface.target_channel}")
    print(f"Recording {interface.duration_s:.1f}s from {device_label}...")

    recording = sd.rec(
        frame_count,
        samplerate=interface.sample_rate,
        channels=interface.input_channels,
        dtype="float64",
        device=interface.device,
    )
    sd.wait()
    recording = np.asarray(recording, dtype=np.float64)

    di_audio = recording[:, interface.di_channel - 1]
    target_audio = recording[:, interface.target_channel - 1]

    di_path = output_dir / f"{take_name}_clean_di.wav"
    target_path = output_dir / f"{take_name}_amp_mic_target.wav"
    manifest_path = output_dir / f"{take_name}_hardware_manifest.json"

    write_wav_float(di_path, interface.sample_rate, di_audio)
    write_wav_float(target_path, interface.sample_rate, target_audio)

    context = build_hardware_context(
        take_name=take_name,
        interface=interface,
        di_box=di_box,
        take_metadata=take_metadata,
        di_path=di_path,
        target_path=target_path,
    )
    context["recording_levels"] = build_level_report(di_audio, target_audio, level_profile=level_profile)
    save_hardware_context(manifest_path, context)

    print(f"Wrote clean DI: {di_path}")
    print(f"Wrote amp/mic target: {target_path}")
    print(f"Wrote hardware manifest: {manifest_path}")
    print_level_report(context["recording_levels"])

    return {
        "di_path": di_path,
        "target_path": target_path,
        "manifest_path": manifest_path,
        "hardware_context": context,
        "di_audio": di_audio,
        "target_audio": target_audio,
    }


def write_summary(path: Path, demo_results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Guitar/Bass Amp Tone Capture Engine",
        "",
        "Pipeline:",
        "1. Load a clean DI recording and a processed amp/cab target recording.",
        "2. Align both recordings.",
        "3. Search compact dynamic nonlinear saturation, sag, and compression behavior.",
        "4. Estimate a cabinet/tone impulse response with regularized deconvolution.",
        "5. Save the captured tone profile as JSON.",
        "6. Apply the saved profile to another DI performance.",
        "",
        "Demo outputs:",
    ]

    for result in demo_results:
        lines.extend(
            [
                f"- {result['instrument'].title()} profile: {result['profile_path']}",
                f"  Captured target: {result['target_path']}",
                f"  Profiled output: {result['profiled_path']}",
                f"  Nonlinear model: drive={result['drive']:.2f}, sag={result['sag']:.2f}, compression={result['compression']:.2f}",
                f"  Match correlation: {result['match_correlation']:.3f}",
                f"  Spectral error: {result['spectral_error_db']:.2f} dB",
            ]
        )

    lines.extend(
        [
            "",
            "Important note:",
            "This is a compact DSP portfolio prototype.",
            "It is not a commercial neural amp modeler or hardware clone.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_devices_command(args: argparse.Namespace) -> None:
    sd = require_sounddevice()
    devices = sd.query_devices()
    default_input = sd.default.device[0] if sd.default.device else None

    print("Available input devices:")
    for index, device in enumerate(devices):
        input_channels = int(device.get("max_input_channels", 0))
        if input_channels <= 0:
            continue

        default_marker = " *default" if index == default_input else ""
        print(
            f"{index:>3}: {device['name']} | inputs={input_channels} "
            f"default_sr={device.get('default_samplerate', 'unknown')}{default_marker}"
        )


def run_audio_stack_check_command(args: argparse.Namespace) -> None:
    stack = installed_audio_stack()
    routes = {
        "soundfile": "float WAV read/write when installed",
        "soxr": "very-high-quality resampling when installed",
        "librosa": "advanced quality-gate descriptors",
        "pyloudnorm": "integrated loudness checks in quality gate",
        "pedalboard": "explicit pedalboard-preview effect-chain command",
        "noisereduce": "explicit denoise-preview command only",
        "scikit-learn": "optional calibrated live pickup/blower reference classifier",
    }
    print("Advanced audio stack:")
    for name in sorted(routes):
        value = stack.get(name, False)
        version = value if value else "not installed"
        print(f"  {name:12s} {version} | {routes[name]}")


def run_system_work_log_command(args: argparse.Namespace) -> None:
    from scripts.build_system_work_log import build_report

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir() / output_dir
    paths = build_report(project_dir=project_dir(), output_dir=output_dir)
    print("Wrote system work log:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def run_research_stack_check_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_research_stack_check

    run_research_stack_check(args)


def run_prepare_research_capture_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_prepare_research_capture

    run_prepare_research_capture(args)


def run_prepare_conditioned_nam_reference_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_prepare_conditioned_nam_reference

    run_prepare_conditioned_nam_reference(args)


def run_train_torch_reference_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_train_torch_reference

    run_train_torch_reference(args)


def run_train_conditioned_torch_reference_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_train_conditioned_torch_reference

    run_train_conditioned_torch_reference(args)


def run_apply_torch_reference_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_apply_torch_reference

    run_apply_torch_reference(args)


def run_apply_nam_reference_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_apply_nam_reference

    run_apply_nam_reference(args)


def run_hybrid_model_compare_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_hybrid_model_compare

    run_hybrid_model_compare(args)


def run_build_conditioned_dataset_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_build_conditioned_dataset

    run_build_conditioned_dataset(args)


def run_freeze_conditioned_dataset_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from research_model_workflow import run_freeze_conditioned_dataset

    run_freeze_conditioned_dataset(args)


def run_rig_probe_generate_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from rig_capture_workflow import run_probe_generate

    run_probe_generate(args)


def run_rig_probe_record_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from rig_capture_workflow import run_probe_record

    run_probe_record(args)


def run_train_rig_capture_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from rig_capture_workflow import run_train_rig_capture

    run_train_rig_capture(args)


def run_refine_rig_capture_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from rig_capture_workflow import run_refine_rig_capture

    run_refine_rig_capture(args)


def run_build_cabinet_variant_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from cabinet_variant_workflow import run_build_cabinet_variant

    run_build_cabinet_variant(args)


def run_apply_cabinet_variant_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from cabinet_variant_workflow import run_apply_cabinet_variant

    run_apply_cabinet_variant(args)


def run_build_separated_cabinet_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from cabinet_variant_workflow import run_build_separated_cabinet

    run_build_separated_cabinet(args)


def run_apply_virtual_studio_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from virtual_studio_workflow import run_apply_virtual_studio

    run_apply_virtual_studio(args)


def run_apply_rig_capture_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from rig_capture_workflow import run_apply_rig_capture

    run_apply_rig_capture(args)


def run_build_performance_rig_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from performance_rig_workflow import run_build_performance_rig

    run_build_performance_rig(args)


def run_apply_performance_rig_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    from performance_rig_workflow import run_apply_performance_rig

    run_apply_performance_rig(args)


def run_denoise_preview_command(args: argparse.Namespace) -> None:
    sample_rate, audio = read_wav_float(args.input)
    output = reduce_noise_preview_audio(
        audio,
        sample_rate=sample_rate,
        stationary=bool(args.stationary),
        prop_decrease=float(args.prop_decrease),
    )
    if bool(args.level_match):
        output = match_reference_level(output, audio, mode="rms")
    output = normalize_for_audition(output, peak=float(args.peak))
    write_wav_float(args.output, sample_rate, output)
    before = advanced_audio_descriptors(audio, sample_rate)
    after = advanced_audio_descriptors(output, sample_rate)
    print(f"Wrote denoise preview: {args.output}")
    if before or after:
        print("Advanced descriptors:")
        print(f"  before: {before}")
        print(f"  after:  {after}")


def run_pedalboard_preview_command(args: argparse.Namespace) -> None:
    sample_rate, audio = read_wav_float(args.input)
    output = pedalboard_preview_audio(audio, sample_rate=sample_rate, preset=str(args.preset))
    if bool(args.level_match):
        output = match_reference_level(output, audio, mode="rms")
    output = normalize_for_audition(output, peak=float(args.peak))
    write_wav_float(args.output, sample_rate, output)
    print(f"Wrote pedalboard preview: {args.output}")
    print(f"Preset: {args.preset}")


def run_system_on_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    apply_system_on_defaults(args)

    interface = build_audio_interface_config(args)
    validate_interface_channels(interface)
    prepared_dirs = prepare_system_on_workspace(args.feature_log)
    device_label = "system default input" if interface.device is None else interface.device

    print("Tone capture system on.")
    print(f"Project: {project_dir()}")
    print(f"Device: {device_label} | sample_rate={interface.sample_rate} | channels={interface.input_channels}")
    print(f"Clean DI channel={interface.di_channel} | Amp/Mic channel={interface.target_channel}")
    print(f"Level profile: {args.level_profile}")
    print(f"Pickup/blower analysis window: {args.source_analysis_ms:g} ms")
    print(
        f"Live pickup view: boost={getattr(args, 'pickup_view_boost', 2.2):g} "
        f"frames={getattr(args, 'analysis_fft_frames', 6)} "
        f"eye={getattr(args, 'frequency_eye_attack', 0.34):g}/{getattr(args, 'frequency_eye_release', 0.18):g} "
        f"dOut={getattr(args, 'output_change_delta_db', 0.45):g}/{getattr(args, 'output_hot_delta_db', 1.20):g} "
        f"switch={getattr(args, 'pickup_switch_score_threshold', 0.75):g}/"
        f"{getattr(args, 'pickup_switch_hold_ms', 3200.0):g}ms "
        f"floor={getattr(args, 'pickup_signal_floor_dbfs', -62.0):g}dB "
        f"activity={getattr(args, 'pickup_activity_margin_db', 5.0):g}/"
        f"{getattr(args, 'pickup_activity_peak_margin_db', 8.0):g}dB "
        f"refs={'on' if getattr(args, 'live_pickup_reference_enabled', True) else 'off'} "
        f"ampw={getattr(args, 'live_pickup_reference_amp_weight', 2.75):g} "
        f"{'raw' if not getattr(args, 'pickup_frequency_sensitivity', True) else 'sensitive'}"
    )
    print("Prepared folders:")
    for path in prepared_dirs:
        print(f"  {path}")
    if args.feature_log is not None:
        print(f"Feature log: {args.feature_log}")

    if args.check_only:
        print("Check complete. Run system-on without --check-only to open the live scope.")
        return

    run_live_scope_qt_command(args)


def run_hardware_plan_command(args: argparse.Namespace) -> None:
    interface = build_audio_interface_config(args)
    di_box = build_di_box_config(args)
    take_metadata = build_take_metadata(args)
    validate_interface_channels(interface)

    context = build_hardware_context(
        take_name=args.take_name,
        interface=interface,
        di_box=di_box,
        take_metadata=take_metadata,
    )
    save_hardware_context(args.output, context)

    print(f"Wrote DI/interface hardware plan: {args.output}")
    for route in context["routing"]:
        print(f"- {route}")


def run_level_check_command(args: argparse.Namespace) -> None:
    interface = build_audio_interface_config(args)
    di_box = build_di_box_config(args)
    validate_interface_channels(interface)
    sd = require_sounddevice()

    frame_count = int(round(interface.duration_s * interface.sample_rate))
    device_label = "system default input" if interface.device is None else interface.device

    print("Hardware routing:")
    print("  Guitar/bass -> DI input")
    print(f"  DI XLR out -> interface channel {interface.di_channel} clean DI")
    print(f"  DI THRU -> amp -> cab -> {di_box.mic_name} -> interface channel {interface.target_channel}")
    print(f"Level-check recording {interface.duration_s:.1f}s from {device_label}...")

    recording = sd.rec(
        frame_count,
        samplerate=interface.sample_rate,
        channels=interface.input_channels,
        dtype="float64",
        device=interface.device,
    )
    sd.wait()
    recording = np.asarray(recording, dtype=np.float64)

    di_audio = recording[:, interface.di_channel - 1]
    target_audio = recording[:, interface.target_channel - 1]
    report = build_level_report(di_audio, target_audio, level_profile=args.level_profile)
    print_level_report(report)
    print("No WAV files were written. Use record-capture once the levels are right.")


def run_live_scope_command(args: argparse.Namespace) -> None:
    interface = build_audio_interface_config(args)
    validate_interface_channels(interface)
    sd = require_sounddevice()
    plt, FuncAnimation = require_matplotlib()

    import threading

    sample_rate = int(interface.sample_rate)
    if args.visual_smoothing == "hyperfluid":
        responsive_window_cap = 48.0
    elif args.visual_smoothing == "fluid":
        responsive_window_cap = 96.0
    else:
        responsive_window_cap = 80.0
    block_ms = min(args.block_ms, 8.0) if args.responsive else args.block_ms
    window_ms = min(args.window_ms, responsive_window_cap) if args.responsive else args.window_ms
    refresh_ms = min(args.refresh_ms, 16) if args.responsive else args.refresh_ms
    if args.responsive and args.visual_smoothing in {"hyperfluid", "fluid", "studio"}:
        block_ms = min(block_ms, 2.0 if args.visual_smoothing == "hyperfluid" else 4.0)
        if args.visual_smoothing == "hyperfluid":
            refresh_ms = min(refresh_ms, 5)
        else:
            refresh_ms = min(refresh_ms, 10 if args.visual_smoothing == "fluid" else 12)
    if args.visual_smoothing in {"hyperfluid", "fluid"}:
        responsive_fft_cap = 8192
    elif args.visual_smoothing == "studio":
        responsive_fft_cap = 4096
    else:
        responsive_fft_cap = 2048
    fft_size = max(256, min(int(args.fft_size), responsive_fft_cap) if args.responsive else int(args.fft_size))
    block_samples = max(128, int(round(sample_rate * block_ms / 1000.0)))
    window_samples = max(block_samples, int(round(sample_rate * window_ms / 1000.0)))
    if args.responsive and args.visual_smoothing == "hyperfluid":
        response_window_ms = 18.0
    elif args.responsive and args.visual_smoothing in {"fluid", "studio"}:
        response_window_ms = 32.0
    else:
        response_window_ms = window_ms
    response_samples = max(block_samples, int(round(sample_rate * response_window_ms / 1000.0)))
    max_freq = min(float(args.max_freq), sample_rate / 2.0)
    min_freq = max(20.0, float(args.min_freq))
    if max_freq <= min_freq:
        raise SystemExit("--max-freq must be greater than --min-freq.")
    source_names = ("Clean DI", "Amp/Mic")
    source_channels = (interface.di_channel - 1, interface.target_channel - 1)
    colors = ("#2563eb", "#dc2626")
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    attack_fft_size = min(1024, fft_size) if args.visual_smoothing == "hyperfluid" else fft_size
    attack_freqs = np.fft.rfftfreq(attack_fft_size, d=1.0 / sample_rate)
    spectrum_mask = (freqs >= min_freq) & (freqs <= max_freq)
    display_point_multiplier = 5 if args.visual_smoothing == "hyperfluid" else 4 if args.visual_smoothing in {"fluid", "studio", "ultra"} else 2
    display_point_cap = 4096 if args.visual_smoothing == "hyperfluid" else 3200 if args.visual_smoothing in {"fluid", "studio", "ultra"} else 1600
    display_point_count = max(
        256,
        min(display_point_cap, max(1, int(np.count_nonzero(spectrum_mask))) * display_point_multiplier),
    )
    if args.log_frequency:
        display_freqs = np.geomspace(min_freq, max_freq, display_point_count)
    else:
        display_freqs = np.linspace(min_freq, max_freq, display_point_count)
    show_tone_diff = args.show_tone_diff and args.view in {"both", "spectrum"}
    waveform_is_stacked = args.waveform_layout == "stacked"
    metrics_max_delay_samples = max(1, int(round(sample_rate * args.metrics_max_delay_ms / 1000.0)))
    smoothing_attack_alpha, smoothing_release_alpha = {
        "off": (1.0, 1.0),
        "light": (0.75, 0.45),
        "medium": (0.80, 0.30),
        "heavy": (0.88, 0.15),
        "hyperfluid": (0.998, 0.18),
        "fluid": (0.96, 0.065),
        "studio": (0.96, 0.10),
        "ultra": (0.35, 0.06),
    }[args.visual_smoothing]
    spectrum_smoothing_bins = max(1, int(args.spectrum_smoothing_bins))
    tone_diff_smoothing_bins = max(1, int(args.tone_diff_smoothing_bins))
    if args.visual_smoothing == "medium":
        spectrum_smoothing_bins = max(spectrum_smoothing_bins, 15)
        tone_diff_smoothing_bins = max(tone_diff_smoothing_bins, 61)
    elif args.visual_smoothing == "heavy":
        spectrum_smoothing_bins = max(spectrum_smoothing_bins, 31)
        tone_diff_smoothing_bins = max(tone_diff_smoothing_bins, 101)
    elif args.visual_smoothing == "hyperfluid":
        spectrum_smoothing_bins = max(spectrum_smoothing_bins, 7)
        tone_diff_smoothing_bins = max(tone_diff_smoothing_bins, 71)
    elif args.visual_smoothing == "fluid":
        spectrum_smoothing_bins = max(spectrum_smoothing_bins, 21)
        tone_diff_smoothing_bins = max(tone_diff_smoothing_bins, 151)
    elif args.visual_smoothing == "studio":
        spectrum_smoothing_bins = max(spectrum_smoothing_bins, 9)
        tone_diff_smoothing_bins = max(tone_diff_smoothing_bins, 61)
    elif args.visual_smoothing == "ultra":
        spectrum_smoothing_bins = max(spectrum_smoothing_bins, 61)
        tone_diff_smoothing_bins = max(tone_diff_smoothing_bins, 181)
    smooth_state: dict[str, np.ndarray] = {}
    peak_hold_state: dict[str, np.ndarray] = {}
    studio_peak_hold = args.visual_smoothing in {"hyperfluid", "fluid", "studio"}
    studio_peak_decay_db = 0.90 if args.visual_smoothing == "hyperfluid" else 0.28 if args.visual_smoothing == "fluid" else 0.55

    state = {
        "history": np.zeros((window_samples, 2), dtype=np.float64),
        "status": "",
    }
    state_lock = threading.Lock()

    def callback(indata, frames, time_info, status):
        del time_info
        captured = np.column_stack(
            [
                indata[:, source_channels[0]],
                indata[:, source_channels[1]],
            ]
        )
        with state_lock:
            history = state["history"]
            if frames >= window_samples:
                history[:] = captured[-window_samples:]
            else:
                history[:-frames] = history[frames:]
                history[-frames:] = captured
            state["status"] = str(status) if status else ""

    panel_count = int(args.view in {"both", "waveform"})
    panel_count += int(args.view in {"both", "spectrum"})
    panel_count += int(show_tone_diff)
    panel_count += int(args.show_levels)
    panel_count += int(args.show_metrics)
    fig, axes = plt.subplots(
        panel_count,
        1,
        figsize=(15, max(4.0, 2.8 * panel_count)),
        constrained_layout=True,
    )
    axes_iter = iter(np.atleast_1d(axes))
    wave_ax = next(axes_iter) if args.view in {"both", "waveform"} else None
    spectrum_ax = next(axes_iter) if args.view in {"both", "spectrum"} else None
    tone_diff_ax = next(axes_iter) if show_tone_diff else None
    level_ax = next(axes_iter) if args.show_levels else None
    metrics_ax = next(axes_iter) if args.show_metrics else None

    wave_lines = []
    wave_offsets = np.zeros(2, dtype=np.float64)
    wave_rms_lines = []
    wave_peak_markers = []
    wave_text = None
    if wave_ax is not None:
        time_ms = (np.arange(window_samples) - window_samples) * 1000.0 / sample_rate
        if waveform_is_stacked:
            lane_gap = args.amplitude_range * 2.5
            wave_offsets = np.array([lane_gap / 2.0, -lane_gap / 2.0], dtype=np.float64)
        else:
            wave_offsets = np.zeros(2, dtype=np.float64)

        for name, color in zip(source_names, colors):
            source_index = len(wave_lines)
            offset = wave_offsets[source_index]
            (line,) = wave_ax.plot(
                time_ms,
                np.full(window_samples, offset),
                label=name,
                linewidth=args.waveform_linewidth,
                color=color,
            )
            wave_lines.append(line)
            if args.show_waveform_details:
                wave_ax.axhline(offset, color=color, alpha=0.32, linewidth=0.9)
                rms_pos = wave_ax.axhline(offset, color=color, alpha=0.0, linewidth=1.0, linestyle=":")
                rms_neg = wave_ax.axhline(offset, color=color, alpha=0.0, linewidth=1.0, linestyle=":")
                wave_rms_lines.append((rms_pos, rms_neg))
                (pos_marker,) = wave_ax.plot([], [], marker="o", linestyle="", color=color, alpha=0.85, markersize=4)
                (neg_marker,) = wave_ax.plot([], [], marker="o", linestyle="", color=color, alpha=0.85, markersize=4)
                wave_peak_markers.append((pos_marker, neg_marker))

        if args.show_waveform_details:
            if waveform_is_stacked:
                for offset in wave_offsets:
                    wave_ax.axhline(offset + args.clip_guard, color="#991b1b", alpha=0.18, linewidth=0.8)
                    wave_ax.axhline(offset - args.clip_guard, color="#991b1b", alpha=0.18, linewidth=0.8)
            else:
                wave_ax.axhline(args.clip_guard, color="#991b1b", alpha=0.24, linewidth=0.9, linestyle="--")
                wave_ax.axhline(-args.clip_guard, color="#991b1b", alpha=0.24, linewidth=0.9, linestyle="--")
            wave_text = wave_ax.text(
                0.01,
                0.98,
                "",
                transform=wave_ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.74, "edgecolor": "#d4d4d4"},
            )

        wave_ax.set_title("Live Waveform Detail")
        wave_ax.set_xlabel("Time (ms)")
        wave_ax.set_ylabel("Amplitude" if not waveform_is_stacked else "Stacked Inputs")
        if waveform_is_stacked:
            wave_ax.set_ylim(
                float(np.min(wave_offsets) - args.amplitude_range * 1.35),
                float(np.max(wave_offsets) + args.amplitude_range * 1.35),
            )
            wave_ax.set_yticks(wave_offsets, labels=source_names)
        else:
            wave_ax.set_ylim(-args.amplitude_range, args.amplitude_range)
        wave_ax.grid(True, alpha=0.25)
        wave_ax.legend(loc="upper right")

    spectrum_lines = []
    spectrum_peak_lines = []
    spectrum_text = None
    if spectrum_ax is not None:
        for name, color in zip(source_names, colors):
            if studio_peak_hold:
                (peak_line,) = spectrum_ax.plot(
                    display_freqs,
                    np.full(len(display_freqs), -120.0),
                    label="_nolegend_",
                    linewidth=0.95,
                    color=color,
                    alpha=0.30,
                    zorder=2,
                )
                spectrum_peak_lines.append(peak_line)
            (line,) = spectrum_ax.plot(
                display_freqs,
                np.full(len(display_freqs), -120.0),
                label=name,
                linewidth=1.45 if studio_peak_hold else 1.2,
                color=color,
                zorder=3,
            )
            spectrum_lines.append(line)
        spectrum_ax.set_title("Live Frequency Spectrum")
        spectrum_ax.set_xlabel("Frequency (Hz)")
        spectrum_ax.set_ylabel("Magnitude (dBFS approx.)")
        spectrum_ax.set_ylim(args.min_db, args.max_db)
        spectrum_ax.set_xlim(min_freq, max_freq)
        if args.log_frequency:
            spectrum_ax.set_xscale("log")
        spectrum_ax.grid(True, which="both", alpha=0.25)
        spectrum_ax.legend(loc="upper right")
        for marker_hz in [80.0, 250.0, 750.0, 1500.0, 3000.0, 6000.0]:
            if min_freq <= marker_hz <= max_freq:
                spectrum_ax.axvline(marker_hz, color="#737373", alpha=0.16, linewidth=0.9)
        spectrum_text = spectrum_ax.text(
            0.01,
            0.98,
            "",
            transform=spectrum_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.72, "edgecolor": "#d4d4d4"},
        )

    tone_diff_line = None
    tone_diff_text = None
    if tone_diff_ax is not None:
        (tone_diff_line,) = tone_diff_ax.plot(
            display_freqs,
            np.zeros(len(display_freqs)),
            color="#7c3aed",
            linewidth=1.25,
            label="Amp/Mic minus Clean DI",
        )
        tone_diff_ax.axhline(0.0, color="#171717", linewidth=0.9, alpha=0.55)
        tone_diff_ax.axhspan(-3.0, 3.0, color="#e5e7eb", alpha=0.55)
        tone_diff_ax.set_title("Live Tone Difference")
        tone_diff_ax.set_xlabel("Frequency (Hz)")
        tone_diff_ax.set_ylabel("Amp/Mic - DI (dB)")
        tone_diff_ax.set_ylim(args.diff_min_db, args.diff_max_db)
        tone_diff_ax.set_xlim(min_freq, max_freq)
        if args.log_frequency:
            tone_diff_ax.set_xscale("log")
        tone_diff_ax.grid(True, which="both", alpha=0.25)
        tone_diff_ax.legend(loc="upper right")
        for marker_hz in [80.0, 250.0, 750.0, 1500.0, 3000.0, 6000.0]:
            if min_freq <= marker_hz <= max_freq:
                tone_diff_ax.axvline(marker_hz, color="#737373", alpha=0.16, linewidth=0.9)
        tone_diff_text = tone_diff_ax.text(
            0.01,
            0.98,
            "",
            transform=tone_diff_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.72, "edgecolor": "#d4d4d4"},
        )

    level_floor_db = -60.0
    level_bars = []
    level_value_texts = []
    level_match_text = None
    if level_ax is not None:
        level_ax.axvspan(-24.0, -6.0, color="#bbf7d0", alpha=0.45, label="usable")
        level_ax.axvspan(-18.0, -10.0, color="#22c55e", alpha=0.22, label="ideal")
        level_ax.axvline(-0.5, color="#991b1b", linestyle="--", linewidth=1.2, label="clip risk")
        level_bars = list(
            level_ax.barh(
                [1, 0],
                [0.0, 0.0],
                left=level_floor_db,
                color=colors,
                alpha=0.82,
                height=0.42,
            )
        )
        level_ax.set_xlim(level_floor_db, 0.0)
        level_ax.set_yticks([1, 0], labels=source_names)
        level_ax.set_xlabel("Peak Level (dBFS)")
        level_ax.set_title("Live Level Match")
        level_ax.grid(True, axis="x", alpha=0.25)
        level_ax.legend(loc="lower right")
        for y in [1, 0]:
            level_value_texts.append(level_ax.text(-58.5, y, "", va="center", ha="left", fontsize=10))
        level_match_text = level_ax.text(
            0.01,
            0.98,
            "",
            transform=level_ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78, "edgecolor": "#d4d4d4"},
        )

    metrics_texts = []
    if metrics_ax is not None:
        metrics_ax.set_title("Live Capture Metrics Dashboard")
        metrics_ax.set_xlim(0.0, 1.0)
        metrics_ax.set_ylim(0.0, 1.0)
        metrics_ax.axis("off")
        metric_boxes = [
            (0.01, "#eff6ff", "#bfdbfe"),
            (0.26, "#f0fdf4", "#bbf7d0"),
            (0.51, "#faf5ff", "#e9d5ff"),
            (0.76, "#fff7ed", "#fed7aa"),
        ]
        for x_position, facecolor, edgecolor in metric_boxes:
            metrics_texts.append(
                metrics_ax.text(
                    x_position,
                    0.90,
                    "",
                    transform=metrics_ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    family="monospace",
                    bbox={
                        "boxstyle": "round,pad=0.35",
                        "facecolor": facecolor,
                        "alpha": 0.88,
                        "edgecolor": edgecolor,
                    },
                )
            )

    def spectrum_db(audio: np.ndarray) -> np.ndarray:
        segment = audio[-min(len(audio), fft_size) :]
        if len(segment) < fft_size:
            segment = np.pad(segment, (fft_size - len(segment), 0))
        window = np.hanning(fft_size)
        spectrum = np.fft.rfft(remove_dc(segment) * window)
        scale = float(np.sum(window) / 2.0 + 1e-12)
        return 20.0 * np.log10((np.abs(spectrum) / scale) + 1e-12)

    def averaged_spectral_power(audio: np.ndarray) -> np.ndarray:
        audio = remove_dc(audio.astype(np.float64, copy=False))
        if len(audio) < fft_size:
            audio = np.pad(audio, (fft_size - len(audio), 0))
        hop = max(1, fft_size // 4)
        window = np.hanning(fft_size)
        scale = float(np.sum(window) / 2.0 + 1e-12)
        power = np.zeros(fft_size // 2 + 1, dtype=np.float64)
        frame_count = 0
        frame_starts = list(range(0, len(audio) - fft_size + 1, hop))
        if len(frame_starts) > analysis_fft_frames:
            selected = np.linspace(0, len(frame_starts) - 1, analysis_fft_frames)
            frame_starts = [frame_starts[int(round(index))] for index in selected]
        for start in frame_starts:
            frame = audio[start : start + fft_size] * window
            spectrum = np.fft.rfft(frame)
            power += np.square(np.abs(spectrum) / scale)
            frame_count += 1
        if frame_count == 0:
            frame = audio[-fft_size:] * window
            spectrum = np.fft.rfft(frame)
            power += np.square(np.abs(spectrum) / scale)
            frame_count = 1
        return power / max(1, frame_count)

    def power_spectrum_db(power: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(power + 1e-18)

    def attack_spectrum_db(audio: np.ndarray) -> np.ndarray:
        segment = audio[-min(len(audio), attack_fft_size) :]
        if len(segment) < attack_fft_size:
            segment = np.pad(segment, (attack_fft_size - len(segment), 0))
        window = np.hanning(attack_fft_size)
        spectrum = np.fft.rfft(remove_dc(segment) * window)
        scale = float(np.sum(window) / 2.0 + 1e-12)
        return 20.0 * np.log10((np.abs(spectrum) / scale) + 1e-12)

    def display_curve(values: np.ndarray, source_freqs: np.ndarray | None = None) -> np.ndarray:
        return np.interp(display_freqs, freqs if source_freqs is None else source_freqs, values)

    def smooth_display_curve(key: str, values: np.ndarray, bipolar: bool = False) -> np.ndarray:
        if smoothing_attack_alpha >= 1.0 and smoothing_release_alpha >= 1.0:
            return values
        previous = smooth_state.get(key)
        if previous is None or previous.shape != values.shape:
            smooth_state[key] = values.copy()
            return values
        if bipolar:
            fast_mask = np.abs(values) >= np.abs(previous)
        else:
            fast_mask = values >= previous
        alpha = np.where(fast_mask, smoothing_attack_alpha, smoothing_release_alpha)
        smoothed = (alpha * values) + ((1.0 - alpha) * previous)
        smooth_state[key] = smoothed
        return smoothed

    def peak_hold_curve(key: str, values: np.ndarray) -> np.ndarray:
        previous = peak_hold_state.get(key)
        if previous is None or previous.shape != values.shape:
            peak_hold_state[key] = values.copy()
            return values
        held = np.maximum(values, previous - studio_peak_decay_db)
        peak_hold_state[key] = held
        return held

    def rms_dbfs(audio: np.ndarray) -> float:
        return float(20.0 * np.log10(rms(audio) + 1e-12))

    def dominant_frequency(spec_db: np.ndarray) -> float:
        masked = spec_db[spectrum_mask]
        if len(masked) == 0:
            return 0.0
        return float(freqs[spectrum_mask][int(np.argmax(masked))])

    def band_mean(values: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    def zero_crossing_rate(audio: np.ndarray) -> float:
        centered = remove_dc(audio)
        if len(centered) < 2:
            return 0.0
        signs = np.signbit(centered)
        crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
        return float(crossings / max(len(centered) / sample_rate, 1e-12))

    def max_slew_rate(audio: np.ndarray) -> float:
        if len(audio) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(audio))) * sample_rate)

    def spectral_power(audio: np.ndarray) -> np.ndarray:
        segment = audio[-min(len(audio), fft_size) :]
        if len(segment) < fft_size:
            segment = np.pad(segment, (fft_size - len(segment), 0))
        window = np.hanning(fft_size)
        spectrum = np.fft.rfft(remove_dc(segment) * window)
        return np.abs(spectrum) ** 2

    def spectral_centroid_hz(power: np.ndarray) -> float:
        usable = spectrum_mask & (freqs > 0.0)
        total = float(np.sum(power[usable]) + 1e-12)
        return float(np.sum(freqs[usable] * power[usable]) / total)

    def spectral_rolloff_hz(power: np.ndarray, fraction: float = 0.85) -> float:
        usable = spectrum_mask & (freqs > 0.0)
        usable_freqs = freqs[usable]
        usable_power = power[usable]
        if len(usable_power) == 0:
            return 0.0
        cumulative = np.cumsum(usable_power)
        threshold = float(cumulative[-1] * fraction)
        return float(usable_freqs[min(int(np.searchsorted(cumulative, threshold)), len(usable_freqs) - 1)])

    def band_percent(power: np.ndarray, low_hz: float, high_hz: float) -> float:
        total_mask = (freqs >= 80.0) & (freqs <= 10000.0)
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        total = float(np.sum(power[total_mask]) + 1e-12)
        return float(100.0 * np.sum(power[band_mask]) / total)

    def transient_rate(audio: np.ndarray) -> float:
        if len(audio) < 4:
            return 0.0
        envelope = np.abs(remove_dc(audio))
        diff = np.diff(envelope)
        threshold = float(np.percentile(np.abs(diff), 97.5) + 1e-12)
        if threshold <= 1e-10:
            return 0.0
        hits = diff > threshold
        min_gap = max(1, int(round(sample_rate * 0.012)))
        count = 0
        last = -min_gap
        for index in np.flatnonzero(hits):
            if index - last >= min_gap:
                count += 1
                last = int(index)
        return float(count / max(len(audio) / sample_rate, 1e-12))

    def clipping_percent(audio: np.ndarray) -> float:
        return float(100.0 * np.mean(np.abs(audio) >= args.clip_guard))

    def noise_floor_estimate_dbfs(audio: np.ndarray) -> float:
        centered = np.abs(remove_dc(audio))
        return float(20.0 * np.log10(float(np.percentile(centered, 10.0)) + 1e-12))

    def estimate_delay_and_correlation(di_audio: np.ndarray, target_audio: np.ndarray) -> tuple[float, float]:
        compare_len = min(len(di_audio), len(target_audio), int(round(sample_rate * args.metrics_window_ms / 1000.0)))
        if compare_len < 128:
            return 0.0, 0.0
        di_ref = remove_dc(di_audio[-compare_len:])
        target_ref = remove_dc(target_audio[-compare_len:])
        corr = correlate(target_ref, di_ref, mode="full", method="fft")
        center = len(di_ref) - 1
        start = max(0, center - metrics_max_delay_samples)
        end = min(len(corr), center + metrics_max_delay_samples + 1)
        search = corr[start:end]
        if len(search) == 0:
            return 0.0, 0.0
        peak_index = int(np.argmax(np.abs(search)))
        lag_samples = int(peak_index + start - center)
        norm = float((np.linalg.norm(di_ref) * np.linalg.norm(target_ref)) + 1e-12)
        peak_corr = float(search[peak_index] / norm)
        return float(1000.0 * lag_samples / sample_rate), peak_corr

    def update(frame_index):
        del frame_index
        with state_lock:
            history = state["history"].copy()
            status_text = state["status"]

        response_history = history[-response_samples:]
        metric_history = response_history if args.responsive and args.visual_smoothing == "hyperfluid" else history
        di_spec = spectrum_db(history[:, 0])
        target_spec = spectrum_db(history[:, 1])
        di_attack_spec = attack_spectrum_db(history[:, 0]) if args.visual_smoothing == "hyperfluid" else None
        target_attack_spec = attack_spectrum_db(history[:, 1]) if args.visual_smoothing == "hyperfluid" else None
        di_power = spectral_power(metric_history[:, 0]) if metrics_ax is not None else None
        target_power = spectral_power(metric_history[:, 1]) if metrics_ax is not None else None

        if wave_ax is not None:
            for source_index, line in enumerate(wave_lines):
                line.set_ydata(history[:, source_index] + wave_offsets[source_index])

            if args.show_waveform_details:
                wave_rows = []
                for source_index, name in enumerate(source_names):
                    signal = history[:, source_index]
                    offset = wave_offsets[source_index]
                    signal_rms = rms(signal)
                    peak_pos_index = int(np.argmax(signal))
                    peak_neg_index = int(np.argmin(signal))
                    if source_index < len(wave_rms_lines):
                        rms_pos, rms_neg = wave_rms_lines[source_index]
                        rms_pos.set_ydata([offset + signal_rms, offset + signal_rms])
                        rms_neg.set_ydata([offset - signal_rms, offset - signal_rms])
                        rms_pos.set_alpha(0.55)
                        rms_neg.set_alpha(0.55)
                    if source_index < len(wave_peak_markers):
                        pos_marker, neg_marker = wave_peak_markers[source_index]
                        pos_marker.set_data(
                            [time_ms[peak_pos_index]],
                            [offset + signal[peak_pos_index]],
                        )
                        neg_marker.set_data(
                            [time_ms[peak_neg_index]],
                            [offset + signal[peak_neg_index]],
                        )

                    wave_rows.append(
                        f"{name}: p2p {np.ptp(signal):.3f} | "
                        f"rms {signal_rms:.3f} | "
                        f"zcr {zero_crossing_rate(signal):.0f}/s | "
                        f"slew {max_slew_rate(signal):.0f}/s"
                    )
                if wave_text is not None:
                    wave_text.set_text("\n".join(wave_rows))

        di_display_spec_for_diff = None
        target_display_spec_for_diff = None
        if spectrum_ax is not None:
            di_display_spec = display_curve(smooth_gain_db(di_spec, smoothing_bins=spectrum_smoothing_bins))
            target_display_spec = display_curve(smooth_gain_db(target_spec, smoothing_bins=spectrum_smoothing_bins))
            if di_attack_spec is not None and target_attack_spec is not None:
                attack_bins = max(5, spectrum_smoothing_bins // 2)
                di_attack_display = display_curve(
                    smooth_gain_db(di_attack_spec, smoothing_bins=attack_bins),
                    attack_freqs,
                )
                target_attack_display = display_curve(
                    smooth_gain_db(target_attack_spec, smoothing_bins=attack_bins),
                    attack_freqs,
                )
                di_display_spec = np.maximum(di_display_spec, di_attack_display - 0.5)
                target_display_spec = np.maximum(target_display_spec, target_attack_display - 0.5)
            di_smoothed_spec = smooth_display_curve("di_spec", di_display_spec)
            target_smoothed_spec = smooth_display_curve("target_spec", target_display_spec)
            di_display_spec_for_diff = di_smoothed_spec
            target_display_spec_for_diff = target_smoothed_spec
            spectrum_lines[0].set_ydata(di_smoothed_spec)
            spectrum_lines[1].set_ydata(target_smoothed_spec)
            if spectrum_peak_lines:
                spectrum_peak_lines[0].set_ydata(peak_hold_curve("di_spec_peak", di_smoothed_spec))
                spectrum_peak_lines[1].set_ydata(peak_hold_curve("target_spec_peak", target_smoothed_spec))
            if spectrum_text is not None:
                spectrum_text.set_text(
                    "Dominant freq: "
                    f"DI {dominant_frequency(di_spec):.0f} Hz | "
                    f"Amp/Mic {dominant_frequency(target_spec):.0f} Hz"
                )

        if tone_diff_ax is not None and tone_diff_line is not None:
            tone_diff = target_spec - di_spec
            tone_diff = smooth_gain_db(tone_diff, smoothing_bins=tone_diff_smoothing_bins)
            if (
                args.visual_smoothing == "hyperfluid"
                and di_display_spec_for_diff is not None
                and target_display_spec_for_diff is not None
            ):
                tone_diff_display = target_display_spec_for_diff - di_display_spec_for_diff
                tone_diff_display = smooth_gain_db(tone_diff_display, smoothing_bins=tone_diff_smoothing_bins)
            else:
                tone_diff_display = display_curve(tone_diff)
            tone_diff_line.set_ydata(smooth_display_curve("tone_diff", tone_diff_display, bipolar=True))
            if tone_diff_text is not None:
                tone_diff_text.set_text(
                    "Avg difference: "
                    f"low {band_mean(tone_diff, 80.0, 250.0):+.1f} dB | "
                    f"mid {band_mean(tone_diff, 750.0, 2000.0):+.1f} dB | "
                    f"bite {band_mean(tone_diff, 2500.0, 5000.0):+.1f} dB | "
                    f"air {band_mean(tone_diff, 6000.0, 10000.0):+.1f} dB"
                )

        di_peak = peak_dbfs(response_history[:, 0])
        target_peak = peak_dbfs(response_history[:, 1])

        if level_ax is not None and level_match_text is not None:
            peaks = [di_peak, target_peak]
            statuses = [
                classify_peak_level(di_peak, args.level_profile)[0],
                classify_peak_level(target_peak, args.level_profile)[0],
            ]
            rms_values = [rms_dbfs(response_history[:, 0]), rms_dbfs(response_history[:, 1])]
            crests = [
                float((np.max(np.abs(response_history[:, 0])) + 1e-12) / (rms(response_history[:, 0]) + 1e-12)),
                float((np.max(np.abs(response_history[:, 1])) + 1e-12) / (rms(response_history[:, 1]) + 1e-12)),
            ]
            for bar, value_text, peak, status, rms_value, crest in zip(
                level_bars,
                level_value_texts,
                peaks,
                statuses,
                rms_values,
                crests,
            ):
                clamped_peak = float(np.clip(peak, level_floor_db, 0.0))
                bar.set_width(clamped_peak - level_floor_db)
                value_text.set_text(
                    f"peak {peak:.1f} dBFS | rms {rms_value:.1f} dBFS | crest {crest:.1f}x | {status}"
                )

            delta = di_peak - target_peak
            if abs(delta) <= 6.0:
                match_status = "matched enough"
            elif delta > 6.0:
                match_status = "DI hotter"
            else:
                match_status = "mic hotter"
            level_match_text.set_text(f"DI - Amp/Mic: {delta:+.1f} dB | {match_status}")

        if metrics_ax is not None and metrics_texts and di_power is not None and target_power is not None:
            delay_ms, corr_value = estimate_delay_and_correlation(metric_history[:, 0], metric_history[:, 1])
            clip_di = clipping_percent(response_history[:, 0])
            clip_target = clipping_percent(response_history[:, 1])
            floor_di = noise_floor_estimate_dbfs(metric_history[:, 0])
            floor_target = noise_floor_estimate_dbfs(metric_history[:, 1])
            transient_di = transient_rate(response_history[:, 0])
            transient_target = transient_rate(response_history[:, 1])
            centroid_di = spectral_centroid_hz(di_power)
            centroid_target = spectral_centroid_hz(target_power)
            rolloff_di = spectral_rolloff_hz(di_power)
            rolloff_target = spectral_rolloff_hz(target_power)
            low_di = band_percent(di_power, 80.0, 250.0)
            low_target = band_percent(target_power, 80.0, 250.0)
            body_di = band_percent(di_power, 250.0, 750.0)
            body_target = band_percent(target_power, 250.0, 750.0)
            mid_di = band_percent(di_power, 750.0, 2000.0)
            mid_target = band_percent(target_power, 750.0, 2000.0)
            bite_di = band_percent(di_power, 2500.0, 5000.0)
            bite_target = band_percent(target_power, 2500.0, 5000.0)
            air_di = band_percent(di_power, 6000.0, 10000.0)
            air_target = band_percent(target_power, 6000.0, 10000.0)

            metric_columns = [
                "\n".join(
                    [
                        "CAPTURE",
                        f"delay {delay_ms:+.2f} ms",
                        f"corr  {corr_value:+.3f}",
                        f"clip DI  {clip_di:.3f}%",
                        f"clip mic {clip_target:.3f}%",
                    ]
                ),
                "\n".join(
                    [
                        "DYNAMICS",
                        f"floor DI  {floor_di:.1f} dBFS",
                        f"floor mic {floor_target:.1f} dBFS",
                        f"trans DI  {transient_di:.1f}/s",
                        f"trans mic {transient_target:.1f}/s",
                    ]
                ),
                "\n".join(
                    [
                        "SPECTRUM",
                        f"cent DI  {centroid_di:.0f} Hz",
                        f"cent mic {centroid_target:.0f} Hz",
                        f"roll DI  {rolloff_di:.0f} Hz",
                        f"roll mic {rolloff_target:.0f} Hz",
                    ]
                ),
                "\n".join(
                    [
                        "BAND ENERGY",
                        f"low  DI {low_di:4.1f}% mic {low_target:4.1f}%",
                        f"body DI {body_di:4.1f}% mic {body_target:4.1f}%",
                        f"mid  DI {mid_di:4.1f}% mic {mid_target:4.1f}%",
                        f"bite DI {bite_di:4.1f}% mic {bite_target:4.1f}%",
                        f"air  DI {air_di:4.1f}% mic {air_target:4.1f}%",
                    ]
                ),
            ]
            for text_artist, column_text in zip(metrics_texts, metric_columns):
                text_artist.set_text(column_text)

        title = (
            f"Live Amp Capture Scope | DI {di_peak:.1f} dBFS | "
            f"Amp/Mic {target_peak:.1f} dBFS"
        )
        if status_text:
            title += f" | {status_text}"
        fig.suptitle(title)
        artists = [*wave_lines, *spectrum_peak_lines, *spectrum_lines, *level_bars, *level_value_texts]
        for rms_pair in wave_rms_lines:
            artists.extend(rms_pair)
        for marker_pair in wave_peak_markers:
            artists.extend(marker_pair)
        if wave_text is not None:
            artists.append(wave_text)
        if tone_diff_line is not None:
            artists.append(tone_diff_line)
        if spectrum_text is not None:
            artists.append(spectrum_text)
        if tone_diff_text is not None:
            artists.append(tone_diff_text)
        if level_match_text is not None:
            artists.append(level_match_text)
        artists.extend(metrics_texts)
        return artists

    device_label = "system default input" if interface.device is None else interface.device
    print("Opening live graph. Close the graph window or press Ctrl+C to stop.")
    print(f"Device: {device_label} | sample_rate={sample_rate} | channels={interface.input_channels}")
    if args.responsive:
        print(
            f"Responsive scope: block_ms={block_ms:g} refresh_ms={refresh_ms:g} "
            f"window_ms={window_ms:g} response_ms={response_window_ms:g} fft_size={fft_size}"
        )
    print(
        f"Visual smoothing: {args.visual_smoothing} "
        f"(attack={smoothing_attack_alpha:.2f}, release={smoothing_release_alpha:.2f}, "
        f"spectrum_bins={spectrum_smoothing_bins}, tone_diff_bins={tone_diff_smoothing_bins})"
    )
    print(f"Clean DI channel={interface.di_channel} | Amp/Mic channel={interface.target_channel}")

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=interface.input_channels,
        dtype="float32",
        device=interface.device,
        blocksize=block_samples,
        callback=callback,
    )
    animation = FuncAnimation(fig, update, interval=refresh_ms, blit=False, cache_frame_data=False)
    try:
        with stream:
            plt.show()
    finally:
        # Keep a reference alive until after plt.show() exits.
        del animation


def run_live_scope_qt_command(args: argparse.Namespace) -> None:
    interface = build_audio_interface_config(args)
    validate_interface_channels(interface)
    sd = require_sounddevice()
    pg, QtCore, QtWidgets = require_pyqtgraph()

    import threading

    sample_rate = int(interface.sample_rate)
    block_ms = min(float(args.block_ms), 4.0) if args.responsive else float(args.block_ms)
    refresh_ms = min(int(args.refresh_ms), 8) if args.responsive else int(args.refresh_ms)
    window_ms = min(float(args.window_ms), 120.0) if args.responsive else float(args.window_ms)
    fft_size = max(256, int(args.fft_size))
    block_samples = max(64, int(round(sample_rate * block_ms / 1000.0)))
    window_samples = max(block_samples, int(round(sample_rate * window_ms / 1000.0)))
    source_analysis_ms = max(float(args.source_analysis_ms), float(args.metrics_window_ms), window_ms)
    source_analysis_samples = max(block_samples, int(round(sample_rate * source_analysis_ms / 1000.0)))
    history_samples = max(window_samples, source_analysis_samples)
    min_freq = max(20.0, float(args.min_freq))
    max_freq = min(float(args.max_freq), sample_rate / 2.0)
    if max_freq <= min_freq:
        raise SystemExit("--max-freq must be greater than --min-freq.")

    pg.setConfigOptions(antialias=args.antialias, useOpenGL=args.opengl)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("Tone Capture Live Scope")

    source_names = ("Clean DI", "Amp/Mic")
    source_channels = (interface.di_channel - 1, interface.target_channel - 1)
    colors = ("#38bdf8", "#fb7185")
    level_floor_db = -60.0
    level_settings = level_profile_settings(args.level_profile)
    level_ranges = level_channel_ranges(args.level_profile)
    usable_low, usable_high = level_ranges["usable"]
    di_ideal_low, di_ideal_high = level_ranges["di_ideal"]
    mic_ideal_low, mic_ideal_high = level_ranges["mic_ideal"]
    balance_tolerance_db = float(level_settings["balance_tolerance_db"])
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    display_point_count = max(2, int(args.display_points))
    if args.log_frequency:
        display_freqs = np.geomspace(min_freq, max_freq, display_point_count)
        plot_freqs = np.log10(display_freqs)
        plot_min_freq = float(np.log10(min_freq))
        plot_max_freq = float(np.log10(max_freq))
    else:
        display_freqs = np.linspace(min_freq, max_freq, display_point_count)
        plot_freqs = display_freqs
        plot_min_freq = min_freq
        plot_max_freq = max_freq
    display_freqs[0] = min_freq
    display_freqs[-1] = max_freq

    def frequency_x(hz: float) -> float:
        hz = float(np.clip(hz, min_freq, max_freq))
        return float(np.log10(hz)) if args.log_frequency else hz

    def frequency_label(hz: float) -> str:
        if hz >= 1000.0:
            return f"{hz / 1000.0:g}k"
        return f"{hz:g}"

    frequency_tick_hz = [
        tick for tick in (20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 12000.0)
        if min_freq <= tick <= max_freq
    ]
    frequency_ticks = [(frequency_x(tick), frequency_label(tick)) for tick in frequency_tick_hz]
    time_ms = np.linspace(-window_ms, 0.0, window_samples, dtype=np.float64)
    lane_gap = args.amplitude_range * 2.45
    wave_offsets = np.array([lane_gap / 2.0, -lane_gap / 2.0], dtype=np.float64)
    smooth_state: dict[str, np.ndarray] = {}
    peak_hold_state: dict[str, np.ndarray] = {}
    voicing_baseline_state: dict[str, np.ndarray] = {}
    pickup_frequency_baseline_state: dict[str, np.ndarray] = {}
    output_level_baseline_state: dict[str, float] = {}
    pickup_switch_state: dict[str, dict[str, object]] = {}
    pickup_activity_state: dict[str, dict[str, float]] = {}
    attack_alpha = float(args.smoothing_attack)
    release_alpha = float(args.smoothing_release)
    analysis_fft_frames = max(1, int(getattr(args, "analysis_fft_frames", 6)))
    pickup_frequency_sensitivity = bool(getattr(args, "pickup_frequency_sensitivity", True))
    pickup_view_boost = max(1.0, float(getattr(args, "pickup_view_boost", 2.2)))
    pickup_view_release = max(release_alpha, min(1.0, float(getattr(args, "pickup_view_release", 0.42))))
    frequency_eye_attack = max(0.01, min(1.0, float(getattr(args, "frequency_eye_attack", 0.34))))
    frequency_eye_release = max(0.01, min(1.0, float(getattr(args, "frequency_eye_release", 0.18))))
    pickup_view_fast_delta_db = max(0.1, float(getattr(args, "pickup_view_fast_delta_db", 0.9)))
    pickup_view_baseline_alpha = max(0.0, min(0.25, float(getattr(args, "pickup_view_baseline_alpha", 0.004))))
    pickup_view_max_delta_db = max(0.0, float(getattr(args, "pickup_view_max_delta_db", 14.0)))
    output_change_delta_db = max(0.05, float(getattr(args, "output_change_delta_db", 0.45)))
    output_hot_delta_db = max(output_change_delta_db, float(getattr(args, "output_hot_delta_db", 1.20)))
    output_baseline_alpha = max(0.0, min(0.25, float(getattr(args, "output_baseline_alpha", 0.025))))
    output_hold_alpha = max(0.0, min(output_baseline_alpha, float(getattr(args, "output_hold_alpha", 0.0025))))
    pickup_switch_score_threshold = max(0.20, float(getattr(args, "pickup_switch_score_threshold", 0.75)))
    pickup_switch_hold_s = max(0.10, float(getattr(args, "pickup_switch_hold_ms", 3200.0)) / 1000.0)
    pickup_switch_baseline_alpha = max(0.0, min(0.50, float(getattr(args, "pickup_switch_baseline_alpha", 0.10))))
    pickup_switch_hold_alpha = max(
        0.0,
        min(pickup_switch_baseline_alpha, float(getattr(args, "pickup_switch_hold_alpha", 0.012))),
    )
    pickup_signal_floor_dbfs = float(getattr(args, "pickup_signal_floor_dbfs", -62.0))
    pickup_activity_margin_db = max(1.0, float(getattr(args, "pickup_activity_margin_db", 5.0)))
    pickup_activity_peak_margin_db = max(1.0, float(getattr(args, "pickup_activity_peak_margin_db", 8.0)))
    pickup_activity_hold_s = max(0.05, float(getattr(args, "pickup_activity_hold_ms", 1500.0)) / 1000.0)
    live_pickup_reference_enabled = bool(getattr(args, "live_pickup_reference_enabled", True))
    live_pickup_reference_dir = Path(getattr(args, "live_pickup_reference_dir", Path("recordings")))
    if not live_pickup_reference_dir.is_absolute():
        project_reference_dir = project_dir() / live_pickup_reference_dir
        if project_reference_dir.exists():
            live_pickup_reference_dir = project_reference_dir
    live_pickup_reference_seconds = max(1.0, float(getattr(args, "live_pickup_reference_seconds", 14.0)))
    live_pickup_reference_amp_weight = max(0.0, float(getattr(args, "live_pickup_reference_amp_weight", 2.75)))
    live_pickup_reference_margin = max(0.0, float(getattr(args, "live_pickup_reference_margin", 0.18)))
    live_pickup_reference_max_distance = max(0.25, float(getattr(args, "live_pickup_reference_max_distance", 3.2)))
    live_pickup_reference_library = (
        build_live_pickup_reference_library(
            recordings_dir=live_pickup_reference_dir,
            fft_size=fft_size,
            max_seconds=live_pickup_reference_seconds,
        )
        if live_pickup_reference_enabled
        else {"entries": [], "labels": [], "label_counts": {}}
    )
    spectrum_peak_decay_db = 0.65

    state = {
        "history": np.zeros((history_samples, 2), dtype=np.float64),
        "status": "",
    }
    state_lock = threading.Lock()
    feature_log_handle = None
    feature_log_frame = 0
    last_feature_log_time = 0.0
    feature_log_interval_s = max(0.02, float(args.feature_log_interval_ms) / 1000.0)
    feature_log_session = datetime.now().strftime("live_scope_%Y%m%d_%H%M%S")
    if args.feature_log:
        args.feature_log.parent.mkdir(parents=True, exist_ok=True)
        feature_log_handle = args.feature_log.open("a", encoding="utf-8")
        feature_log_handle.write(
            json.dumps(
                {
                    "type": "live_scope_session_start",
                    "session_id": feature_log_session,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "sample_rate_hz": sample_rate,
                    "fft_size": fft_size,
                    "di_channel": interface.di_channel,
                    "target_channel": interface.target_channel,
                    "feature_log_interval_ms": float(args.feature_log_interval_ms),
                    "source_analysis_ms": float(source_analysis_ms),
                    "analysis_fft_frames": int(analysis_fft_frames),
                    "pickup_frequency_sensitivity": bool(pickup_frequency_sensitivity),
                    "pickup_view_boost": float(pickup_view_boost),
                    "pickup_view_release": float(pickup_view_release),
                    "frequency_eye_attack": float(frequency_eye_attack),
                    "frequency_eye_release": float(frequency_eye_release),
                    "pickup_view_fast_delta_db": float(pickup_view_fast_delta_db),
                    "pickup_view_baseline_alpha": float(pickup_view_baseline_alpha),
                    "pickup_view_max_delta_db": float(pickup_view_max_delta_db),
                    "output_change_delta_db": float(output_change_delta_db),
                    "output_hot_delta_db": float(output_hot_delta_db),
                    "output_baseline_alpha": float(output_baseline_alpha),
                    "output_hold_alpha": float(output_hold_alpha),
                    "pickup_switch_score_threshold": float(pickup_switch_score_threshold),
                    "pickup_switch_hold_ms": float(pickup_switch_hold_s * 1000.0),
                    "pickup_switch_baseline_alpha": float(pickup_switch_baseline_alpha),
                    "pickup_switch_hold_alpha": float(pickup_switch_hold_alpha),
                    "pickup_signal_floor_dbfs": float(pickup_signal_floor_dbfs),
                    "pickup_activity_margin_db": float(pickup_activity_margin_db),
                    "pickup_activity_peak_margin_db": float(pickup_activity_peak_margin_db),
                    "pickup_activity_hold_ms": float(pickup_activity_hold_s * 1000.0),
                    "live_pickup_reference_enabled": bool(live_pickup_reference_enabled),
                    "live_pickup_reference_dir": str(live_pickup_reference_dir),
                    "live_pickup_reference_labels": list(live_pickup_reference_library.get("labels", [])),
                    "live_pickup_reference_counts": dict(live_pickup_reference_library.get("label_counts", {})),
                    "live_pickup_reference_classifier": {
                        "enabled": bool(dict(live_pickup_reference_library.get("sklearn", {})).get("enabled", False)),
                        "routes": sorted(dict(dict(live_pickup_reference_library.get("sklearn", {})).get("models", {}))),
                    },
                    "live_pickup_reference_amp_weight": float(live_pickup_reference_amp_weight),
                    "live_pickup_reference_margin": float(live_pickup_reference_margin),
                    "live_pickup_reference_max_distance": float(live_pickup_reference_max_distance),
                    "note": "Machine-readable graph telemetry for MLX conditioning, not chart pixels.",
                }
            )
            + "\n"
        )
        feature_log_handle.flush()

    def callback(indata, frames, time_info, status):
        del time_info
        captured = np.column_stack(
            [
                indata[:, source_channels[0]],
                indata[:, source_channels[1]],
            ]
        ).astype(np.float64, copy=False)
        captured = np.nan_to_num(captured, nan=0.0, posinf=0.0, neginf=0.0)
        with state_lock:
            history = state["history"]
            if frames >= history_samples:
                history[:] = captured[-history_samples:]
            else:
                history[:-frames] = history[frames:]
                history[-frames:] = captured
            state["status"] = str(status) if status else ""

    main = QtWidgets.QMainWindow()
    main.setWindowTitle("Tone Capture Live Scope - PyQtGraph")
    central = QtWidgets.QWidget()
    layout = QtWidgets.QGridLayout(central)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    main.setCentralWidget(central)

    plots_panel = QtWidgets.QWidget()
    plots_layout = QtWidgets.QGridLayout(plots_panel)
    plots_layout.setContentsMargins(0, 0, 0, 0)
    plots_layout.setSpacing(8)
    wave_plot = pg.PlotWidget(title="Live Waveform")
    level_plot = pg.PlotWidget(title="Input Level Target Range")
    spectrum_title = (
        f"Live Frequency Spectrum - pickup-sensitive x{pickup_view_boost:.1f}"
        if pickup_frequency_sensitivity
        else "Live Frequency Spectrum"
    )
    spectrum_plot = pg.PlotWidget(title=spectrum_title)
    diff_plot = pg.PlotWidget(title="Tone Difference: Amp/Mic minus DI")
    # Live scope layout contract: the bottom metrics block must fit without
    # vertical scrolling so all four graphs stay visible on a 720px-tall screen.
    metric_card_columns = 4
    metric_card_rows = 2
    metric_card_height = 96
    metric_layout_spacing = 6
    metric_panel_height = metric_card_rows * metric_card_height + metric_layout_spacing + 4
    metric_scroll_height = metric_panel_height + 2
    if metric_scroll_height > 204:
        raise RuntimeError("Live scope metric panel must stay at or below 204px.")
    metrics_panel = QtWidgets.QWidget()
    metrics_layout = QtWidgets.QGridLayout(metrics_panel)
    metrics_layout.setContentsMargins(2, 2, 2, 2)
    metrics_layout.setSpacing(metric_layout_spacing)
    metrics_scroll = QtWidgets.QScrollArea()
    metrics_scroll.setWidgetResizable(True)
    metrics_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    metrics_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    metrics_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    metrics_scroll.setWidget(metrics_panel)
    metrics_panel.setFixedHeight(metric_panel_height)
    metrics_scroll.setFixedHeight(metric_scroll_height)
    metrics_scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
    level_status_label = QtWidgets.QLabel("")
    level_status_label.setStyleSheet(
        "QLabel { background: #0f172a; color: #e5e7eb; border: 1px solid #334155; "
        "border-radius: 6px; padding: 6px; font-family: Menlo; font-size: 10px; }"
    )
    level_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
    level_status_label.setWordWrap(True)
    level_status_label.setMinimumHeight(40)
    level_status_label.setMaximumHeight(52)
    metric_labels = []
    for index in range(8):
        label = QtWidgets.QLabel("")
        label.setMinimumWidth(0)
        label.setStyleSheet(
            "QLabel { background: #111827; color: #e5e7eb; border: 1px solid #374151; "
            "border-radius: 6px; padding: 4px; font-family: Menlo; font-size: 9px; }"
        )
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        label.setWordWrap(False)
        label.setFixedHeight(metric_card_height)
        label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        metrics_layout.addWidget(label, index // metric_card_columns, index % metric_card_columns)
        metric_labels.append(label)
    for column in range(metric_card_columns):
        metrics_layout.setColumnStretch(column, 1)
    for row in range(metric_card_rows):
        metrics_layout.setRowMinimumHeight(row, metric_card_height)

    plots_layout.addWidget(wave_plot, 0, 0)
    plots_layout.addWidget(spectrum_plot, 0, 1)
    plots_layout.addWidget(diff_plot, 1, 0)
    plots_layout.addWidget(level_plot, 1, 1)
    plots_layout.setRowStretch(0, 1)
    plots_layout.setRowStretch(1, 1)
    plots_layout.setColumnStretch(0, 1)
    plots_layout.setColumnStretch(1, 1)

    splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
    splitter.addWidget(plots_panel)
    splitter.addWidget(metrics_scroll)
    splitter.setStretchFactor(0, 5)
    splitter.setStretchFactor(1, 2)
    splitter.setChildrenCollapsible(False)

    def set_scope_splitter_sizes(total_height: int) -> None:
        splitter.setSizes([max(360, total_height - metric_scroll_height - 78), metric_scroll_height])

    layout.addWidget(level_status_label, 0, 0)
    layout.addWidget(splitter, 1, 0)
    layout.setRowStretch(0, 0)
    layout.setRowStretch(1, 1)

    for plot in (wave_plot, level_plot, spectrum_plot, diff_plot):
        plot.setBackground("#020617")
        plot.showGrid(x=True, y=True, alpha=0.24)
        plot.setMinimumHeight(180)
        plot.getViewBox().setDefaultPadding(0.0)

    wave_plot.setLabel("bottom", "Time", units="ms")
    wave_plot.setLabel("left", "Stacked Inputs")
    wave_plot.setYRange(
        float(np.min(wave_offsets) - args.amplitude_range * 1.35),
        float(np.max(wave_offsets) + args.amplitude_range * 1.35),
        padding=0.0,
    )
    wave_plot.setXRange(float(time_ms[0]), 0.0, padding=0.0)
    for offset, color in zip(wave_offsets, colors):
        wave_plot.addLine(y=float(offset), pen=pg.mkPen(color, width=1, style=QtCore.Qt.PenStyle.DotLine))
        wave_plot.addLine(y=float(offset + args.clip_guard), pen=pg.mkPen("#7f1d1d", width=1))
        wave_plot.addLine(y=float(offset - args.clip_guard), pen=pg.mkPen("#7f1d1d", width=1))
    wave_curves = [
        wave_plot.plot(
            time_ms,
            np.full(window_samples, wave_offsets[i]),
            pen=pg.mkPen(colors[i], width=1.35),
            name=source_names[i],
        )
        for i in range(2)
    ]

    level_plot.setLabel("bottom", "Peak / RMS", units="dBFS")
    level_plot.setLabel("left", "Channel")
    level_plot.setXRange(level_floor_db, 0.0, padding=0.0)
    level_plot.setYRange(-0.75, 1.85, padding=0.0)
    level_plot.getAxis("left").setTicks([[(1.0, "Clean DI"), (0.0, "Amp/Mic")]])
    level_plot.addLine(x=-0.5, pen=pg.mkPen("#ef4444", width=1.4, style=QtCore.Qt.PenStyle.DashLine))
    level_plot.addLine(x=usable_low, pen=pg.mkPen("#64748b", width=1.0, style=QtCore.Qt.PenStyle.DotLine))
    level_plot.addLine(x=usable_high, pen=pg.mkPen("#64748b", width=1.0, style=QtCore.Qt.PenStyle.DotLine))
    for label_text, x_pos in [
        ("quiet", (level_floor_db + usable_low) / 2.0),
        ("usable", (usable_low + usable_high) / 2.0),
        ("hot", (usable_high - 0.5) / 2.0),
        ("clip", -0.5),
    ]:
        marker = pg.TextItem(text=label_text, color="#cbd5e1", anchor=(0.5, 1.0))
        marker.setPos(float(x_pos), 1.73)
        level_plot.addItem(marker)
    usable_zone = pg.BarGraphItem(
        x=[(usable_low + usable_high) / 2.0, (usable_low + usable_high) / 2.0],
        y=[1.0, 0.0],
        width=[usable_high - usable_low, usable_high - usable_low],
        height=0.72,
        brush=pg.mkBrush(34, 197, 94, 55),
        pen=pg.mkPen(34, 197, 94, 80),
    )
    ideal_zone = pg.BarGraphItem(
        x=[(di_ideal_low + di_ideal_high) / 2.0, (mic_ideal_low + mic_ideal_high) / 2.0],
        y=[1.0, 0.0],
        width=[di_ideal_high - di_ideal_low, mic_ideal_high - mic_ideal_low],
        height=0.44,
        brush=pg.mkBrush(34, 197, 94, 105),
        pen=pg.mkPen(134, 239, 172, 150),
    )
    hot_zone = pg.BarGraphItem(
        x=[(usable_high - 0.5) / 2.0, (usable_high - 0.5) / 2.0],
        y=[1.0, 0.0],
        width=[-0.5 - usable_high, -0.5 - usable_high],
        height=0.72,
        brush=pg.mkBrush(251, 146, 60, 45),
        pen=pg.mkPen(251, 146, 60, 80),
    )
    level_plot.addItem(usable_zone)
    level_plot.addItem(hot_zone)
    level_plot.addItem(ideal_zone)
    level_bars = pg.BarGraphItem(
        x=[level_floor_db, level_floor_db],
        y=[1.0, 0.0],
        width=[0.0, 0.0],
        height=0.30,
        brushes=[pg.mkBrush(colors[0]), pg.mkBrush(colors[1])],
        pens=[pg.mkPen(colors[0]), pg.mkPen(colors[1])],
    )
    level_plot.addItem(level_bars)
    peak_points = pg.ScatterPlotItem(
        x=[level_floor_db, level_floor_db],
        y=[1.0, 0.0],
        size=13,
        symbol="o",
        brush=[pg.mkBrush(colors[0]), pg.mkBrush(colors[1])],
        pen=pg.mkPen("#f8fafc", width=1.0),
    )
    rms_points = pg.ScatterPlotItem(
        x=[level_floor_db, level_floor_db],
        y=[1.0, 0.0],
        size=11,
        symbol="d",
        brush=[pg.mkBrush("#0f172a"), pg.mkBrush("#0f172a")],
        pen=[pg.mkPen(colors[0], width=1.6), pg.mkPen(colors[1], width=1.6)],
    )
    p999_points = pg.ScatterPlotItem(
        x=[level_floor_db, level_floor_db],
        y=[1.0, 0.0],
        size=12,
        symbol="t",
        brush=[pg.mkBrush("#fde68a"), pg.mkBrush("#fde68a")],
        pen=pg.mkPen("#78350f", width=1.0),
    )
    level_plot.addItem(peak_points)
    level_plot.addItem(rms_points)
    level_plot.addItem(p999_points)
    output_delta_text_items = [
        pg.TextItem(text="dOut +0.0 dB", color=colors[0], anchor=(1.0, 0.5)),
        pg.TextItem(text="dOut +0.0 dB", color=colors[1], anchor=(1.0, 0.5)),
    ]
    output_delta_text_items[0].setPos(-1.5, 1.38)
    output_delta_text_items[1].setPos(-1.5, -0.38)
    for item in output_delta_text_items:
        level_plot.addItem(item)
    level_status_label.setText(
        f"Profile {args.level_profile} | usable {usable_low:.0f} to {usable_high:.0f} dBFS | "
        f"DI ideal {di_ideal_low:.0f} to {di_ideal_high:.0f} | "
        f"mic ideal {mic_ideal_low:.0f} to {mic_ideal_high:.0f} | "
        "circle=peak triangle=p99.9 diamond=RMS"
    )

    spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
    spectrum_plot.setLabel("left", "Magnitude", units="dBFS")
    spectrum_plot.setXRange(plot_min_freq, plot_max_freq, padding=0.0)
    spectrum_plot.setYRange(float(args.min_db), float(args.max_db), padding=0.0)
    spectrum_plot.enableAutoRange(x=False, y=False)
    spectrum_plot.getAxis("bottom").setTicks([frequency_ticks])
    spectrum_peak_curves = [
        spectrum_plot.plot(
            plot_freqs,
            np.full(len(display_freqs), args.min_db),
            pen=pg.mkPen(colors[i], width=0.9),
            name=f"{source_names[i]} peak hold",
        )
        for i in range(2)
    ]
    for curve in spectrum_peak_curves:
        curve.setOpacity(0.35)
    spectrum_curves = [
        spectrum_plot.plot(
            plot_freqs,
            np.full(len(display_freqs), args.min_db),
            pen=pg.mkPen(colors[i], width=1.55),
            name=source_names[i],
        )
        for i in range(2)
    ]
    rolloff_lines = [
        pg.InfiniteLine(
            pos=frequency_x(1000.0),
            angle=90,
            movable=False,
            pen=pg.mkPen(colors[i], width=2.0, style=QtCore.Qt.PenStyle.DashLine),
        )
        for i in range(2)
    ]
    for line in rolloff_lines:
        spectrum_plot.addItem(line)
    frequency_markers = [
        (80.0, "low"),
        (250.0, "body"),
        (750.0, "mid"),
        (1500.0, "upper"),
        (3000.0, "bite"),
        (6000.0, "air"),
    ]
    analysis_bands = [
        ("low", 80.0, 250.0, pg.mkBrush(14, 165, 233, 18)),
        ("body", 250.0, 750.0, pg.mkBrush(34, 197, 94, 16)),
        ("mid", 750.0, 2000.0, pg.mkBrush(234, 179, 8, 15)),
        ("bite", 2500.0, 5000.0, pg.mkBrush(249, 115, 22, 16)),
        ("air", 5000.0, 10000.0, pg.mkBrush(168, 85, 247, 16)),
    ]
    for _, band_low, band_high, band_brush in analysis_bands:
        if band_high < min_freq or band_low > max_freq:
            continue
        region_low = frequency_x(max(band_low, min_freq))
        region_high = frequency_x(min(band_high, max_freq))
        for plot in (spectrum_plot, diff_plot):
            region = pg.LinearRegionItem(
                values=(region_low, region_high),
                movable=False,
                brush=band_brush,
                pen=pg.mkPen((0, 0, 0, 0)),
            )
            region.setZValue(-100)
            plot.addItem(region)
    for marker_hz, marker_label in frequency_markers:
        if min_freq <= marker_hz <= max_freq:
            marker_x = frequency_x(marker_hz)
            spectrum_plot.addLine(x=marker_x, pen=pg.mkPen("#64748b", width=1, style=QtCore.Qt.PenStyle.DotLine))
            diff_plot.addLine(x=marker_x, pen=pg.mkPen("#64748b", width=1, style=QtCore.Qt.PenStyle.DotLine))
            label = pg.TextItem(text=marker_label, color="#94a3b8", anchor=(0.5, 1.0))
            label.setPos(marker_x, args.max_db - 2.0)
            spectrum_plot.addItem(label)

    diff_plot.setLabel("bottom", "Frequency", units="Hz")
    diff_plot.setLabel("left", "Difference", units="dB")
    diff_plot.setXRange(plot_min_freq, plot_max_freq, padding=0.0)
    diff_plot.setYRange(float(args.diff_min_db), float(args.diff_max_db), padding=0.0)
    diff_plot.enableAutoRange(x=False, y=False)
    diff_plot.getAxis("bottom").setTicks([frequency_ticks])
    diff_plot.addLine(y=0.0, pen=pg.mkPen("#94a3b8", width=1))
    diff_plot.addLine(y=3.0, pen=pg.mkPen("#334155", width=1, style=QtCore.Qt.PenStyle.DotLine))
    diff_plot.addLine(y=-3.0, pen=pg.mkPen("#334155", width=1, style=QtCore.Qt.PenStyle.DotLine))
    diff_curve = diff_plot.plot(
        plot_freqs,
        np.zeros(len(display_freqs)),
        pen=pg.mkPen("#c084fc", width=1.45),
        name="Amp/Mic - DI",
    )

    def spectrum_db(audio: np.ndarray) -> np.ndarray:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        segment = audio[-min(len(audio), fft_size) :]
        if len(segment) < fft_size:
            segment = np.pad(segment, (fft_size - len(segment), 0))
        window = np.hanning(fft_size)
        spectrum = np.fft.rfft(remove_dc(segment) * window)
        scale = float(np.sum(window) / 2.0 + 1e-12)
        return np.nan_to_num(
            20.0 * np.log10((np.abs(spectrum) / scale) + 1e-12),
            nan=float(args.min_db),
            posinf=float(args.max_db),
            neginf=float(args.min_db),
        )

    def averaged_spectral_power(audio: np.ndarray) -> np.ndarray:
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        audio = remove_dc(audio.astype(np.float64, copy=False))
        if len(audio) < fft_size:
            audio = np.pad(audio, (fft_size - len(audio), 0))
        hop = max(1, fft_size // 4)
        window = np.hanning(fft_size)
        scale = float(np.sum(window) / 2.0 + 1e-12)
        power = np.zeros(fft_size // 2 + 1, dtype=np.float64)
        frame_count = 0
        for start in range(0, len(audio) - fft_size + 1, hop):
            frame = audio[start : start + fft_size] * window
            spectrum = np.fft.rfft(frame)
            power += np.square(np.abs(spectrum) / scale)
            frame_count += 1
        if frame_count == 0:
            frame = audio[-fft_size:] * window
            spectrum = np.fft.rfft(frame)
            power += np.square(np.abs(spectrum) / scale)
            frame_count = 1
        return np.nan_to_num(power / max(1, frame_count), nan=0.0, posinf=0.0, neginf=0.0)

    def power_spectrum_db(power: np.ndarray) -> np.ndarray:
        return np.nan_to_num(
            10.0 * np.log10(power + 1e-18),
            nan=float(args.min_db),
            posinf=float(args.max_db),
            neginf=float(args.min_db),
        )

    def smooth_curve(key: str, values: np.ndarray, bipolar: bool = False) -> np.ndarray:
        previous = smooth_state.get(key)
        if previous is None or previous.shape != values.shape:
            smooth_state[key] = values.copy()
            return values
        fast_mask = np.abs(values) >= np.abs(previous) if bipolar else values >= previous
        alpha = np.where(fast_mask, attack_alpha, release_alpha)
        smoothed = alpha * values + (1.0 - alpha) * previous
        smooth_state[key] = smoothed
        return smoothed

    def smooth_spectrum_curve(key: str, values: np.ndarray) -> np.ndarray:
        previous = smooth_state.get(key)
        if previous is None or previous.shape != values.shape:
            smooth_state[key] = values.copy()
            return values
        change_db = np.abs(values - previous)
        alpha = np.where(change_db >= pickup_view_fast_delta_db, frequency_eye_attack, frequency_eye_release)
        smoothed = alpha * values + (1.0 - alpha) * previous
        smooth_state[key] = smoothed
        return smoothed

    def pickup_sensitive_spectrum(key: str, spectrum_values: np.ndarray) -> np.ndarray:
        if not pickup_frequency_sensitivity or pickup_view_boost <= 1.0 or pickup_view_max_delta_db <= 0.0:
            return spectrum_values

        baseline = pickup_frequency_baseline_state.get(key)
        if baseline is None or baseline.shape != spectrum_values.shape:
            pickup_frequency_baseline_state[key] = spectrum_values.copy()
            return spectrum_values

        delta = np.nan_to_num(spectrum_values - baseline, nan=0.0, posinf=0.0, neginf=0.0)
        boosted_delta = np.clip(
            delta * (pickup_view_boost - 1.0),
            -pickup_view_max_delta_db,
            pickup_view_max_delta_db,
        )
        visible = np.clip(spectrum_values + boosted_delta, float(args.min_db), float(args.max_db))

        audible_mask = (freqs >= 80.0) & (freqs <= 9000.0)
        median_change_db = float(np.median(np.abs(delta[audible_mask]))) if np.any(audible_mask) else 0.0
        baseline_alpha = pickup_view_baseline_alpha * (0.35 if median_change_db >= pickup_view_fast_delta_db else 1.0)
        pickup_frequency_baseline_state[key] = baseline * (1.0 - baseline_alpha) + spectrum_values * baseline_alpha
        return np.nan_to_num(visible, nan=float(args.min_db), posinf=float(args.max_db), neginf=float(args.min_db))

    def peak_hold_curve(key: str, values: np.ndarray) -> np.ndarray:
        previous = peak_hold_state.get(key)
        if previous is None or previous.shape != values.shape:
            peak_hold_state[key] = values.copy()
            return values
        held = np.maximum(values, previous - spectrum_peak_decay_db)
        peak_hold_state[key] = held
        return held

    def interp_display(values: np.ndarray) -> np.ndarray:
        return np.interp(display_freqs, freqs, values)

    def rms_dbfs(audio: np.ndarray) -> float:
        return float(20.0 * np.log10(rms(audio) + 1e-12))

    def percentile_abs_dbfs(audio: np.ndarray, percentile: float = 99.9) -> float:
        return float(20.0 * np.log10(float(np.percentile(np.abs(audio), percentile)) + 1e-12))

    def short_delay_metrics(di_audio: np.ndarray, mic_audio: np.ndarray) -> tuple[float, float]:
        a = remove_dc(di_audio)
        b = remove_dc(mic_audio)
        if len(a) < 8 or len(b) < 8 or rms(a) < 1e-7 or rms(b) < 1e-7:
            return 0.0, 0.0
        n = min(len(a), len(b))
        a = a[-n:]
        b = b[-n:]
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        correlation_curve = correlate(b, a, mode="full", method="fft")
        lags = np.arange(-n + 1, n, dtype=np.int64)
        max_lag = max(1, int(round(sample_rate * 0.010)))
        mask = np.abs(lags) <= max_lag
        limited_corr = correlation_curve[mask]
        limited_lags = lags[mask]
        if len(limited_corr) == 0:
            return 0.0, 0.0
        best_index = int(np.argmax(np.abs(limited_corr)))
        return float(limited_lags[best_index] * 1000.0 / sample_rate), float(limited_corr[best_index])

    def zero_crossing_rate(audio: np.ndarray) -> float:
        centered = remove_dc(audio)
        if len(centered) < 2:
            return 0.0
        signs = np.signbit(centered)
        return float(np.count_nonzero(signs[1:] != signs[:-1]) / max(len(centered) / sample_rate, 1e-12))

    def max_slew_rate(audio: np.ndarray) -> float:
        if len(audio) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(audio))) * sample_rate)

    def band_mean(values: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    def band_peak_hz(values: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0
        band_freqs = freqs[mask]
        band_values = values[mask]
        return float(band_freqs[int(np.argmax(band_values))])

    def power_band_peak_hz(power: np.ndarray, low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return 0.0
        band_freqs = freqs[mask]
        band_power = power[mask]
        return float(band_freqs[int(np.argmax(band_power))])

    def dominant_diff_band(band_values: dict[str, float]) -> tuple[str, float]:
        if not band_values:
            return "n/a", 0.0
        name = max(band_values, key=lambda key: abs(band_values[key]))
        return name, float(band_values[name])

    def spectral_power(audio: np.ndarray) -> np.ndarray:
        segment = audio[-min(len(audio), fft_size) :]
        if len(segment) < fft_size:
            segment = np.pad(segment, (fft_size - len(segment), 0))
        window = np.hanning(fft_size)
        return np.abs(np.fft.rfft(remove_dc(segment) * window)) ** 2

    def spectral_centroid(power: np.ndarray) -> float:
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        total = float(np.sum(power[mask]) + 1e-12)
        return float(np.sum(freqs[mask] * power[mask]) / total)

    def spectral_rolloff(power: np.ndarray, fraction: float = 0.85) -> float:
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        usable_freqs = freqs[mask]
        usable_power = power[mask]
        if len(usable_power) == 0:
            return 0.0
        cumulative = np.cumsum(usable_power)
        threshold = float(cumulative[-1] * fraction)
        return float(usable_freqs[min(int(np.searchsorted(cumulative, threshold)), len(usable_freqs) - 1)])

    def band_percent(power: np.ndarray, low_hz: float, high_hz: float) -> float:
        total_mask = (freqs >= 80.0) & (freqs <= 10000.0)
        band_mask = (freqs >= low_hz) & (freqs <= high_hz)
        total = float(np.sum(power[total_mask]) + 1e-12)
        return float(100.0 * np.sum(power[band_mask]) / total)

    def tone_rolloff_status(rolloff_hz: float, bite_pct: float, air_pct: float) -> str:
        high_pct = bite_pct + air_pct
        if rolloff_hz < 2800.0 or high_pct < 7.0:
            return "rolled off / dark"
        if rolloff_hz < 4500.0 or high_pct < 13.0:
            return "mellow"
        if rolloff_hz > 7500.0 and high_pct > 24.0:
            return "bright / open"
        return "balanced"

    def pickup_voicing_status(
        centroid_hz: float,
        low_pct: float,
        body_pct: float,
        upper_pct: float,
        bite_pct: float,
        air_pct: float,
        resonant_peak_hz: float,
        output_delta_db: float,
        output_status: str,
    ) -> str:
        low_body = low_pct + body_pct
        high = bite_pct + air_pct
        body_bias = body_pct + low_pct * 0.65
        bright_bias = upper_pct * 0.70 + bite_pct + air_pct * 1.15
        upper_high = upper_pct + high
        output_push = max(0.0, output_delta_db)
        output_cut = max(0.0, -output_delta_db)
        if output_status in {"hotter", "up"}:
            output_push += 0.7
        elif output_status in {"lower", "down"}:
            output_cut += 0.7

        bridge_score = 0.0
        bridge_score += max(0.0, (centroid_hz - 3600.0) / 1600.0)
        bridge_score += max(0.0, (resonant_peak_hz - 1350.0) / 2600.0)
        bridge_score += max(0.0, (bright_bias - body_bias * 0.58) / 8.0)
        bridge_score += max(0.0, (upper_high - low_body * 0.42) / 9.0)
        bridge_score += min(1.4, output_push / 2.1)

        neck_score = 0.0
        neck_score += max(0.0, (4800.0 - centroid_hz) / 1700.0)
        neck_score += max(0.0, (2600.0 - resonant_peak_hz) / 1800.0)
        neck_score += max(0.0, (body_bias - bright_bias * 0.86) / 7.0)
        neck_score += min(0.9, output_cut / 3.0)

        if bridge_score >= 1.05 and bridge_score >= neck_score + 0.25:
            return "bridge-like / bright"
        if (
            centroid_hz <= 4300.0
            and neck_score >= bridge_score + 0.20
            and (low_body >= high * 1.02 or body_bias >= bright_bias * 1.08)
        ):
            return "neck-like / warm"
        if centroid_hz <= 4700.0 and neck_score >= 1.10 and body_bias >= bright_bias * 1.18:
            return "neck-like / warm"
        if air_pct < 3.0 and bite_pct < 10.0:
            return "split/rolled-off-like"
        if bridge_score >= 0.85 and output_status in {"hotter", "up"}:
            return "bridge-like / bright"
        return "middle / balanced"

    def pickup_signal_activity(
        key: str,
        rms_value_db: float,
        p999_value_db: float,
        peak_value_db: float,
        transient_value: float,
        now_s: float,
    ) -> dict[str, float | bool | str]:
        state_entry = pickup_activity_state.get(key)
        if state_entry is None:
            first_active = (
                (rms_value_db >= pickup_signal_floor_dbfs + 3.0 and transient_value >= 0.50)
                or (p999_value_db >= pickup_signal_floor_dbfs + 10.0 and transient_value >= 0.75)
                or (peak_value_db >= pickup_signal_floor_dbfs + 18.0 and transient_value >= 0.50)
            )
            pickup_activity_state[key] = {
                "rms_floor": float(rms_value_db),
                "p999_floor": float(p999_value_db),
                "peak_floor": float(peak_value_db),
                "hold_until": now_s + pickup_activity_hold_s if first_active else 0.0,
                "frames": 1.0,
            }
            return {
                "active": bool(first_active),
                "label": "playing" if first_active else "waiting",
                "score": 1.0 if first_active else 0.0,
                "rms_margin": 0.0,
                "p999_margin": 0.0,
                "peak_margin": 0.0,
                "rms_floor": float(rms_value_db),
            }

        rms_floor = float(state_entry.get("rms_floor", rms_value_db))
        p999_floor = float(state_entry.get("p999_floor", p999_value_db))
        peak_floor = float(state_entry.get("peak_floor", peak_value_db))
        frames_seen = int(float(state_entry.get("frames", 1.0))) + 1
        state_entry["frames"] = float(frames_seen)
        rms_margin = float(rms_value_db - rms_floor)
        p999_margin = float(p999_value_db - p999_floor)
        peak_margin = float(peak_value_db - peak_floor)
        absolute_active = (
            (rms_value_db >= pickup_signal_floor_dbfs + 3.0 and transient_value >= 0.50)
            or (p999_value_db >= pickup_signal_floor_dbfs + 10.0 and transient_value >= 0.75)
            or (peak_value_db >= pickup_signal_floor_dbfs + 18.0 and transient_value >= 0.50)
        )
        relative_active = (
            rms_margin >= pickup_activity_margin_db
            or p999_margin >= pickup_activity_peak_margin_db
            or peak_margin >= pickup_activity_peak_margin_db + 4.0
            or (transient_value >= 1.5 and p999_margin >= pickup_activity_peak_margin_db * 0.45)
        )
        activity_score = max(
            (rms_value_db - pickup_signal_floor_dbfs) / 5.0,
            rms_margin / pickup_activity_margin_db,
            p999_margin / pickup_activity_peak_margin_db,
            peak_margin / (pickup_activity_peak_margin_db + 4.0),
            transient_value / 3.0,
            0.0,
        )
        active_now = (
            relative_active
            or (
                absolute_active
                and (
                    frames_seen < 8
                    or rms_margin >= 1.0
                    or p999_margin >= 3.0
                    or transient_value >= 1.0
                )
            )
        )
        if active_now:
            state_entry["hold_until"] = now_s + pickup_activity_hold_s

        active = active_now or now_s < float(state_entry.get("hold_until", 0.0))
        label = "playing" if active else "waiting"
        floor_alpha = 0.004 if active else 0.08
        state_entry["rms_floor"] = rms_floor * (1.0 - floor_alpha) + float(rms_value_db) * floor_alpha
        state_entry["p999_floor"] = p999_floor * (1.0 - floor_alpha) + float(p999_value_db) * floor_alpha
        state_entry["peak_floor"] = peak_floor * (1.0 - floor_alpha) + float(peak_value_db) * floor_alpha
        return {
            "active": bool(active),
            "label": label,
            "score": float(activity_score),
            "rms_margin": float(rms_margin),
            "p999_margin": float(p999_margin),
            "peak_margin": float(peak_margin),
            "rms_floor": float(state_entry["rms_floor"]),
        }

    def pickup_change_metrics(
        key: str,
        vector: np.ndarray,
        rms_value_db: float,
        active_signal: bool,
    ) -> dict[str, float | str]:
        if not active_signal:
            return {
                "score": 0.0,
                "rolloff_delta": 0.0,
                "resonant_delta": 0.0,
                "upper_delta": 0.0,
                "high_delta": 0.0,
                "level_delta": 0.0,
                "body_to_bite_delta": 0.0,
                "label": "waiting for signal",
            }

        baseline = voicing_baseline_state.get(key)
        if baseline is None or baseline.shape != vector.shape:
            voicing_baseline_state[key] = vector.copy()
            return {
                "score": 0.0,
                "rolloff_delta": 0.0,
                "resonant_delta": 0.0,
                "upper_delta": 0.0,
                "high_delta": 0.0,
                "level_delta": 0.0,
                "body_to_bite_delta": 0.0,
                "label": "baseline set",
            }

        delta = vector - baseline
        rolloff_delta = float(delta[1])
        upper_delta = float(delta[5])
        high_delta = float(delta[6] + delta[7])
        resonant_delta = float(delta[8])
        body_to_bite_delta = float(delta[9])
        level_delta = float(delta[10])
        score = float(
            abs(delta[0]) / 900.0
            + abs(rolloff_delta) / 1200.0
            + abs(resonant_delta) / 1600.0
            + abs(upper_delta) / 5.0
            + abs(high_delta) / 5.0
            + abs(delta[2] + delta[3]) / 7.0
            + abs(level_delta) / 7.0
        )

        if level_delta >= 2.5 and (high_delta >= 2.0 or upper_delta >= 2.0 or resonant_delta >= 550.0):
            label = "blower-like: hotter/brighter"
        elif level_delta <= -2.5 and (high_delta <= -1.5 or upper_delta <= -1.5 or resonant_delta <= -550.0):
            label = "blower off / lower output"
        elif level_delta < -4.0 and high_delta < -2.0:
            label = "lower output / split-like"
        elif high_delta < -4.0 or rolloff_delta < -1200.0:
            label = "darker / tone down"
        elif high_delta > 4.0 or rolloff_delta > 1200.0:
            label = "brighter / bridge-like"
        elif score >= 1.25:
            label = "strong pickup/tone shift"
        elif score >= 0.55:
            label = "subtle pickup/tone shift"
        else:
            label = "stable"

        # Keep the reference slow so pickup/tone switches remain visible long enough to read.
        alpha = 0.0025 if score >= 0.55 else 0.012
        voicing_baseline_state[key] = baseline * (1.0 - alpha) + vector * alpha
        return {
            "score": score,
            "rolloff_delta": rolloff_delta,
            "resonant_delta": resonant_delta,
            "upper_delta": upper_delta,
            "high_delta": high_delta,
            "level_delta": level_delta,
            "body_to_bite_delta": body_to_bite_delta,
            "label": label,
        }

    def transient_rate(audio: np.ndarray) -> float:
        if len(audio) < 4:
            return 0.0
        envelope = np.abs(remove_dc(audio))
        diff = np.diff(envelope)
        threshold = float(np.percentile(np.abs(diff), 97.5) + 1e-12)
        hits = np.flatnonzero(diff > threshold)
        min_gap = max(1, int(round(sample_rate * 0.012)))
        count = 0
        last = -min_gap
        for index in hits:
            if int(index) - last >= min_gap:
                count += 1
                last = int(index)
        return float(count / max(len(audio) / sample_rate, 1e-12))

    def clipping_percent(audio: np.ndarray) -> float:
        return float(100.0 * np.mean(np.abs(audio) >= args.clip_guard))

    def output_level_change(key: str, level_db: float, active_signal: bool) -> tuple[float, str]:
        if not active_signal:
            return 0.0, "waiting"
        baseline = output_level_baseline_state.get(key)
        if baseline is None:
            output_level_baseline_state[key] = float(level_db)
            return 0.0, "baseline"

        delta = float(level_db - baseline)
        if delta >= output_hot_delta_db:
            label = "hotter"
        elif delta <= -output_hot_delta_db:
            label = "lower"
        elif delta >= output_change_delta_db:
            label = "up"
        elif delta <= -output_change_delta_db:
            label = "down"
        else:
            label = "stable"

        # Hold obvious pickup/blower jumps long enough to read; follow normal playing drift faster.
        alpha = output_hold_alpha if abs(delta) >= output_change_delta_db else output_baseline_alpha
        output_level_baseline_state[key] = baseline * (1.0 - alpha) + float(level_db) * alpha
        return delta, label

    def pickup_switch_event(
        key: str,
        vector: np.ndarray,
        rms_value_db: float,
        now_s: float,
        active_signal: bool,
    ) -> dict[str, float | str]:
        state_entry = pickup_switch_state.get(key)
        if not active_signal:
            if state_entry is not None:
                state_entry["label"] = "waiting signal"
                state_entry["hold_until"] = 0.0
            return {
                "label": "waiting signal",
                "score": 0.0,
                "level_delta": 0.0,
                "high_delta": 0.0,
                "upper_delta": 0.0,
                "resonant_delta": 0.0,
                "centroid_delta": 0.0,
            }

        if state_entry is None or not isinstance(state_entry.get("vector"), np.ndarray):
            pickup_switch_state[key] = {
                "vector": vector.copy(),
                "label": "watching",
                "hold_until": 0.0,
                "score": 0.0,
                "level_delta": 0.0,
                "high_delta": 0.0,
                "upper_delta": 0.0,
                "resonant_delta": 0.0,
                "centroid_delta": 0.0,
            }
            return {
                "label": "watching",
                "score": 0.0,
                "level_delta": 0.0,
                "high_delta": 0.0,
                "upper_delta": 0.0,
                "resonant_delta": 0.0,
                "centroid_delta": 0.0,
            }

        reference = state_entry["vector"]
        if not isinstance(reference, np.ndarray) or reference.shape != vector.shape:
            pickup_switch_state[key]["vector"] = vector.copy()
            return {
                "label": "watching",
                "score": 0.0,
                "level_delta": 0.0,
                "high_delta": 0.0,
                "upper_delta": 0.0,
                "resonant_delta": 0.0,
                "centroid_delta": 0.0,
            }

        delta = vector - reference
        centroid_delta = float(delta[0])
        rolloff_delta = float(delta[1])
        upper_delta = float(delta[5])
        high_delta = float(delta[6] + delta[7])
        resonant_delta = float(delta[8])
        level_delta = float(delta[10])
        score = float(
            abs(level_delta) / max(output_change_delta_db, 0.05)
            + abs(high_delta) / 0.85
            + abs(upper_delta) / 0.85
            + abs(centroid_delta) / 420.0
            + abs(rolloff_delta) / 750.0
            + abs(resonant_delta) / 420.0
        ) / 3.0

        label = "watching"
        detected = score >= pickup_switch_score_threshold
        if detected:
            brighter = high_delta >= 0.45 or upper_delta >= 0.45 or centroid_delta >= 220.0 or resonant_delta >= 220.0
            darker = high_delta <= -0.45 or upper_delta <= -0.45 or centroid_delta <= -220.0 or resonant_delta <= -220.0
            hotter = level_delta >= output_change_delta_db
            lower = level_delta <= -output_change_delta_db
            if hotter and brighter:
                label = "bridge/hotter switch"
            elif brighter:
                label = "bridge/brighter switch"
            elif lower and darker:
                label = "neck/lower switch"
            elif darker:
                label = "neck/darker switch"
            elif hotter:
                label = "output hotter switch"
            elif lower:
                label = "output lower switch"
            else:
                label = "tone switch"
            state_entry["hold_until"] = now_s + pickup_switch_hold_s
        elif now_s < float(state_entry.get("hold_until", 0.0)):
            label = str(state_entry.get("label", "watching"))
            score = float(state_entry.get("score", score))
            level_delta = float(state_entry.get("level_delta", level_delta))
            high_delta = float(state_entry.get("high_delta", high_delta))
            upper_delta = float(state_entry.get("upper_delta", upper_delta))
            resonant_delta = float(state_entry.get("resonant_delta", resonant_delta))
            centroid_delta = float(state_entry.get("centroid_delta", centroid_delta))

        alpha = pickup_switch_hold_alpha if label != "watching" else pickup_switch_baseline_alpha
        state_entry["vector"] = reference * (1.0 - alpha) + vector * alpha
        state_entry["label"] = label
        state_entry["score"] = score
        state_entry["level_delta"] = level_delta
        state_entry["high_delta"] = high_delta
        state_entry["upper_delta"] = upper_delta
        state_entry["resonant_delta"] = resonant_delta
        state_entry["centroid_delta"] = centroid_delta
        return {
            "label": label,
            "score": score,
            "level_delta": level_delta,
            "high_delta": high_delta,
            "upper_delta": upper_delta,
            "resonant_delta": resonant_delta,
            "centroid_delta": centroid_delta,
        }

    def resolve_pickup_event(di_event: dict[str, float | str], mic_event: dict[str, float | str]) -> str:
        labels = [str(di_event.get("label", "watching")), str(mic_event.get("label", "watching"))]
        active_labels = [label for label in labels if label not in {"watching", "waiting signal"}]
        if not active_labels:
            return "watching"
        if any("bridge" in label for label in active_labels):
            return "bridge switch"
        if any("neck" in label for label in active_labels):
            return "neck switch"
        if any("hotter" in label for label in active_labels):
            return "output hotter"
        if any("lower" in label for label in active_labels):
            return "output lower"
        return active_labels[0]

    def pickup_channel_reliable(
        source_rms_db: float,
        output_status: str,
        change_label: str,
        active_signal: bool,
    ) -> bool:
        if not active_signal:
            return False
        if output_status == "waiting" or change_label == "waiting for signal":
            return False
        return True

    def resolve_pickup_statuses(
        di_raw_status: str,
        mic_raw_status: str,
        di_source_rms_db: float,
        mic_source_rms_db: float,
        di_output_status: str,
        mic_output_status: str,
        di_change_label: str,
        mic_change_label: str,
        di_active_signal: bool,
        mic_active_signal: bool,
    ) -> tuple[str, str, str]:
        di_reliable = pickup_channel_reliable(di_source_rms_db, di_output_status, di_change_label, di_active_signal)
        mic_reliable = pickup_channel_reliable(mic_source_rms_db, mic_output_status, mic_change_label, mic_active_signal)
        reliable_statuses = []
        if di_reliable:
            reliable_statuses.append(di_raw_status)
        if mic_reliable:
            reliable_statuses.append(mic_raw_status)
        if not reliable_statuses:
            return "waiting for signal", "waiting for signal", "waiting for signal"

        if any("bridge" in status for status in reliable_statuses):
            system_status = "bridge-like / bright"
        elif any("neck" in status for status in reliable_statuses):
            system_status = "neck-like / warm"
        elif any("split" in status or "rolled-off" in status for status in reliable_statuses):
            system_status = "split/rolled-off-like"
        else:
            system_status = "middle / balanced"

        di_status = di_raw_status
        mic_status = mic_raw_status
        if not di_reliable:
            di_status = system_status
        if not mic_reliable:
            mic_status = system_status
        if system_status == "bridge-like / bright":
            if "bridge" in mic_status and "bridge" not in di_status:
                di_status = system_status
            if "bridge" in di_status and "bridge" not in mic_status:
                mic_status = system_status
        return system_status, di_status, mic_status

    def measured_channel_status(change: dict[str, float | str], output_status: str, active_signal: bool) -> str:
        if not active_signal:
            return "waiting for signal"
        label = str(change.get("label", "stable"))
        high_delta = float(change.get("high_delta", 0.0))
        upper_delta = float(change.get("upper_delta", 0.0))
        rolloff_delta = float(change.get("rolloff_delta", 0.0))
        level_delta = float(change.get("level_delta", 0.0))

        if "blower" in label or (level_delta >= output_hot_delta_db and (high_delta > 0.8 or upper_delta > 0.8)):
            return "measured blower/hotter"
        if output_status in {"hotter", "up"} and (high_delta > 0.4 or upper_delta > 0.4 or rolloff_delta > 500.0):
            return "measured hotter/brighter"
        if output_status in {"lower", "down"} and (high_delta < -0.4 or upper_delta < -0.4 or rolloff_delta < -500.0):
            return "measured lower/darker"
        if high_delta > 1.0 or upper_delta > 1.0 or rolloff_delta > 900.0:
            return "measured brighter"
        if high_delta < -1.0 or upper_delta < -1.0 or rolloff_delta < -900.0:
            return "measured darker"
        if output_status in {"hotter", "up"}:
            return "measured output up"
        if output_status in {"lower", "down"}:
            return "measured output down"
        if "shift" in label or "tone" in label:
            return "measured tone shift"
        return "measured stable"

    def measured_system_status(
        di_change: dict[str, float | str],
        mic_change: dict[str, float | str],
        di_output_status: str,
        mic_output_status: str,
        di_active_signal: bool,
        mic_active_signal: bool,
    ) -> str:
        if not di_active_signal and not mic_active_signal:
            return "waiting for signal"
        mic_status = measured_channel_status(mic_change, mic_output_status, mic_active_signal)
        di_status = measured_channel_status(di_change, di_output_status, di_active_signal)
        if mic_active_signal and mic_status != "measured stable":
            return mic_status
        if di_active_signal and di_status != "measured stable":
            return di_status
        if mic_active_signal:
            return mic_status
        return di_status

    def compact_text(value: object, limit: int = 28) -> str:
        text = str(value).replace("\n", " ").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def set_metric_label(index: int, visible: str, detail: str | None = None) -> None:
        metric_labels[index].setText(visible)
        metric_labels[index].setToolTip(detail if detail is not None else visible)

    def update() -> None:
        nonlocal feature_log_frame, last_feature_log_time

        with state_lock:
            history = state["history"].copy()
            status_text = state["status"]

        display_history = history[-window_samples:]
        response = history[-max(64, int(round(sample_rate * args.metrics_window_ms / 1000.0))) :]
        source_response = history[-source_analysis_samples:]
        di = display_history[:, 0]
        mic = display_history[:, 1]
        wave_curves[0].setData(time_ms, di + wave_offsets[0])
        wave_curves[1].setData(time_ms, mic + wave_offsets[1])

        graph_di_spec = smooth_gain_db(spectrum_db(di), args.spectrum_smoothing_bins)
        graph_mic_spec = smooth_gain_db(spectrum_db(mic), args.spectrum_smoothing_bins)
        display_di_spec = pickup_sensitive_spectrum("di_frequency_view", graph_di_spec)
        display_mic_spec = pickup_sensitive_spectrum("mic_frequency_view", graph_mic_spec)
        di_display = smooth_spectrum_curve("di_spec", interp_display(display_di_spec))
        mic_display = smooth_spectrum_curve("mic_spec", interp_display(display_mic_spec))
        spectrum_curves[0].setData(plot_freqs, di_display)
        spectrum_curves[1].setData(plot_freqs, mic_display)
        spectrum_peak_curves[0].setData(plot_freqs, peak_hold_curve("di_spec_peak", di_display))
        spectrum_peak_curves[1].setData(plot_freqs, peak_hold_curve("mic_spec_peak", mic_display))

        graph_diff = smooth_gain_db(graph_mic_spec - graph_di_spec, args.tone_diff_smoothing_bins)
        diff_display = smooth_curve("diff", interp_display(graph_diff), bipolar=True)
        diff_curve.setData(plot_freqs, np.clip(diff_display, float(args.diff_min_db), float(args.diff_max_db)))

        di_peak = peak_dbfs(response[:, 0])
        mic_peak = peak_dbfs(response[:, 1])
        di_rms = rms_dbfs(response[:, 0])
        mic_rms = rms_dbfs(response[:, 1])
        di_headroom = -di_peak
        mic_headroom = -mic_peak
        di_p999 = percentile_abs_dbfs(response[:, 0], 99.9)
        mic_p999 = percentile_abs_dbfs(response[:, 1], 99.9)
        di_crest = crest_factor(response[:, 0])
        mic_crest = crest_factor(response[:, 1])
        di_zcr = zero_crossing_rate(response[:, 0])
        mic_zcr = zero_crossing_rate(response[:, 1])
        di_slew = max_slew_rate(response[:, 0])
        mic_slew = max_slew_rate(response[:, 1])
        di_transients = transient_rate(response[:, 0])
        mic_transients = transient_rate(response[:, 1])
        di_noise_floor = percentile_abs_dbfs(response[:, 0], 50.0)
        mic_noise_floor = percentile_abs_dbfs(response[:, 1], 50.0)
        di_source_peak = peak_dbfs(source_response[:, 0])
        mic_source_peak = peak_dbfs(source_response[:, 1])
        di_source_rms = rms_dbfs(source_response[:, 0])
        mic_source_rms = rms_dbfs(source_response[:, 1])
        di_source_p999 = percentile_abs_dbfs(source_response[:, 0], 99.9)
        mic_source_p999 = percentile_abs_dbfs(source_response[:, 1], 99.9)
        di_source_transients = transient_rate(source_response[:, 0])
        mic_source_transients = transient_rate(source_response[:, 1])
        activity_now = time.monotonic()
        di_activity = pickup_signal_activity(
            "di",
            di_source_rms,
            di_source_p999,
            di_source_peak,
            max(di_transients, di_source_transients),
            activity_now,
        )
        mic_activity = pickup_signal_activity(
            "mic",
            mic_source_rms,
            mic_source_p999,
            mic_source_peak,
            max(mic_transients, mic_source_transients),
            activity_now,
        )
        di_active = bool(di_activity["active"])
        mic_active = bool(mic_activity["active"])
        di_output_level = max(di_rms, di_p999 - 9.0, di_source_rms, di_source_p999 - 9.0)
        mic_output_level = max(mic_rms, mic_p999 - 9.0, mic_source_rms, mic_source_p999 - 9.0)
        di_output_delta, di_output_status = output_level_change("di_output", di_output_level, di_active)
        mic_output_delta, mic_output_status = output_level_change("mic_output", mic_output_level, mic_active)
        delta_peak = di_peak - mic_peak
        delta_rms = di_rms - mic_rms
        di_status, _ = classify_peak_level(di_peak, args.level_profile)
        mic_status, _ = classify_peak_level(mic_peak, args.level_profile)
        match_status, _ = classify_level_match(delta_peak, args.level_profile)
        di_power = averaged_spectral_power(source_response[:, 0])
        mic_power = averaged_spectral_power(source_response[:, 1])
        analysis_di_spec = smooth_gain_db(power_spectrum_db(di_power), args.spectrum_smoothing_bins)
        analysis_mic_spec = smooth_gain_db(power_spectrum_db(mic_power), args.spectrum_smoothing_bins)
        diff = smooth_gain_db(analysis_mic_spec - analysis_di_spec, args.tone_diff_smoothing_bins)
        di_centroid = spectral_centroid(di_power)
        mic_centroid = spectral_centroid(mic_power)
        di_rolloff = spectral_rolloff(di_power)
        mic_rolloff = spectral_rolloff(mic_power)
        rolloff_lines[0].setValue(frequency_x(di_rolloff))
        rolloff_lines[1].setValue(frequency_x(mic_rolloff))
        di_peak_freq = float(display_freqs[int(np.argmax(di_display))])
        mic_peak_freq = float(display_freqs[int(np.argmax(mic_display))])
        diff_low = band_mean(diff, 80, 250)
        diff_body = band_mean(diff, 250, 750)
        diff_mid = band_mean(diff, 750, 2000)
        diff_upper = band_mean(diff, 1500, 2500)
        diff_bite = band_mean(diff, 2500, 5000)
        diff_air = band_mean(diff, 5000, 9000)
        diff_bands = {
            "low": diff_low,
            "body": diff_body,
            "mid": diff_mid,
            "upper": diff_upper,
            "bite": diff_bite,
            "air": diff_air,
        }
        dominant_band, dominant_band_db = dominant_diff_band(diff_bands)
        di_low_pct = band_percent(di_power, 80, 250)
        mic_low_pct = band_percent(mic_power, 80, 250)
        di_body_pct = band_percent(di_power, 250, 750)
        mic_body_pct = band_percent(mic_power, 250, 750)
        di_mid_pct = band_percent(di_power, 750, 2000)
        mic_mid_pct = band_percent(mic_power, 750, 2000)
        di_upper_pct = band_percent(di_power, 1500, 2500)
        mic_upper_pct = band_percent(mic_power, 1500, 2500)
        di_bite_pct = band_percent(di_power, 2500, 5000)
        mic_bite_pct = band_percent(mic_power, 2500, 5000)
        di_air_pct = band_percent(di_power, 5000, 9000)
        mic_air_pct = band_percent(mic_power, 5000, 9000)
        di_high_pct = di_bite_pct + di_air_pct
        mic_high_pct = mic_bite_pct + mic_air_pct
        di_body_to_bite = di_body_pct / max(di_bite_pct, 1e-6)
        mic_body_to_bite = mic_body_pct / max(mic_bite_pct, 1e-6)
        di_band_peak = power_band_peak_hz(di_power, 80, 9000)
        mic_band_peak = power_band_peak_hz(mic_power, 80, 9000)
        di_resonant_peak = power_band_peak_hz(di_power, 700, 6500)
        mic_resonant_peak = power_band_peak_hz(mic_power, 700, 6500)
        di_pickup_vector = live_pickup_metric_vector(
            di_centroid,
            di_rolloff,
            di_low_pct,
            di_body_pct,
            di_mid_pct,
            di_upper_pct,
            di_bite_pct,
            di_air_pct,
            di_resonant_peak,
            di_body_to_bite,
            di_source_rms,
            di_source_p999,
        )
        mic_pickup_vector = live_pickup_metric_vector(
            mic_centroid,
            mic_rolloff,
            mic_low_pct,
            mic_body_pct,
            mic_mid_pct,
            mic_upper_pct,
            mic_bite_pct,
            mic_air_pct,
            mic_resonant_peak,
            mic_body_to_bite,
            mic_source_rms,
            mic_source_p999,
        )
        centroid_gap = mic_centroid - di_centroid
        rolloff_gap = mic_rolloff - di_rolloff
        high_gap = mic_high_pct - di_high_pct
        delay_ms, short_corr = short_delay_metrics(response[:, 0], response[:, 1])
        di_dynamic_range = di_p999 - di_noise_floor
        mic_dynamic_range = mic_p999 - mic_noise_floor
        di_peak_overshoot = di_peak - di_p999
        mic_peak_overshoot = mic_peak - mic_p999
        di_tone_status = tone_rolloff_status(di_rolloff, di_bite_pct, di_air_pct)
        mic_tone_status = tone_rolloff_status(mic_rolloff, mic_bite_pct, mic_air_pct)
        di_pickup_raw_status = pickup_voicing_status(
            di_centroid,
            di_low_pct,
            di_body_pct,
            di_upper_pct,
            di_bite_pct,
            di_air_pct,
            di_resonant_peak,
            di_output_delta,
            di_output_status,
        )
        mic_pickup_raw_status = pickup_voicing_status(
            mic_centroid,
            mic_low_pct,
            mic_body_pct,
            mic_upper_pct,
            mic_bite_pct,
            mic_air_pct,
            mic_resonant_peak,
            mic_output_delta,
            mic_output_status,
        )
        di_pickup_change = pickup_change_metrics(
            "di",
            di_pickup_vector,
            di_source_rms,
            di_active,
        )
        mic_pickup_change = pickup_change_metrics(
            "mic",
            mic_pickup_vector,
            mic_source_rms,
            mic_active,
        )
        pickup_event_now = time.monotonic()
        di_pickup_event = pickup_switch_event(
            "di",
            di_pickup_vector,
            di_source_rms,
            pickup_event_now,
            di_active,
        )
        mic_pickup_event = pickup_switch_event(
            "mic",
            mic_pickup_vector,
            mic_source_rms,
            pickup_event_now,
            mic_active,
        )
        system_pickup_event = resolve_pickup_event(di_pickup_event, mic_pickup_event)
        system_pickup_status, di_pickup_status, mic_pickup_status = resolve_pickup_statuses(
            di_pickup_raw_status,
            mic_pickup_raw_status,
            di_source_rms,
            mic_source_rms,
            di_output_status,
            mic_output_status,
            str(di_pickup_change["label"]),
            str(mic_pickup_change["label"]),
            di_active,
            mic_active,
        )
        di_measured_status = measured_channel_status(di_pickup_change, di_output_status, di_active)
        mic_measured_status = measured_channel_status(mic_pickup_change, mic_output_status, mic_active)
        measured_pickup_status = measured_system_status(
            di_pickup_change,
            mic_pickup_change,
            di_output_status,
            mic_output_status,
            di_active,
            mic_active,
        )
        pickup_reference = classify_live_pickup_reference(
            live_pickup_reference_library,
            di_pickup_vector,
            mic_pickup_vector,
            di_active,
            mic_active,
            amp_weight=live_pickup_reference_amp_weight,
            min_margin=live_pickup_reference_margin,
            max_distance=live_pickup_reference_max_distance,
        )
        reference_pickup_status = str(pickup_reference.get("label", "no recording refs"))
        if bool(pickup_reference.get("reliable", False)):
            confidence_pct = int(round(float(pickup_reference.get("confidence", 0.0)) * 100.0))
            system_pickup_status = f"ref {reference_pickup_status} {confidence_pct}%"
            di_pickup_status = di_measured_status
            mic_pickup_status = mic_measured_status
        else:
            system_pickup_status = measured_pickup_status
            di_pickup_status = di_measured_status
            mic_pickup_status = mic_measured_status
        di_clip = clipping_percent(response[:, 0])
        mic_clip = clipping_percent(response[:, 1])
        if max(di_clip, mic_clip) > 0.0:
            capture_status = "clipping seen"
        elif di_status in {"good", "transient hot"} and mic_status in {"good", "transient hot"} and abs(delta_peak) <= balance_tolerance_db:
            capture_status = "model-ready"
        elif di_status in {"quiet", "too quiet", "silent"} or mic_status in {"quiet", "too quiet", "silent"}:
            capture_status = "raise quiet channel"
        elif di_status in {"hot", "clipping risk"} or mic_status in {"hot", "clipping risk"}:
            capture_status = "lower hot channel"
        else:
            capture_status = "usable, check balance"
        status_suffix = f"\nstatus {status_text}" if status_text else ""

        clamped_peaks = [float(np.clip(di_peak, level_floor_db, 0.0)), float(np.clip(mic_peak, level_floor_db, 0.0))]
        level_bars.setOpts(
            x=[(level_floor_db + clamped_peaks[0]) / 2.0, (level_floor_db + clamped_peaks[1]) / 2.0],
            y=[1.0, 0.0],
            width=[max(0.0, clamped_peaks[0] - level_floor_db), max(0.0, clamped_peaks[1] - level_floor_db)],
            height=0.30,
        )
        peak_points.setData(x=clamped_peaks, y=[1.0, 0.0])
        rms_points.setData(
            x=[float(np.clip(di_rms, level_floor_db, 0.0)), float(np.clip(mic_rms, level_floor_db, 0.0))],
            y=[1.0, 0.0],
        )
        p999_points.setData(
            x=[float(np.clip(di_p999, level_floor_db, 0.0)), float(np.clip(mic_p999, level_floor_db, 0.0))],
            y=[1.0, 0.0],
        )
        output_delta_text_items[0].setText(f"dOut {di_output_delta:+.1f} dB {di_output_status}")
        output_delta_text_items[1].setText(f"dOut {mic_output_delta:+.1f} dB {mic_output_status}")
        level_status_label.setText(
            f"Clean DI=blue | Amp/Mic=red | spectrum {'pickup-sensitive' if pickup_frequency_sensitivity else 'raw'} "
            f"x{pickup_view_boost:.1f} | dashed spectrum lines=rolloff | "
            "level markers: circle peak, triangle p99.9, diamond RMS    "
            f"DI peak {di_peak:.1f} / RMS {di_rms:.1f} / p99.9 {di_p999:.1f} / headroom {di_headroom:.1f} dB "
            f"({di_status}, dOut {di_output_delta:+.1f} {di_output_status})    "
            f"Mic peak {mic_peak:.1f} / RMS {mic_rms:.1f} / p99.9 {mic_p999:.1f} / headroom {mic_headroom:.1f} dB "
            f"({mic_status}, dOut {mic_output_delta:+.1f} {mic_output_status})    "
            f"Balance peak {delta_peak:+.1f} dB, RMS {delta_rms:+.1f} dB ({match_status})"
        )

        set_metric_label(
            0,
            "LEVELS\n"
            f"DI pk/rms/p99 {di_peak:5.1f}/{di_rms:5.1f}/{di_p999:5.1f}\n"
            f"MC pk/rms/p99 {mic_peak:5.1f}/{mic_rms:5.1f}/{mic_p999:5.1f}\n"
            f"dOut DI/MC    {di_output_delta:+4.1f}/{mic_output_delta:+4.1f} dB\n"
            f"HD DI/MC      {di_headroom:4.1f}/{mic_headroom:4.1f} dB\n"
            f"range {compact_text(level_settings['check_target'], 31)}"
            f"{status_suffix}",
            "LEVELS\n"
            f"DI peak  {di_peak:6.1f} dBFS\n"
            f"Mic peak {mic_peak:6.1f} dBFS\n"
            f"DI RMS   {di_rms:6.1f} dBFS\n"
            f"Mic RMS  {mic_rms:6.1f} dBFS\n"
            f"p99.9    {di_p999:5.1f}/{mic_p999:5.1f}\n"
            f"output delta {di_output_delta:+5.1f}/{mic_output_delta:+5.1f} dB "
            f"({di_output_status}/{mic_output_status})\n"
            f"headroom {di_headroom:5.1f}/{mic_headroom:5.1f}\n"
            f"range    {level_settings['check_target']}"
            f"{status_suffix}",
        )
        set_metric_label(
            1,
            "BALANCE\n"
            f"pk/rms DI-MC {delta_peak:+5.1f}/{delta_rms:+5.1f} dB\n"
            f"clip DI/MC   {di_clip:5.2f}/{mic_clip:5.2f}%\n"
            f"HD DI/MC     {di_headroom:4.1f}/{mic_headroom:4.1f} dB\n"
            f"target +/-{balance_tolerance_db:.0f} {compact_text(match_status, 18)}",
            "BALANCE\n"
            f"peak DI-mic {delta_peak:+6.1f} dB\n"
            f"RMS  DI-mic {delta_rms:+6.1f} dB\n"
            f"DI clip      {di_clip:6.3f}%\n"
            f"Mic clip     {mic_clip:6.3f}%\n"
            f"headroom     {di_headroom:4.1f}/{mic_headroom:4.1f}\n"
            f"target       +/-{balance_tolerance_db:.0f} dB\n"
            f"status       {match_status}",
        )
        set_metric_label(
            2,
            "DIFF AMP-DI\n"
            f"L/B/M   {diff_low:+4.1f}/{diff_body:+4.1f}/{diff_mid:+4.1f} dB\n"
            f"U/Bt/A  {diff_upper:+4.1f}/{diff_bite:+4.1f}/{diff_air:+4.1f} dB\n"
            f"dom {compact_text(dominant_band, 12)} {dominant_band_db:+4.1f} dB",
            "TONE DIFF\n"
            f"low   {diff_low:+6.1f} dB\n"
            f"body  {diff_body:+6.1f} dB\n"
            f"mid   {diff_mid:+6.1f} dB\n"
            f"upper {diff_upper:+6.1f} dB\n"
            f"bite  {diff_bite:+6.1f} dB\n"
            f"air   {diff_air:+6.1f} dB\n"
            f"dom   {dominant_band} {dominant_band_db:+.1f}",
        )
        set_metric_label(
            3,
            "SPECTRUM\n"
            f"C/R DI {di_centroid:5.0f}/{di_rolloff:5.0f} Hz\n"
            f"C/R MC {mic_centroid:5.0f}/{mic_rolloff:5.0f} Hz\n"
            f"Pk Hz  {di_peak_freq:5.0f}/{mic_peak_freq:5.0f}\n"
            f"L/B/M D {di_low_pct:3.0f}/{di_body_pct:3.0f}/{di_mid_pct:3.0f}%\n"
            f"L/B/M M {mic_low_pct:3.0f}/{mic_body_pct:3.0f}/{mic_mid_pct:3.0f}%\n"
            f"Bt/A D/M {di_bite_pct:3.0f}/{di_air_pct:3.0f} {mic_bite_pct:3.0f}/{mic_air_pct:3.0f}%",
            "SPECTRUM\n"
            f"DI cent/roll {di_centroid:5.0f}/{di_rolloff:5.0f}\n"
            f"Mic cent/roll{mic_centroid:5.0f}/{mic_rolloff:5.0f}\n"
            f"DI peak freq {di_peak_freq:7.0f} Hz\n"
            f"Mic peak     {mic_peak_freq:7.0f} Hz\n"
            f"low% DI/Mic  {di_low_pct:4.1f}/{mic_low_pct:4.1f}\n"
            f"body%        {di_body_pct:4.1f}/{mic_body_pct:4.1f}\n"
            f"mid%         {di_mid_pct:4.1f}/{mic_mid_pct:4.1f}\n"
            f"bite%        {di_bite_pct:4.1f}/{mic_bite_pct:4.1f}\n"
            f"air%         {di_air_pct:4.1f}/{mic_air_pct:4.1f}",
        )
        set_metric_label(
            4,
            "WAVEFORM\n"
            f"crest DI/MC {di_crest:4.1f}/{mic_crest:4.1f}x\n"
            f"trans/s     {di_transients:4.1f}/{mic_transients:4.1f}\n"
            f"ZCR/s       {di_zcr:4.0f}/{mic_zcr:4.0f}\n"
            f"slew/s      {di_slew:4.0f}/{mic_slew:4.0f}\n"
            f"p2p         {np.ptp(response[:, 0]):4.2f}/{np.ptp(response[:, 1]):4.2f}\n"
            f"pOv dB      {di_peak_overshoot:4.1f}/{mic_peak_overshoot:4.1f}",
            "WAVEFORM\n"
            f"crest DI/Mic {di_crest:5.1f}/{mic_crest:5.1f}x\n"
            f"trans DI/Mic {di_transients:5.1f}/{mic_transients:5.1f}/s\n"
            f"ZCR DI/Mic   {di_zcr:5.0f}/{mic_zcr:5.0f}/s\n"
            f"slew DI/Mic  {di_slew:5.0f}/{mic_slew:5.0f}/s\n"
            f"p2p DI/Mic   {np.ptp(response[:, 0]):5.3f}/{np.ptp(response[:, 1]):5.3f}\n"
            f"peak-p99.9   {di_peak_overshoot:4.1f}/{mic_peak_overshoot:4.1f} dB",
        )
        set_metric_label(
            5,
            "PICKUP/BLOWER\n"
            f"EVT {compact_text(system_pickup_event, 24)}\n"
            f"SYS {compact_text(system_pickup_status, 24)}\n"
            f"DI  {compact_text(di_pickup_status, 24)}\n"
            f"MC  {compact_text(mic_pickup_status, 24)}\n"
            f"dOut {di_output_delta:+4.1f}/{mic_output_delta:+4.1f} "
            f"{str(di_activity['label'])[:1].upper()}/{str(mic_activity['label'])[:1].upper()}",
            "PICKUP / BLOWER\n"
            f"Event        {system_pickup_event}\n"
            f"DI event     {di_pickup_event['label']} score {float(di_pickup_event['score']):.2f}\n"
            f"Mic event    {mic_pickup_event['label']} score {float(mic_pickup_event['score']):.2f}\n"
            f"DI activity  {di_activity['label']} score {float(di_activity['score']):.2f} "
            f"rms margin {float(di_activity['rms_margin']):+.1f} p99 margin {float(di_activity['p999_margin']):+.1f}\n"
            f"Mic activity {mic_activity['label']} score {float(mic_activity['score']):.2f} "
            f"rms margin {float(mic_activity['rms_margin']):+.1f} p99 margin {float(mic_activity['p999_margin']):+.1f}\n"
            f"Reference    {reference_pickup_status} reliable={bool(pickup_reference.get('reliable', False))} "
            f"method={pickup_reference.get('method', 'none')} "
            f"conf={float(pickup_reference.get('confidence', 0.0)):.2f} "
            f"dist={float(pickup_reference.get('distance', 0.0)):.2f} "
            f"margin={float(pickup_reference.get('margin', 0.0)):.2f}\n"
            f"System       {system_pickup_status}\n"
            f"DI raw       {di_pickup_raw_status}\n"
            f"Mic raw      {mic_pickup_raw_status}\n"
            f"DI tone      {di_tone_status}\n"
            f"Mic tone     {mic_tone_status}\n"
            f"rolloff Hz   {di_rolloff:5.0f}/{mic_rolloff:5.0f}\n"
            f"res peak Hz  {di_resonant_peak:5.0f}/{mic_resonant_peak:5.0f}\n"
            f"DI voicing   {di_pickup_status}\n"
            f"Mic voicing  {mic_pickup_status}\n"
            f"DI change    {di_pickup_change['label']}\n"
            f"Mic change   {mic_pickup_change['label']}\n"
            f"fast dOut    {di_output_delta:+4.1f}/{mic_output_delta:+4.1f} dB "
            f"({di_output_status}/{mic_output_status})\n"
            f"dUpper%      {float(di_pickup_change['upper_delta']):+4.1f}/{float(mic_pickup_change['upper_delta']):+4.1f}\n"
            f"dHigh%       {float(di_pickup_change['high_delta']):+4.1f}/{float(mic_pickup_change['high_delta']):+4.1f}\n"
            f"dRoll Hz     {float(di_pickup_change['rolloff_delta']):+5.0f}/{float(mic_pickup_change['rolloff_delta']):+5.0f}\n"
            f"dRes Hz      {float(di_pickup_change['resonant_delta']):+5.0f}/{float(mic_pickup_change['resonant_delta']):+5.0f}\n"
            f"dLevel dB    {float(di_pickup_change['level_delta']):+4.1f}/{float(mic_pickup_change['level_delta']):+4.1f}",
        )
        set_metric_label(
            6,
            "MLX FEATURES\n"
            f"cent/roll {centroid_gap:+5.0f}/{rolloff_gap:+5.0f} Hz\n"
            f"high gap  {high_gap:+5.1f}%\n"
            f"body/bite {di_body_to_bite:4.1f}/{mic_body_to_bite:4.1f}\n"
            f"band peak {di_band_peak:5.0f}/{mic_band_peak:5.0f}\n"
            f"analysis  {source_analysis_ms:5.0f} ms\n"
            f"score     {float(di_pickup_change['score']):4.2f}/{float(mic_pickup_change['score']):4.2f}",
            "MLX FEATURE VIEW\n"
            f"cent gap     {centroid_gap:+7.0f} Hz\n"
            f"roll gap     {rolloff_gap:+7.0f} Hz\n"
            f"high gap     {high_gap:+7.1f}%\n"
            f"body/bite    {di_body_to_bite:4.1f}/{mic_body_to_bite:4.1f}\n"
            f"band peak    {di_band_peak:5.0f}/{mic_band_peak:5.0f}\n"
            f"analysis     {source_analysis_ms:5.0f} ms\n"
            f"change score {float(di_pickup_change['score']):4.2f}/{float(mic_pickup_change['score']):4.2f}",
        )
        set_metric_label(
            7,
            "QUALITY\n"
            f"{compact_text(capture_status, 30)}\n"
            f"corr/dly {short_corr:+5.3f}/{delay_ms:+5.2f} ms\n"
            f"dyn dB   {di_dynamic_range:4.1f}/{mic_dynamic_range:4.1f}\n"
            f"floor    {di_noise_floor:5.1f}/{mic_noise_floor:5.1f}\n"
            f"balance  {compact_text(match_status, 22)}",
            "CAPTURE QUALITY\n"
            f"status       {capture_status}\n"
            f"short corr   {short_corr:+6.3f}\n"
            f"delay        {delay_ms:+6.2f} ms\n"
            f"dyn range    {di_dynamic_range:4.1f}/{mic_dynamic_range:4.1f} dB\n"
            f"floor p50    {di_noise_floor:5.1f}/{mic_noise_floor:5.1f}\n"
            f"balance      {match_status}",
        )

        now = time.monotonic()
        if feature_log_handle is not None and now - last_feature_log_time >= feature_log_interval_s:
            feature_log_frame += 1
            last_feature_log_time = now
            di_feature_vector = [
                float(di_centroid / max(1.0, sample_rate / 2.0)),
                float(di_rolloff / max(1.0, sample_rate / 2.0)),
                float(di_low_pct / 100.0),
                float(di_body_pct / 100.0),
                float(di_mid_pct / 100.0),
                float(di_upper_pct / 100.0),
                float(di_bite_pct / 100.0),
                float(di_air_pct / 100.0),
                float(di_peak),
                float(di_rms),
                float(di_p999),
                float(di_crest),
                float(di_transients),
                float(di_zcr),
                float(di_slew),
                float(di_noise_floor),
                float(di_dynamic_range),
                float(di_peak_overshoot),
            ]
            mic_feature_vector = [
                float(mic_centroid / max(1.0, sample_rate / 2.0)),
                float(mic_rolloff / max(1.0, sample_rate / 2.0)),
                float(mic_low_pct / 100.0),
                float(mic_body_pct / 100.0),
                float(mic_mid_pct / 100.0),
                float(mic_upper_pct / 100.0),
                float(mic_bite_pct / 100.0),
                float(mic_air_pct / 100.0),
                float(mic_peak),
                float(mic_rms),
                float(mic_p999),
                float(mic_crest),
                float(mic_transients),
                float(mic_zcr),
                float(mic_slew),
                float(mic_noise_floor),
                float(mic_dynamic_range),
                float(mic_peak_overshoot),
            ]
            feature_log_handle.write(
                json.dumps(
                    {
                        "type": "live_scope_feature_frame",
                        "session_id": feature_log_session,
                        "frame": int(feature_log_frame),
                        "timestamp_monotonic_s": float(now),
                        "source": {
                            "di_channel": int(interface.di_channel),
                            "target_channel": int(interface.target_channel),
                            "sample_rate_hz": int(sample_rate),
                            "analysis_window_ms": float(source_analysis_ms),
                        },
                        "feature_names": [
                            "centroid_ratio",
                            "rolloff_ratio",
                            "low_pct",
                            "body_pct",
                            "mid_pct",
                            "upper_pct",
                            "bite_pct",
                            "air_pct",
                            "peak_dbfs",
                            "rms_dbfs",
                            "p999_dbfs",
                            "crest_factor",
                            "transients_per_s",
                            "zero_crossings_per_s",
                            "slew_per_s",
                            "noise_floor_p50_dbfs",
                            "dynamic_range_p999_p50_db",
                            "peak_over_p999_db",
                        ],
                        "di_feature_vector": di_feature_vector,
                        "mic_feature_vector": mic_feature_vector,
                        "tone_diff_db": {
                            "low": float(diff_low),
                            "body": float(diff_body),
                            "mid": float(diff_mid),
                            "upper": float(diff_upper),
                            "bite": float(diff_bite),
                            "air": float(diff_air),
                            "dominant_band": str(dominant_band),
                            "dominant_band_db": float(dominant_band_db),
                        },
                        "balance": {
                            "peak_delta_db": float(delta_peak),
                            "rms_delta_db": float(delta_rms),
                            "status": str(match_status),
                            "capture_status": str(capture_status),
                            "short_correlation": float(short_corr),
                            "delay_ms": float(delay_ms),
                        },
                        "output_level_change": {
                            "di_output_level_dbfs": float(di_output_level),
                            "mic_output_level_dbfs": float(mic_output_level),
                            "di_source_rms_dbfs": float(di_source_rms),
                            "mic_source_rms_dbfs": float(mic_source_rms),
                            "di_output_delta_db": float(di_output_delta),
                            "mic_output_delta_db": float(mic_output_delta),
                            "di_output_status": str(di_output_status),
                            "mic_output_status": str(mic_output_status),
                            "di_activity": str(di_activity["label"]),
                            "mic_activity": str(mic_activity["label"]),
                            "di_activity_score": float(di_activity["score"]),
                            "mic_activity_score": float(mic_activity["score"]),
                            "di_activity_rms_margin_db": float(di_activity["rms_margin"]),
                            "mic_activity_rms_margin_db": float(mic_activity["rms_margin"]),
                            "di_activity_p999_margin_db": float(di_activity["p999_margin"]),
                            "mic_activity_p999_margin_db": float(mic_activity["p999_margin"]),
                            "di_activity_peak_margin_db": float(di_activity["peak_margin"]),
                            "mic_activity_peak_margin_db": float(mic_activity["peak_margin"]),
                        },
                        "rolloff_pickup": {
                            "system_event": str(system_pickup_event),
                            "di_event": str(di_pickup_event["label"]),
                            "mic_event": str(mic_pickup_event["label"]),
                            "di_event_score": float(di_pickup_event["score"]),
                            "mic_event_score": float(mic_pickup_event["score"]),
                            "di_event_level_delta_db": float(di_pickup_event["level_delta"]),
                            "mic_event_level_delta_db": float(mic_pickup_event["level_delta"]),
                            "di_event_high_delta_pct": float(di_pickup_event["high_delta"]),
                            "mic_event_high_delta_pct": float(mic_pickup_event["high_delta"]),
                            "system_voicing": str(system_pickup_status),
                            "reference_label": str(reference_pickup_status),
                            "reference_reliable": bool(pickup_reference.get("reliable", False)),
                            "reference_confidence": float(pickup_reference.get("confidence", 0.0)),
                            "reference_distance": float(pickup_reference.get("distance", 0.0)),
                            "reference_second_distance": float(pickup_reference.get("second_distance", 0.0)),
                            "reference_margin": float(pickup_reference.get("margin", 0.0)),
                            "reference_method": str(pickup_reference.get("method", "")),
                            "reference_nearest_label": str(pickup_reference.get("nearest_label", "")),
                            "reference_classifier_label": str(pickup_reference.get("classifier_label", "")),
                            "reference_classifier_confidence": float(
                                pickup_reference.get("classifier_confidence", 0.0)
                            ),
                            "measured_system": str(measured_pickup_status),
                            "di_measured": str(di_measured_status),
                            "mic_measured": str(mic_measured_status),
                            "di_raw_voicing": str(di_pickup_raw_status),
                            "mic_raw_voicing": str(mic_pickup_raw_status),
                            "di_tone": str(di_tone_status),
                            "mic_tone": str(mic_tone_status),
                            "di_voicing": str(di_pickup_status),
                            "mic_voicing": str(mic_pickup_status),
                            "di_change": str(di_pickup_change["label"]),
                            "mic_change": str(mic_pickup_change["label"]),
                            "di_change_score": float(di_pickup_change["score"]),
                            "mic_change_score": float(mic_pickup_change["score"]),
                            "di_upper_delta_pct": float(di_pickup_change["upper_delta"]),
                            "mic_upper_delta_pct": float(mic_pickup_change["upper_delta"]),
                            "di_high_delta_pct": float(di_pickup_change["high_delta"]),
                            "mic_high_delta_pct": float(mic_pickup_change["high_delta"]),
                            "di_rolloff_delta_hz": float(di_pickup_change["rolloff_delta"]),
                            "mic_rolloff_delta_hz": float(mic_pickup_change["rolloff_delta"]),
                            "di_resonant_peak_hz": float(di_resonant_peak),
                            "mic_resonant_peak_hz": float(mic_resonant_peak),
                            "di_resonant_delta_hz": float(di_pickup_change["resonant_delta"]),
                            "mic_resonant_delta_hz": float(mic_pickup_change["resonant_delta"]),
                            "di_level_delta_db": float(di_pickup_change["level_delta"]),
                            "mic_level_delta_db": float(mic_pickup_change["level_delta"]),
                            "di_body_to_bite_delta": float(di_pickup_change["body_to_bite_delta"]),
                            "mic_body_to_bite_delta": float(mic_pickup_change["body_to_bite_delta"]),
                        },
                    }
                )
                + "\n"
            )
            feature_log_handle.flush()

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=interface.input_channels,
        dtype="float32",
        device=interface.device,
        blocksize=block_samples,
        callback=callback,
    )
    timer = QtCore.QTimer()
    timer.timeout.connect(update)

    device_label = "system default input" if interface.device is None else interface.device
    print("Opening PyQtGraph live scope. Close the window or press Ctrl+C to stop.")
    print(f"Device: {device_label} | sample_rate={sample_rate} | channels={interface.input_channels}")
    print(
        f"Qt scope: block_ms={block_ms:g} refresh_ms={refresh_ms:g} "
        f"window_ms={window_ms:g} fft_size={fft_size} opengl={args.opengl}"
    )
    print(f"Clean DI channel={interface.di_channel} | Amp/Mic channel={interface.target_channel}")
    if live_pickup_reference_enabled:
        labels = dict(live_pickup_reference_library.get("label_counts", {}))
        if labels:
            label_text = ", ".join(f"{label}:{count}" for label, count in sorted(labels.items()))
            print(f"Live pickup recording references: {label_text}")
        else:
            print(f"Live pickup recording references: none found in {live_pickup_reference_dir}")
        sklearn_state = dict(live_pickup_reference_library.get("sklearn", {}))
        sklearn_routes = sorted(dict(sklearn_state.get("models", {})))
        if sklearn_routes:
            print(f"Live pickup sklearn classifier routes: {', '.join(sklearn_routes)}")
        else:
            print(f"Live pickup sklearn classifier: {sklearn_state.get('reason', 'not available')}")
    if feature_log_handle is not None:
        print(f"Writing MLX-readable live feature log: {args.feature_log}")

    def stop_stream() -> None:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    app.aboutToQuit.connect(stop_stream)
    try:
        stream.start()
        timer.start(refresh_ms)
        screen = app.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            window_width = min(int(args.width), max(960, int(available.width() * 0.88)))
            window_height = min(int(args.height), max(720, int(available.height() * 0.84)))
            main.resize(window_width, window_height)
            set_scope_splitter_sizes(window_height)
            main.move(
                available.x() + max(0, (available.width() - window_width) // 2),
                available.y() + max(0, (available.height() - window_height) // 2),
            )
        else:
            window_height = int(args.height)
            main.resize(int(args.width), window_height)
            set_scope_splitter_sizes(window_height)
        main.setWindowState(QtCore.Qt.WindowState.WindowNoState)
        main.showNormal()
        app.exec()
    finally:
        timer.stop()
        stop_stream()
        if feature_log_handle is not None:
            feature_log_handle.close()


def run_record_command(args: argparse.Namespace) -> None:
    interface = build_audio_interface_config(args)
    di_box = build_di_box_config(args)
    take_metadata = build_take_metadata(args)
    recording = record_interface_take(
        take_name=args.take_name,
        output_dir=args.output_dir,
        interface=interface,
        di_box=di_box,
        take_metadata=take_metadata,
        level_profile=args.level_profile,
    )
    if args.dataset:
        append_recording_to_dataset(args.dataset, recording)


def run_record_capture_command(args: argparse.Namespace) -> None:
    interface = build_audio_interface_config(args)
    di_box = build_di_box_config(args)
    take_metadata = build_take_metadata(args)
    recording = record_interface_take(
        take_name=args.take_name,
        output_dir=args.output_dir,
        interface=interface,
        di_box=di_box,
        take_metadata=take_metadata,
        level_profile=args.level_profile,
    )

    config = CaptureConfig(
        instrument=args.instrument,
        profile_name=args.name,
        ir_ms=args.ir_ms,
        regularization=args.regularization,
        search_bias=not args.no_bias_search,
    )
    result = capture_tone_profile(
        recording["di_audio"],
        recording["target_audio"],
        sample_rate=interface.sample_rate,
        config=config,
        di_source_name=str(recording["di_path"]),
        target_source_name=str(recording["target_path"]),
        hardware_context=recording["hardware_context"],
    )

    profile_path = args.profile or (args.output_dir / f"{args.take_name}_tone_profile.json")
    reconstructed_path = args.reconstructed or (args.output_dir / f"{args.take_name}_captured_match.wav")
    save_profile(profile_path, result.profile)
    write_wav_float(reconstructed_path, interface.sample_rate, normalize_for_audition(result.reconstructed))
    if args.dataset:
        append_recording_to_dataset(
            args.dataset,
            recording,
            profile_path=profile_path,
            reconstructed_path=reconstructed_path,
        )

    print(f"Wrote profile: {profile_path}")
    print(f"Wrote reconstructed match: {reconstructed_path}")
    print(f"Match correlation: {result.match_correlation:.3f}")
    print(f"Spectral error: {result.spectral_error_db:.2f} dB")


def run_capture_command(args: argparse.Namespace) -> None:
    di_rate, di_audio = read_wav_float(args.di)
    target_rate, target_audio = read_wav_float(args.target)
    target_audio = resample_if_needed(target_audio, target_rate, di_rate)
    hardware_context = load_hardware_context(args.manifest)

    config = CaptureConfig(
        instrument=args.instrument,
        profile_name=args.name,
        ir_ms=args.ir_ms,
        regularization=args.regularization,
        search_bias=not args.no_bias_search,
    )
    result = capture_tone_profile(
        di_audio,
        target_audio,
        sample_rate=di_rate,
        config=config,
        di_source_name=str(args.di),
        target_source_name=str(args.target),
        hardware_context=hardware_context,
    )
    save_profile(args.profile, result.profile)

    if args.reconstructed:
        write_wav_float(args.reconstructed, di_rate, normalize_for_audition(result.reconstructed))

    print(f"Wrote profile: {args.profile}")
    print(f"Match correlation: {result.match_correlation:.3f}")
    print(f"Spectral error: {result.spectral_error_db:.2f} dB")


def run_apply_command(args: argparse.Namespace) -> None:
    sample_rate, audio = read_wav_float(args.input)
    profile = load_profile(args.profile)
    output = apply_profile_to_audio(audio, sample_rate, profile)
    write_wav_float(args.output, int(profile["sample_rate_hz"]), output)
    print(f"Wrote profiled audio: {args.output}")


def run_tone_match_command(args: argparse.Namespace) -> None:
    di_rate, di_audio = read_wav_float(args.di)
    target_rate, target_audio = read_wav_float(args.target)
    target_audio = resample_if_needed(target_audio, target_rate, di_rate)
    profile = load_profile(args.profile) if args.profile else None

    matched, metrics = render_spectral_tone_match(
        di_audio=di_audio,
        target_audio=target_audio,
        sample_rate=di_rate,
        profile=profile,
        fft_size=args.fft_size,
        smoothing_bins=args.smoothing_bins,
        amp_style=args.amp_style,
        drive_boost=args.drive_boost,
    )
    write_wav_float(args.output, di_rate, matched)

    if args.comparison_output:
        target_audition = normalize_for_audition(target_audio[: len(matched)], peak=0.70)
        matched_audition = normalize_for_audition(matched, peak=0.70)
        silence = np.zeros(di_rate, dtype=np.float64)
        comparison = np.concatenate([target_audition, silence, matched_audition])
        write_wav_float(args.comparison_output, di_rate, comparison)

    print(f"Wrote spectral tone-match audition: {args.output}")
    if args.comparison_output:
        print(f"Wrote mic-then-match comparison: {args.comparison_output}")
    print(f"Match correlation: {metrics['match_correlation']:.3f}")
    print(f"Spectral error: {metrics['spectral_error_db']:.2f} dB")
    print(f"Audition peak: {metrics['audition_peak_dbfs']:.1f} dBFS")


def run_mic_learn_command(args: argparse.Namespace) -> None:
    di_rate, di_audio = read_wav_float(args.di)
    target_rate, target_audio = read_wav_float(args.target)
    target_audio = resample_if_needed(target_audio, target_rate, di_rate)

    model, reconstructed, metrics = estimate_parallel_hammerstein_model(
        di_audio=di_audio,
        target_audio=target_audio,
        sample_rate=di_rate,
        ir_ms=args.ir_ms,
        regularization=args.regularization,
    )
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2), encoding="utf-8")
    write_wav_float(args.output, di_rate, reconstructed)

    if args.comparison_output:
        target_audition = normalize_for_audition(target_audio[: len(reconstructed)], peak=0.70)
        reconstructed_audition = normalize_for_audition(reconstructed, peak=0.70)
        silence = np.zeros(di_rate, dtype=np.float64)
        comparison = np.concatenate([target_audition, silence, reconstructed_audition])
        write_wav_float(args.comparison_output, di_rate, comparison)

    print(f"Wrote mic-learned model: {args.model}")
    print(f"Wrote mic-learned render: {args.output}")
    if args.comparison_output:
        print(f"Wrote mic-then-learned comparison: {args.comparison_output}")
    print(f"Match correlation: {metrics['match_correlation']:.3f}")
    print(f"Spectral error: {metrics['spectral_error_db']:.2f} dB")
    print(f"Audition peak: {metrics['audition_peak_dbfs']:.1f} dBFS")


def run_apply_mic_model_command(args: argparse.Namespace) -> None:
    sample_rate, audio = read_wav_float(args.input)
    model = json.loads(args.model.read_text(encoding="utf-8"))
    output_rate, output = apply_parallel_hammerstein_model(audio, sample_rate, model)
    write_wav_float(args.output, output_rate, output)
    print(f"Wrote mic-learned model output: {args.output}")


def run_train_mlx_bridge_command(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.context_radius < 1:
        raise SystemExit("--context-radius must be at least 1.")
    if args.max_train_samples < 32:
        raise SystemExit("--max-train-samples must be at least 32.")
    if args.chunk_samples < 1024:
        raise SystemExit("--chunk-samples must be at least 1024.")

    mx = require_mlx()
    rng = np.random.default_rng(args.seed)

    di_rate, di_audio = read_wav_float(args.di)
    target_rate, target_audio = read_wav_float(args.target)
    target_audio = resample_if_needed(target_audio, target_rate, di_rate)

    base_model, base_render, base_metrics = estimate_parallel_hammerstein_model(
        di_audio=di_audio,
        target_audio=target_audio,
        sample_rate=di_rate,
        ir_ms=args.ir_ms,
        regularization=args.regularization,
    )
    args.base_model.parent.mkdir(parents=True, exist_ok=True)
    args.base_model.write_text(json.dumps(base_model, indent=2), encoding="utf-8")
    if args.base_output:
        write_wav_float(args.base_output, di_rate, base_render)

    di_norm = normalize_peak(remove_dc(di_audio), peak=0.95)
    target_norm = normalize_peak(remove_dc(target_audio), peak=0.95)
    di_aligned, target_aligned, lag, polarity = align_pair(
        di_norm,
        target_norm,
        max_lag_s=0.05,
        sample_rate=di_rate,
    )
    _, base_audio = apply_parallel_hammerstein_model(di_aligned, di_rate, base_model)
    min_len = min(len(di_aligned), len(target_aligned), len(base_audio))
    di_aligned = di_aligned[:min_len]
    target_aligned = target_aligned[:min_len]
    base_audio = base_audio[:min_len]

    base_gain = estimate_gain(base_audio, target_aligned)
    base_audio = base_audio * base_gain
    baseline_rmse = rms(target_aligned - base_audio)
    baseline_corr = correlation(target_aligned, base_audio)
    baseline_spec = spectral_error_db(target_aligned, base_audio, di_rate)
    residual = target_aligned - base_audio
    residual_scale = max(float(np.percentile(np.abs(residual), 99.5)), 1e-5)

    usable_end = min_len - args.context_radius
    if args.max_training_seconds > 0:
        usable_end = min(usable_end, int(round(args.max_training_seconds * di_rate)))

    candidate_indices = np.arange(args.context_radius, max(args.context_radius + 1, usable_end), dtype=np.int64)
    if len(candidate_indices) > args.max_train_samples:
        candidate_indices = rng.choice(candidate_indices, size=args.max_train_samples, replace=False)
    rng.shuffle(candidate_indices)
    if len(candidate_indices) < 32:
        raise SystemExit("Not enough usable samples for MLX bridge training.")

    raw_features = build_context_features(
        di_aligned,
        base_audio,
        candidate_indices,
        context_radius=args.context_radius,
    )
    feature_mean = raw_features.mean(axis=0).astype(np.float32)
    feature_std = raw_features.std(axis=0).astype(np.float32)
    features = ((raw_features - feature_mean) / np.maximum(feature_std, 1e-6)).astype(np.float32)
    targets = (residual[candidate_indices] / residual_scale).reshape(-1, 1).astype(np.float32)

    validation_count = int(round(len(features) * args.validation_fraction))
    validation_count = max(0, min(validation_count, len(features) // 3))
    if validation_count:
        x_val = features[:validation_count]
        y_val = targets[:validation_count]
        x_train = features[validation_count:]
        y_train = targets[validation_count:]
    else:
        x_val = np.empty((0, features.shape[1]), dtype=np.float32)
        y_val = np.empty((0, 1), dtype=np.float32)
        x_train = features
        y_train = targets

    params = init_mlx_mlp_params(
        mx=mx,
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    first_moment = mlx_tree_zeros_like(mx, params)
    second_moment = mlx_tree_zeros_like(mx, params)

    def loss_fn(candidate_params, x_batch, y_batch):
        return mlx_mse_loss(mx, candidate_params, x_batch, y_batch)

    value_and_grad = mx.value_and_grad(loss_fn)
    step = 0
    history = []

    print("Training MLX bridge residual...")
    print(f"Base mic-learn bridge: corr={base_metrics['match_correlation']:.3f} spec={base_metrics['spectral_error_db']:.2f} dB")
    print(f"Residual baseline: corr={baseline_corr:.3f} rmse={baseline_rmse:.5f} spec={baseline_spec:.2f} dB")
    print(f"Samples: train={len(x_train)} validation={len(x_val)} features={features.shape[1]}")

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        epoch_loss = 0.0

        for batch_start in range(0, len(order), args.batch_size):
            batch_rows = order[batch_start : batch_start + args.batch_size]
            x_batch = mx.array(x_train[batch_rows])
            y_batch = mx.array(y_train[batch_rows])
            loss, grads = value_and_grad(params, x_batch, y_batch)
            step += 1
            params, first_moment, second_moment = mlx_adam_update(
                mx=mx,
                params=params,
                grads=grads,
                first_moment=first_moment,
                second_moment=second_moment,
                step=step,
                learning_rate=args.learning_rate,
            )
            mx.eval(loss, *params.values(), *first_moment.values(), *second_moment.values())
            epoch_loss += float(loss.item()) * len(batch_rows)

        train_loss = epoch_loss / max(1, len(x_train))
        if len(x_val):
            val_loss = loss_fn(params, mx.array(x_val), mx.array(y_val))
            mx.eval(val_loss)
            val_loss_float = float(val_loss.item())
        else:
            val_loss_float = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss_float})
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} validation_loss={val_loss_float:.6f}")

    mx.eval(*params.values())
    params_np = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    metadata = {
        "model_version": MLX_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "MLX residual bridge on top of a mic-learned DI-to-SM57 DSP model.",
        "bridge_mode": "mic_learned_parallel_hammerstein_plus_mlx_residual",
        "base_model": str(args.base_model),
        "sample_rate_hz": int(di_rate),
        "alignment_lag_samples": int(lag),
        "target_polarity_after_alignment": int(polarity),
        "context_radius": int(args.context_radius),
        "input_feature_count": int(features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "residual_scale": float(residual_scale),
        "residual_mix": float(args.residual_mix),
        "base_gain": float(base_gain),
        "feature_mean": [float(value) for value in feature_mean],
        "feature_std": [float(max(value, 1e-6)) for value in feature_std],
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "max_train_samples": int(args.max_train_samples),
            "max_training_seconds": float(args.max_training_seconds),
            "validation_fraction": float(args.validation_fraction),
            "history": history,
        },
        "baseline_validation": {
            "rmse": float(baseline_rmse),
            "correlation": float(baseline_corr),
            "spectral_error_db": float(baseline_spec),
        },
    }

    predicted_residual = predict_mlx_residual(
        di_audio=di_aligned,
        base_audio=base_audio,
        metadata=metadata,
        params=params_np,
        chunk_samples=args.chunk_samples,
    )
    enhanced = normalize_for_audition(soft_limiter(base_audio + predicted_residual), peak=0.86)
    enhanced_rmse = rms(target_aligned - enhanced)
    enhanced_corr = correlation(target_aligned, enhanced)
    enhanced_spec = spectral_error_db(target_aligned, enhanced, di_rate)
    metadata["enhanced_validation"] = {
        "rmse": float(enhanced_rmse),
        "correlation": float(enhanced_corr),
        "spectral_error_db": float(enhanced_spec),
    }

    save_mlx_residual_model(args.model, metadata, params_np)
    write_wav_float(args.output, di_rate, enhanced)

    if args.comparison_output:
        target_audition = normalize_for_audition(target_aligned[: len(enhanced)], peak=0.70)
        enhanced_audition = normalize_for_audition(enhanced, peak=0.70)
        silence = np.zeros(di_rate, dtype=np.float64)
        comparison = np.concatenate([target_audition, silence, enhanced_audition])
        write_wav_float(args.comparison_output, di_rate, comparison)

    print(f"Wrote mic-learned base model: {args.base_model}")
    print(f"Wrote MLX bridge model: {args.model}")
    print(f"Wrote MLX bridge render: {args.output}")
    if args.comparison_output:
        print(f"Wrote mic-then-MLX comparison: {args.comparison_output}")
    print(f"Enhanced bridge: corr={enhanced_corr:.3f} rmse={enhanced_rmse:.5f} spec={enhanced_spec:.2f} dB")


def run_apply_mlx_bridge_command(args: argparse.Namespace) -> None:
    sample_rate, audio = read_wav_float(args.input)
    base_model = json.loads(args.base_model.read_text(encoding="utf-8"))
    metadata, params = load_mlx_residual_model(args.model)
    model_rate = int(metadata["sample_rate_hz"])
    audio = resample_if_needed(audio, sample_rate, model_rate)
    audio = normalize_peak(remove_dc(audio), peak=0.95)
    _, base_audio = apply_parallel_hammerstein_model(audio, model_rate, base_model)
    residual = predict_mlx_residual(
        di_audio=audio,
        base_audio=base_audio,
        metadata=metadata,
        params=params,
        chunk_samples=args.chunk_samples,
    )
    output = normalize_for_audition(soft_limiter(base_audio + residual), peak=0.86)
    write_wav_float(args.output, model_rate, output)
    print(f"Wrote MLX bridge output: {args.output}")


def run_train_mlx_spectrum_command(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.fft_size < 256:
        raise SystemExit("--fft-size must be at least 256.")
    if args.hop_size < 1 or args.hop_size >= args.fft_size:
        raise SystemExit("--hop-size must be between 1 and fft-size - 1.")
    if args.batch_frames < 1:
        raise SystemExit("--batch-frames must be at least 1.")

    mx = require_mlx()
    rng = np.random.default_rng(args.seed)

    di_rate, di_audio = read_wav_float(args.di)
    target_rate, target_audio = read_wav_float(args.target)
    target_audio = resample_if_needed(target_audio, target_rate, di_rate)
    di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
    target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
    di_aligned, target_aligned, lag, polarity = align_pair(
        di_audio,
        target_audio,
        max_lag_s=0.05,
        sample_rate=di_rate,
    )
    min_len = min(len(di_aligned), len(target_aligned))
    di_aligned = di_aligned[:min_len]
    target_aligned = target_aligned[:min_len]

    di_frames, window = stft_audio(di_aligned, fft_size=args.fft_size, hop_size=args.hop_size)
    target_frames, _ = stft_audio(target_aligned, fft_size=args.fft_size, hop_size=args.hop_size)
    frame_count = min(len(di_frames), len(target_frames))
    di_frames = di_frames[:frame_count]
    target_frames = target_frames[:frame_count]

    di_db = 20.0 * np.log10(np.abs(di_frames).astype(np.float32) + 1e-7)
    target_db = 20.0 * np.log10(np.abs(target_frames).astype(np.float32) + 1e-7)
    gain_db = np.clip(target_db - di_db, -args.max_gain_db, args.max_gain_db)
    gain_db = smooth_gain_frames(gain_db, smoothing_bins=args.training_smoothing_bins)

    feature_mean = di_db.mean(axis=0).astype(np.float32)
    feature_std = di_db.std(axis=0).astype(np.float32)
    features = ((di_db - feature_mean) / np.maximum(feature_std, 1e-6)).astype(np.float32)
    targets = (gain_db / args.gain_scale_db).astype(np.float32)

    order = np.arange(frame_count)
    rng.shuffle(order)
    validation_count = int(round(frame_count * args.validation_fraction))
    validation_count = max(0, min(validation_count, frame_count // 3))
    validation_rows = order[:validation_count]
    train_rows = order[validation_count:]

    x_train = features[train_rows]
    y_train = targets[train_rows]
    x_val = features[validation_rows] if validation_count else np.empty((0, features.shape[1]), dtype=np.float32)
    y_val = targets[validation_rows] if validation_count else np.empty((0, targets.shape[1]), dtype=np.float32)

    params = init_mlx_mlp_params(
        mx=mx,
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        output_dim=targets.shape[1],
    )
    first_moment = mlx_tree_zeros_like(mx, params)
    second_moment = mlx_tree_zeros_like(mx, params)

    def loss_fn(candidate_params, x_batch, y_batch):
        prediction = mlx_mlp_forward(mx, candidate_params, x_batch)
        error = prediction - y_batch
        smooth_penalty = mx.mean(mx.square(prediction[:, 1:] - prediction[:, :-1]))
        return mx.mean(mx.square(error)) + (args.smoothness_weight * smooth_penalty)

    value_and_grad = mx.value_and_grad(loss_fn)
    step = 0
    history = []

    print("Training MLX full-spectrum bridge...")
    print(f"Frames: train={len(x_train)} validation={len(x_val)} bins={features.shape[1]}")
    print(f"FFT size={args.fft_size} hop={args.hop_size} sample_rate={di_rate}")

    for epoch in range(1, args.epochs + 1):
        epoch_order = rng.permutation(len(x_train))
        epoch_loss = 0.0

        for batch_start in range(0, len(epoch_order), args.batch_size):
            batch_rows = epoch_order[batch_start : batch_start + args.batch_size]
            x_batch = mx.array(x_train[batch_rows])
            y_batch = mx.array(y_train[batch_rows])
            loss, grads = value_and_grad(params, x_batch, y_batch)
            step += 1
            params, first_moment, second_moment = mlx_adam_update(
                mx=mx,
                params=params,
                grads=grads,
                first_moment=first_moment,
                second_moment=second_moment,
                step=step,
                learning_rate=args.learning_rate,
            )
            mx.eval(loss, *params.values(), *first_moment.values(), *second_moment.values())
            epoch_loss += float(loss.item()) * len(batch_rows)

        train_loss = epoch_loss / max(1, len(x_train))
        if len(x_val):
            val_loss = loss_fn(params, mx.array(x_val), mx.array(y_val))
            mx.eval(val_loss)
            val_loss_float = float(val_loss.item())
        else:
            val_loss_float = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss_float})
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} validation_loss={val_loss_float:.6f}")

    mx.eval(*params.values())
    params_np = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    metadata = {
        "model_version": MLX_SPECTRAL_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Full-spectrum MLX bridge from clean DI spectrogram to SM57 amp/cab spectrogram.",
        "sample_rate_hz": int(di_rate),
        "alignment_lag_samples": int(lag),
        "target_polarity_after_alignment": int(polarity),
        "fft_size": int(args.fft_size),
        "hop_size": int(args.hop_size),
        "frequency_bins": int(features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "gain_scale_db": float(args.gain_scale_db),
        "max_gain_db": float(args.max_gain_db),
        "training_smoothing_bins": int(args.training_smoothing_bins),
        "output_smoothing_bins": int(args.output_smoothing_bins),
        "feature_mean": [float(value) for value in feature_mean],
        "feature_std": [float(max(value, 1e-6)) for value in feature_std],
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "validation_fraction": float(args.validation_fraction),
            "smoothness_weight": float(args.smoothness_weight),
            "history": history,
        },
        "portfolio_note": "Frequency-domain MLX tone bridge. It learns per-frame spectral gain from DI to SM57 target.",
    }

    save_mlx_spectral_model(args.model, metadata, params_np)
    output_rate, rendered, _ = render_mlx_spectral_bridge(
        di_audio=di_aligned,
        sample_rate=di_rate,
        model_path=args.model,
        batch_frames=args.batch_frames,
    )
    write_wav_float(args.output, output_rate, rendered)

    target_audition = normalize_for_audition(target_aligned[: len(rendered)], peak=0.70)
    rendered_audition = normalize_for_audition(rendered, peak=0.70)
    min_compare = min(len(target_audition), len(rendered_audition))
    metrics = {
        "match_correlation": correlation(target_audition[:min_compare], rendered_audition[:min_compare]),
        "spectral_error_db": spectral_error_db(target_audition[:min_compare], rendered_audition[:min_compare], output_rate),
        "audition_peak_dbfs": peak_dbfs(rendered),
    }
    metadata["render_validation"] = metrics
    save_mlx_spectral_model(args.model, metadata, params_np)

    if args.comparison_output:
        silence = np.zeros(output_rate, dtype=np.float64)
        comparison = np.concatenate([target_audition[:min_compare], silence, rendered_audition[:min_compare]])
        write_wav_float(args.comparison_output, output_rate, comparison)

    print(f"Wrote MLX full-spectrum model: {args.model}")
    print(f"Wrote MLX full-spectrum render: {args.output}")
    if args.comparison_output:
        print(f"Wrote mic-then-spectrum comparison: {args.comparison_output}")
    print(f"Spectral bridge correlation: {metrics['match_correlation']:.3f}")
    print(f"Spectral bridge error: {metrics['spectral_error_db']:.2f} dB")


def run_apply_mlx_spectrum_command(args: argparse.Namespace) -> None:
    if args.batch_frames < 1:
        raise SystemExit("--batch-frames must be at least 1.")

    sample_rate, audio = read_wav_float(args.input)
    output_rate, output, metadata = render_mlx_spectral_bridge(
        di_audio=audio,
        sample_rate=sample_rate,
        model_path=args.model,
        batch_frames=args.batch_frames,
    )
    write_wav_float(args.output, output_rate, output)
    print(f"Wrote MLX full-spectrum output: {args.output}")
    print(f"Model bins: {metadata['frequency_bins']} fft_size={metadata['fft_size']}")


def run_cleanup_unused_takes_command(args: argparse.Namespace) -> None:
    cleanup_unused_dataset_takes(
        dataset_path=args.dataset,
        profile_family=args.profile_family,
        include_takes=args.include_take or [],
        exclude_takes=args.exclude_take or [],
        usable_only=not args.include_unusable,
        preferred_only=args.preferred_only,
        archive_dir=args.archive_dir,
        mode=args.cleanup_mode,
        apply_changes=args.apply,
        confirm_delete=args.confirm_delete_unused,
    )


def run_train_all_recordings_amp_command(args: argparse.Namespace) -> None:
    os.chdir(project_dir())
    pair_specs = discover_recording_pair_specs(
        recordings_dir=args.recordings_dir,
        include_level_tests=args.include_level_tests,
        usable_only=args.usable_only,
    )
    input_di = Path(args.input) if args.input else latest_clean_di_from_pairs(pair_specs)
    pair_specs, rig_plan = select_pair_specs_by_rig_policy(
        pair_specs,
        policy=str(args.rig_policy),
        input_path=input_di,
        requested_fingerprint=str(args.rig_fingerprint or ""),
    )

    print("All-recordings amp-dominant training plan:")
    print(f"Recordings directory: {args.recordings_dir}")
    print(f"Training pairs: {len(pair_specs)}")
    print(f"Rig groups: {len(rig_plan['group_counts'])} | policy={rig_plan['policy']}")
    for index, spec in enumerate(pair_specs, start=1):
        levels = spec.get("recording_levels", {})
        usable = levels.get("usable_for_training", "unknown")
        preferred = levels.get("preferred_for_training", "unknown")
        print(
            f"  {index:02d}: {Path(spec['di_path']).name} -> {Path(spec['target_path']).name} "
            f"usable={usable} preferred={preferred}"
        )

    if bool(getattr(args, "quality_gate", True)):
        print("Quality-gated amp data plan:")
        quality_reports = [
            quality_report_for_pair_spec(spec, sample_rate=int(args.model_sample_rate), context_radius=int(args.context_radius))
            for spec in pair_specs
        ]
        apply_recording_quality_context(quality_reports)
        quality_min_weight = float(getattr(args, "quality_min_weight", AMP_QUALITY_MIN_WEIGHT))
        kept_count = 0
        excluded_count = 0
        for index, (spec, report) in enumerate(zip(pair_specs, quality_reports), start=1):
            quality_weight = float(report.get("quality_weight", 0.0))
            excluded = quality_weight < quality_min_weight
            if excluded:
                excluded_count += 1
            else:
                kept_count += 1
            state = "EXCLUDE" if excluded else ("DOWNWEIGHT" if quality_weight < 0.90 else "KEEP")
            issue_text = "; ".join(report.get("issues", [])[:2]) or "good amp/mic difference"
            print(
                f"  {index:02d}: {state:10s} weight={quality_weight:.2f} "
                f"spec={float(report.get('di_gain_baseline_spectral_error_db', 0.0)):.2f}dB "
                f"corr={float(report.get('di_target_correlation', 0.0)):.3f} "
                f"{Path(spec['di_path']).name} ({issue_text})"
            )
        print(f"Quality-gated result: kept={kept_count} excluded={excluded_count}")

    comparison_target = args.comparison_target
    if comparison_target is None:
        inferred_target = matching_amp_target_path(input_di)
        if inferred_target is not None and inferred_target.exists():
            comparison_target = inferred_target

    print(f"Apply DI: {input_di}")
    if comparison_target is not None:
        print(f"Apply reference amp/mic target: {comparison_target}")
    else:
        print("Apply reference amp/mic target: none")

    if args.list_only:
        print("List complete. Run without --list-only to train and render.")
        return

    train_args = argparse.Namespace(
        di=None,
        target=None,
        mic_position=args.mic_position,
        dataset=None,
        recordings_dir=args.recordings_dir,
        include_level_tests=args.include_level_tests,
        recordings_usable_only=args.usable_only,
        profile_family="",
        include_take=[],
        exclude_take=[],
        preferred_only=False,
        include_unusable=True,
        cleanup_unused=False,
        cleanup_mode="archive",
        cleanup_archive_dir=Path("archived_unused_takes"),
        confirm_delete_unused=False,
        extra_pair=[],
        model=args.model,
        output=args.training_output,
        comparison_output=args.training_comparison_output,
        model_sample_rate=args.model_sample_rate,
        context_radius=args.context_radius,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        lr_patience=args.lr_patience,
        lr_decay=args.lr_decay,
        min_learning_rate=args.min_learning_rate,
        min_delta=args.min_delta,
        max_train_samples=args.max_train_samples,
        max_training_seconds=args.max_training_seconds,
        validation_fraction=args.validation_fraction,
        take_sampling="balanced",
        conditioning_mode="source-stats",
        mic_position_conditioning=True,
        loss_mode=args.loss_mode,
        detail_chunk_samples=args.detail_chunk_samples,
        detail_chunks_per_epoch=args.detail_chunks_per_epoch,
        transient_loss_weight=args.transient_loss_weight,
        highfreq_loss_weight=args.highfreq_loss_weight,
        envelope_loss_weight=args.envelope_loss_weight,
        esr_loss_weight=args.esr_loss_weight,
        spectral_loss_weight=args.spectral_loss_weight,
        cab_lowpass_hz=args.cab_lowpass_hz,
        cab_highpass_hz=args.cab_highpass_hz,
        cab_presence_db=args.cab_presence_db,
        cab_air_db=args.cab_air_db,
        amp_anchor_strength=args.amp_anchor_strength,
        amp_anchor_smoothing_bins=args.amp_anchor_smoothing_bins,
        amp_anchor_max_gain_db=args.amp_anchor_max_gain_db,
        render_sample_rate=args.render_sample_rate,
        model_input_trim_db=args.model_input_trim_db,
        render_limiter="soft",
        output_peak_dbfs=args.output_peak_dbfs,
        per_take_output_dir=args.per_take_output_dir,
        skip_per_take_validation=args.skip_per_take_validation,
        chunk_samples=args.chunk_samples,
        print_every=args.print_every,
        seed=args.seed,
        quality_gate=args.quality_gate,
        quality_exclude_bad=args.quality_exclude_bad,
        quality_min_weight=args.quality_min_weight,
        rig_policy=args.rig_policy,
        rig_fingerprint=args.rig_fingerprint,
        rig_match_input=input_di,
        preserve_training_levels=args.preserve_training_levels,
        robust_loss_delta=args.robust_loss_delta,
        max_loss_spike_ratio=args.max_loss_spike_ratio,
        hybrid_cabinet_profile=args.hybrid_cabinet_profile,
        hybrid_cabinet_mix=args.hybrid_cabinet_mix,
        max_heldout_spectral_error_db=args.max_heldout_spectral_error_db,
        min_heldout_correlation=args.min_heldout_correlation,
        max_heldout_level_error_db=args.max_heldout_level_error_db,
        max_heldout_mean_spectral_error_db=args.max_heldout_mean_spectral_error_db,
        min_heldout_mean_correlation=args.min_heldout_mean_correlation,
        min_heldout_pass_rate=args.min_heldout_pass_rate,
        min_existing_model_improvement_db=args.min_existing_model_improvement_db,
        max_heldout_pair_regression_db=args.max_heldout_pair_regression_db,
        allow_failed_validation=args.allow_failed_validation,
    )
    run_train_mlx_amp_command(train_args)

    apply_args = argparse.Namespace(
        input=input_di,
        model=args.model,
        output=args.output,
        mic_position=args.mic_position or None,
        chunk_samples=args.chunk_samples,
        cab_lowpass_hz=args.cab_lowpass_hz,
        cab_highpass_hz=args.cab_highpass_hz,
        cab_presence_db=args.cab_presence_db,
        cab_air_db=args.cab_air_db,
        render_sample_rate=args.render_sample_rate,
        model_input_trim_db=args.model_input_trim_db,
        render_limiter="soft",
        output_peak_dbfs=args.output_peak_dbfs,
        target_level_match="rms" if comparison_target is not None else "off",
        mic_imprint_strength=args.mic_imprint_strength if comparison_target is not None else 0.0,
        mic_imprint_smoothing_bins=args.mic_imprint_smoothing_bins,
        mic_imprint_max_gain_db=args.mic_imprint_max_gain_db,
        amp_tone_guard=args.amp_tone_guard,
        amp_tone_guard_min_improvement_db=args.amp_tone_guard_min_improvement_db,
        amp_tone_guard_min_movement_db=args.amp_tone_guard_min_movement_db,
        comparison_target=comparison_target,
        comparison_output=args.comparison_output,
        rig_fingerprint=args.rig_fingerprint or None,
        hybrid_cabinet_profile=args.hybrid_cabinet_profile,
        hybrid_cabinet_mix=args.hybrid_cabinet_mix,
    )
    run_apply_mlx_amp_command(apply_args)


def run_train_mlx_amp_command(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.context_radius < 1:
        raise SystemExit("--context-radius must be at least 1.")
    if args.max_train_samples < 64:
        raise SystemExit("--max-train-samples must be at least 64.")
    if args.chunk_samples < 1024:
        raise SystemExit("--chunk-samples must be at least 1024.")
    if args.detail_chunk_samples < 128:
        raise SystemExit("--detail-chunk-samples must be at least 128.")
    if args.detail_chunks_per_epoch < 1:
        raise SystemExit("--detail-chunks-per-epoch must be at least 1.")
    if not 0.05 <= float(args.validation_fraction) <= 0.40:
        raise SystemExit("--validation-fraction must be between 0.05 and 0.40 for held-out promotion.")
    if float(getattr(args, "gradient_clip_norm", 0.0)) <= 0.0:
        raise SystemExit("--gradient-clip-norm must be greater than zero.")
    if float(getattr(args, "robust_loss_delta", 0.08)) <= 0.0:
        raise SystemExit("--robust-loss-delta must be greater than zero.")
    if float(getattr(args, "max_loss_spike_ratio", 10.0)) <= 1.0:
        raise SystemExit("--max-loss-spike-ratio must be greater than one.")
    if not 0.0 <= float(getattr(args, "hybrid_cabinet_mix", 1.0)) <= 1.0:
        raise SystemExit("--hybrid-cabinet-mix must be between zero and one.")
    if not 0.0 <= float(getattr(args, "min_heldout_pass_rate", AMP_HELDOUT_MIN_PASS_RATE)) <= 1.0:
        raise SystemExit("--min-heldout-pass-rate must be between zero and one.")
    hybrid_profile_value = getattr(args, "hybrid_cabinet_profile", None)
    if hybrid_profile_value and not Path(hybrid_profile_value).exists():
        raise SystemExit(f"Measured hybrid cabinet profile not found: {hybrid_profile_value}")
    if min(
        args.transient_loss_weight,
        args.highfreq_loss_weight,
        args.envelope_loss_weight,
        getattr(args, "esr_loss_weight", 0.0),
        getattr(args, "spectral_loss_weight", 0.0),
    ) < 0.0:
        raise SystemExit("Detail loss weights must be zero or greater.")

    extra_pairs = args.extra_pair or []
    mx = require_mlx()
    rng = np.random.default_rng(args.seed)

    model_rate = int(args.model_sample_rate)
    mic_position_conditioning_enabled = bool(
        args.mic_position_conditioning and args.conditioning_mode == "source-stats"
    )

    pair_specs = []
    recordings_dir = getattr(args, "recordings_dir", None)
    if recordings_dir:
        pair_specs.extend(
            discover_recording_pair_specs(
                recordings_dir=Path(recordings_dir),
                include_level_tests=bool(getattr(args, "include_level_tests", False)),
                usable_only=bool(getattr(args, "recordings_usable_only", False)),
            )
        )

    if args.dataset:
        dataset_entries = dataset_selected_take_entries(
            dataset_path=args.dataset,
            profile_family=args.profile_family,
            include_takes=args.include_take or [],
            exclude_takes=args.exclude_take or [],
            usable_only=not args.include_unusable,
            preferred_only=args.preferred_only,
        )
        if not dataset_entries:
            raise SystemExit(f"No training pairs matched dataset filters in {args.dataset}.")
        pair_specs.extend(
            enrich_pair_spec_rig_identity(
                {
                    "di_path": Path(item["clean_di_wav"]),
                    "target_path": Path(item["amp_mic_target_wav"]),
                    "take_name": str(item.get("take_name", "")),
                    "take_metadata": dict(item.get("take_metadata", {})),
                    "recording_levels": dict(item.get("recording_levels", {})),
                    "hardware_manifest_path": item.get("hardware_manifest"),
                    "source": "dataset",
                }
            )
            for item in dataset_entries
        )

    direct_metadata = {"mic_position": str(args.mic_position or "")}
    if args.di or args.target:
        if not args.di or not args.target:
            raise SystemExit("Use both --di and --target, or use --dataset.")
        pair_specs.append(
            {
                "di_path": Path(args.di),
                "target_path": Path(args.target),
                "take_name": Path(args.di).stem,
                "take_metadata": direct_metadata,
                "recording_levels": {},
                "source": "direct",
            }
        )
    pair_specs.extend(
        {
            "di_path": Path(pair[0]),
            "target_path": Path(pair[1]),
            "take_name": Path(pair[0]).stem,
            "take_metadata": direct_metadata,
            "recording_levels": {},
            "source": "extra_pair",
        }
        for pair in extra_pairs
    )
    seen_pair_specs = set()
    deduped_pair_specs = []
    for spec in pair_specs:
        key = (str(spec["di_path"]), str(spec["target_path"]))
        if key in seen_pair_specs:
            continue
        seen_pair_specs.add(key)
        deduped_pair_specs.append(spec)
    pair_specs = deduped_pair_specs
    if not pair_specs:
        raise SystemExit("At least one DI/target pair is required. Use --di/--target or --dataset.")

    rig_policy = str(getattr(args, "rig_policy", "conditioned"))
    rig_match_input = getattr(args, "rig_match_input", None) or args.di
    pair_specs, rig_group_report = select_pair_specs_by_rig_policy(
        pair_specs,
        policy=rig_policy,
        input_path=Path(rig_match_input) if rig_match_input else None,
        requested_fingerprint=str(getattr(args, "rig_fingerprint", "") or ""),
    )
    known_rig_fingerprints = sorted({str(spec["rig_fingerprint"]) for spec in pair_specs})

    quality_gate_enabled = bool(getattr(args, "quality_gate", True))
    quality_exclude_bad = bool(getattr(args, "quality_exclude_bad", True))
    quality_min_weight = float(getattr(args, "quality_min_weight", AMP_QUALITY_MIN_WEIGHT))
    quality_min_weight = float(np.clip(quality_min_weight, 0.0, 1.0))
    preserve_training_levels = bool(getattr(args, "preserve_training_levels", True))
    input_scale = 1.0 if preserve_training_levels else None
    training_pairs = []
    for pair_index, spec in enumerate(pair_specs, start=1):
        di_path = Path(spec["di_path"])
        target_path = Path(spec["target_path"])
        take_metadata = dict(spec.get("take_metadata", {}))
        recording_levels = dict(spec.get("recording_levels", {}))
        mic_position = str(take_metadata.get("mic_position") or args.mic_position or "")
        di_rate, di_audio = read_wav_float(di_path)
        target_rate, target_audio = read_wav_float(target_path)
        di_audio = resample_if_needed(di_audio, di_rate, model_rate)
        target_audio = resample_if_needed(target_audio, target_rate, model_rate)
        di_audio = remove_dc(di_audio)
        target_audio = remove_dc(target_audio)
        if not preserve_training_levels:
            di_audio = normalize_peak(di_audio, peak=0.95)
            target_audio = normalize_peak(target_audio, peak=0.95)
        di_aligned, target_aligned, lag, polarity = align_pair_fractional(
            di_audio,
            target_audio,
            max_lag_s=0.05,
            sample_rate=model_rate,
        )
        min_len = min(len(di_aligned), len(target_aligned))
        di_aligned = di_aligned[:min_len]
        target_aligned = target_aligned[:min_len]
        if input_scale is None:
            feature_audio = normalize_basis_signal(di_aligned).astype(np.float32)
        else:
            feature_audio = np.clip(di_aligned / max(float(input_scale), 1e-6), -4.0, 4.0).astype(np.float32)

        usable_end = min_len - args.context_radius
        if args.max_training_seconds > 0:
            usable_end = min(usable_end, int(round(args.max_training_seconds * model_rate)))

        validation_samples = max(
            int(args.detail_chunk_samples),
            int(round(max(0, usable_end - args.context_radius) * float(args.validation_fraction))),
        )
        validation_start = max(args.context_radius + 64, usable_end - validation_samples)
        candidate_indices = np.arange(
            args.context_radius,
            max(args.context_radius + 1, validation_start),
            dtype=np.int64,
        )
        if len(candidate_indices) < 64:
            raise SystemExit(f"Not enough usable samples for MLX amp training in pair {pair_index}: {di_path}")

        quality_report = {}
        quality_weight = 1.0
        if quality_gate_enabled:
            quality_report = recording_pair_quality_report(
                di_aligned,
                target_aligned,
                sample_rate=model_rate,
                di_path=di_path,
                target_path=target_path,
                lag=lag,
                polarity=polarity,
            )
            quality_weight = float(quality_report.get("quality_weight", 1.0))

        conditioning_features = amp_conditioning_features(
            di_aligned,
            model_rate,
            conditioning_mode=args.conditioning_mode,
            mic_position=mic_position,
            include_mic_position=mic_position_conditioning_enabled,
            input_scale=input_scale,
            rig_fingerprint_value=str(spec.get("rig_fingerprint", "")),
            known_rig_fingerprints=known_rig_fingerprints,
        )

        training_pairs.append(
            {
                "index": pair_index,
                "take_name": str(spec.get("take_name", "")),
                "di_path": di_path,
                "target_path": target_path,
                "take_metadata": take_metadata,
                "recording_levels": recording_levels,
                "source": str(spec.get("source", "")),
                "rig_identity": dict(spec.get("rig_identity", {})),
                "rig_fingerprint": str(spec.get("rig_fingerprint", "")),
                "mic_position": mic_position,
                "di_aligned": di_aligned,
                "target_aligned": target_aligned,
                "feature_audio": feature_audio,
                "lag": float(lag),
                "polarity": int(polarity),
                "min_len": int(min_len),
                "usable_end": int(usable_end),
                "validation_start": int(validation_start),
                "candidate_indices": candidate_indices,
                "conditioning_features": conditioning_features,
                "quality_weight": float(quality_weight),
                "quality_report": quality_report,
            }
        )

    quality_excluded_pairs = []
    if quality_gate_enabled:
        apply_recording_quality_context([pair["quality_report"] for pair in training_pairs])
        for pair in training_pairs:
            quality_report = dict(pair.get("quality_report", {}))
            quality_weight = float(np.clip(quality_report.get("quality_weight", pair.get("quality_weight", 1.0)), 0.0, 1.0))
            quality_excluded = quality_weight < quality_min_weight
            quality_report["quality_gate_excluded"] = bool(quality_excluded)
            pair["quality_weight"] = quality_weight
            pair["quality_report"] = quality_report
            pair["quality_gate_excluded"] = bool(quality_excluded)
        quality_excluded_pairs = [pair for pair in training_pairs if pair.get("quality_gate_excluded", False)]
        if quality_exclude_bad:
            training_pairs = [pair for pair in training_pairs if not pair.get("quality_gate_excluded", False)]
        if not training_pairs:
            raise SystemExit(
                "Quality gate excluded every training pair. Re-record stronger amp/mic targets or rerun with --no-quality-gate."
            )

    stats_feature_parts = []
    scale_parts = []
    stat_budget_per_pair = max(64, min(8000, args.max_train_samples) // max(1, len(training_pairs)))
    for pair in training_pairs:
        stat_indices = np.array(pair["candidate_indices"], dtype=np.int64)
        quality_weight = float(np.clip(pair.get("quality_weight", 1.0), 0.05, 1.0))
        stat_budget = max(64, int(round(stat_budget_per_pair * quality_weight)))
        if len(stat_indices) > stat_budget:
            stat_indices = rng.choice(stat_indices, size=stat_budget, replace=False)
        rng.shuffle(stat_indices)
        stats_feature_parts.append(
            build_amp_window_features(
                pair["di_aligned"],
                stat_indices,
                context_radius=args.context_radius,
                conditioning_features=pair["conditioning_features"],
                input_scale=input_scale,
                prepared_audio=pair["feature_audio"],
            )
        )
        scale_parts.append(pair["target_aligned"][stat_indices])

    raw_features = np.concatenate(stats_feature_parts, axis=0)
    feature_mean = raw_features.mean(axis=0).astype(np.float32)
    feature_std = raw_features.std(axis=0).astype(np.float32)
    input_feature_count = int(raw_features.shape[1])
    target_scale = max(float(np.percentile(np.abs(np.concatenate(scale_parts)), 99.5)), 1e-5)
    primary_pair = training_pairs[0]
    default_rig_fingerprint = str(primary_pair.get("rig_fingerprint", ""))
    if rig_match_input:
        resolved_match_input = Path(rig_match_input).resolve()
        for pair in training_pairs:
            if Path(pair["di_path"]).resolve() == resolved_match_input:
                default_rig_fingerprint = str(pair.get("rig_fingerprint", default_rig_fingerprint))
                break

    params = init_mlx_mlp_params(
        mx=mx,
        input_dim=input_feature_count,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        output_dim=1,
    )
    first_moment = mlx_tree_zeros_like(mx, params)
    second_moment = mlx_tree_zeros_like(mx, params)

    def mse_loss_fn(candidate_params, x_batch, y_batch):
        return mlx_mse_loss(mx, candidate_params, x_batch, y_batch)

    def detail_loss_fn(candidate_params, x_batch, y_batch):
        return mlx_amp_detail_loss(
            mx,
            candidate_params,
            x_batch,
            y_batch,
            transient_weight=args.transient_loss_weight,
            highfreq_weight=args.highfreq_loss_weight,
            envelope_weight=args.envelope_loss_weight,
            robust_delta=float(getattr(args, "robust_loss_delta", 0.08)),
        )

    def detail_spectral_loss_fn(candidate_params, x_batch, y_batch):
        return mlx_amp_detail_spectral_loss(
            mx,
            candidate_params,
            x_batch,
            y_batch,
            transient_weight=args.transient_loss_weight,
            highfreq_weight=args.highfreq_loss_weight,
            envelope_weight=args.envelope_loss_weight,
            esr_weight=float(getattr(args, "esr_loss_weight", 0.35)),
            spectral_weight=float(getattr(args, "spectral_loss_weight", 0.22)),
            robust_delta=float(getattr(args, "robust_loss_delta", 0.08)),
        )

    if args.loss_mode == "mse":
        loss_fn = mse_loss_fn
    elif args.loss_mode == "detail-spectral":
        loss_fn = detail_spectral_loss_fn
    else:
        loss_fn = detail_loss_fn
    value_and_grad = mx.value_and_grad(loss_fn)
    step = 0
    history = []
    learning_rate = float(args.learning_rate)
    gradient_clip_norm = float(getattr(args, "gradient_clip_norm", 1.0))
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 18))
    lr_patience = int(getattr(args, "lr_patience", 6))
    lr_decay = float(getattr(args, "lr_decay", 0.5))
    min_learning_rate = float(getattr(args, "min_learning_rate", 0.00002))
    min_delta = float(getattr(args, "min_delta", 0.00001))
    best_validation_loss = float("inf")
    best_epoch = 0
    best_params_np = None
    stale_epochs = 0
    max_loss_spike_ratio = float(getattr(args, "max_loss_spike_ratio", 10.0))
    recent_step_losses: list[float] = []
    skipped_unstable_steps = 0

    def stable_training_step(x_batch, y_batch) -> tuple[float | None, float]:
        nonlocal params, first_moment, second_moment, step, learning_rate, skipped_unstable_steps
        loss, grads = value_and_grad(params, x_batch, y_batch)
        mx.eval(loss, *grads.values())
        loss_value = float(loss.item())
        baseline = float(np.median(recent_step_losses[-64:])) if recent_step_losses else loss_value
        spike = bool(
            len(recent_step_losses) >= 8
            and loss_value > max(1e-6, baseline) * max_loss_spike_ratio
        )
        if not np.isfinite(loss_value) or spike:
            skipped_unstable_steps += 1
            learning_rate = max(min_learning_rate, learning_rate * lr_decay)
            return None, float("inf")
        grads, grad_norm = mlx_clip_gradient_tree(mx, grads, gradient_clip_norm)
        step += 1
        params, first_moment, second_moment = mlx_adam_update(
            mx=mx,
            params=params,
            grads=grads,
            first_moment=first_moment,
            second_moment=second_moment,
            step=step,
            learning_rate=learning_rate,
        )
        mx.eval(*params.values(), *first_moment.values(), *second_moment.values())
        recent_step_losses.append(loss_value)
        return loss_value, grad_norm

    def update_checkpoint(epoch: int, validation_loss: float) -> tuple[bool, bool]:
        nonlocal best_validation_loss, best_epoch, best_params_np, stale_epochs, learning_rate
        improved = bool(np.isfinite(validation_loss) and validation_loss < best_validation_loss - min_delta)
        if improved:
            mx.eval(*params.values())
            best_params_np = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
            best_validation_loss = float(validation_loss)
            best_epoch = int(epoch)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if lr_patience > 0 and stale_epochs % lr_patience == 0:
                learning_rate = max(min_learning_rate, learning_rate * lr_decay)
        should_stop = bool(early_stopping_patience > 0 and stale_epochs >= early_stopping_patience)
        return improved, should_stop

    print("Training MLX neural amp model...")
    print(f"Loss mode: {args.loss_mode}")
    print(f"Conditioning mode: {args.conditioning_mode}")
    print(f"Mic position conditioning: {'on' if mic_position_conditioning_enabled else 'off'}")
    print(
        f"Rig policy: {rig_group_report['policy']} groups={len(rig_group_report['group_counts'])} "
        f"selected_pairs={rig_group_report['selected_pair_count']}/{rig_group_report['total_pair_count']}"
    )
    for fingerprint_value, count in sorted(
        rig_group_report["group_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        selected_marker = " selected" if rig_group_report.get("selected_fingerprint") == fingerprint_value else ""
        print(f"  Rig {fingerprint_value}: {count} takes{selected_marker}")
    print(f"Capture level preservation: {'on' if preserve_training_levels else 'off (legacy normalization)'}")
    if quality_gate_enabled:
        print(
            "Quality gate: "
            f"on min_weight={quality_min_weight:.2f} "
            f"kept={len(training_pairs)} "
            f"excluded={len(quality_excluded_pairs) if quality_exclude_bad else 0} "
            f"downweighted={sum(1 for pair in training_pairs if float(pair.get('quality_weight', 1.0)) < 0.90)}"
        )
        if quality_excluded_pairs and quality_exclude_bad:
            print("Quality gate excluded:")
            for pair in quality_excluded_pairs:
                report = dict(pair.get("quality_report", {}))
                issue_text = "; ".join(report.get("issues", [])[:3]) or "low quality score"
                print(
                    f"  Pair {pair['index']}: {pair['di_path'].name} "
                    f"weight={float(pair.get('quality_weight', 0.0)):.2f} "
                    f"spec={float(report.get('di_gain_baseline_spectral_error_db', 0.0)):.2f}dB "
                    f"corr={float(report.get('di_target_correlation', 0.0)):.3f} "
                    f"issues={issue_text}"
                )
    else:
        print("Quality gate: off")
    print(f"Training pairs: {len(training_pairs)}")
    for pair in training_pairs:
        seconds = pair["usable_end"] / model_rate
        mic_label = pair["mic_position"] or "unlabeled mic position"
        quality_label = ""
        if quality_gate_enabled:
            quality_label = f", quality={float(pair.get('quality_weight', 1.0)):.2f}"
        print(
            f"  Pair {pair['index']}: {pair['di_path'].name} -> {pair['target_path'].name} "
            f"({seconds:.1f}s, mic={mic_label}, source={pair['source'] or 'manual'}{quality_label})"
        )
    memory_ms = 1000.0 * ((2 * int(args.context_radius)) + 1) / model_rate
    print(
        f"Model sample rate={model_rate} context_radius={args.context_radius} "
        f"waveform_memory={memory_ms:.1f} ms"
    )
    print("This learns SM57 waveform detail from DI waveform windows, not just an EQ curve.")

    if args.loss_mode == "mse":
        feature_parts = []
        target_parts = []
        validation_feature_parts = []
        validation_target_parts = []
        sample_budget_per_pair = max(64, args.max_train_samples // max(1, len(training_pairs)))
        for pair in training_pairs:
            pair_indices = np.array(pair["candidate_indices"], dtype=np.int64)
            quality_weight = float(np.clip(pair.get("quality_weight", 1.0), 0.05, 1.0))
            pair_budget = max(64, int(round(sample_budget_per_pair * quality_weight)))
            if len(pair_indices) > pair_budget:
                pair_indices = rng.choice(pair_indices, size=pair_budget, replace=False)
            rng.shuffle(pair_indices)
            raw_train_features = build_amp_window_features(
                pair["di_aligned"],
                pair_indices,
                context_radius=args.context_radius,
                conditioning_features=pair["conditioning_features"],
                input_scale=input_scale,
                prepared_audio=pair["feature_audio"],
            )
            feature_parts.append(((raw_train_features - feature_mean) / np.maximum(feature_std, 1e-6)).astype(np.float32))
            target_parts.append((pair["target_aligned"][pair_indices] / target_scale).reshape(-1, 1).astype(np.float32))

            validation_indices = np.arange(
                int(pair["validation_start"]),
                int(pair["usable_end"]),
                dtype=np.int64,
            )
            validation_budget = max(64, int(round(sample_budget_per_pair * float(args.validation_fraction))))
            if len(validation_indices) > validation_budget:
                validation_indices = rng.choice(validation_indices, size=validation_budget, replace=False)
            raw_validation_features = build_amp_window_features(
                pair["di_aligned"],
                validation_indices,
                context_radius=args.context_radius,
                conditioning_features=pair["conditioning_features"],
                input_scale=input_scale,
                prepared_audio=pair["feature_audio"],
            )
            validation_feature_parts.append(
                ((raw_validation_features - feature_mean) / np.maximum(feature_std, 1e-6)).astype(np.float32)
            )
            validation_target_parts.append(
                (pair["target_aligned"][validation_indices] / target_scale).reshape(-1, 1).astype(np.float32)
            )

        features = np.concatenate(feature_parts, axis=0)
        targets = np.concatenate(target_parts, axis=0)
        shuffle_order = rng.permutation(len(features))
        features = features[shuffle_order]
        targets = targets[shuffle_order]

        x_train = features
        y_train = targets
        x_val = np.concatenate(validation_feature_parts, axis=0)
        y_val = np.concatenate(validation_target_parts, axis=0)

        print(f"Samples: train={len(x_train)} validation={len(x_val)} features={input_feature_count}")

        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(x_train))
            epoch_loss = 0.0
            epoch_rows = 0

            for batch_start in range(0, len(order), args.batch_size):
                batch_rows = order[batch_start : batch_start + args.batch_size]
                x_batch = mx.array(x_train[batch_rows])
                y_batch = mx.array(y_train[batch_rows])
                loss_value, _ = stable_training_step(x_batch, y_batch)
                if loss_value is not None:
                    epoch_loss += loss_value * len(batch_rows)
                    epoch_rows += len(batch_rows)

            train_loss = epoch_loss / max(1, epoch_rows)
            if len(x_val):
                val_loss = loss_fn(params, mx.array(x_val), mx.array(y_val))
                mx.eval(val_loss)
                val_loss_float = float(val_loss.item())
            else:
                val_loss_float = train_loss

            improved, should_stop = update_checkpoint(epoch, val_loss_float)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_loss_float,
                    "learning_rate": learning_rate,
                    "best_checkpoint": improved,
                }
            )
            if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
                marker = " best" if improved else ""
                print(
                    f"Epoch {epoch:03d}: train_loss={train_loss:.6f} validation_loss={val_loss_float:.6f} "
                    f"lr={learning_rate:.7f}{marker}"
                )
            if should_stop:
                print(f"Early stopping at epoch {epoch}; restoring best epoch {best_epoch}.")
                break

    else:
        detail_pairs = []
        for pair in training_pairs:
            max_chunk_start = (
                min(pair["validation_start"], pair["min_len"] - args.context_radius)
                - args.detail_chunk_samples
            )
            min_chunk_start = args.context_radius
            if max_chunk_start > min_chunk_start:
                pair_for_chunks = dict(pair)
                pair_for_chunks["min_chunk_start"] = int(min_chunk_start)
                pair_for_chunks["max_chunk_start"] = int(max_chunk_start)
                detail_pairs.append(pair_for_chunks)

        if not detail_pairs:
            raise SystemExit(
                "Not enough usable audio for detail-mode chunk training. "
                "Try a shorter --detail-chunk-samples value."
            )

        quality_chunk_weights = np.asarray(
            [float(np.clip(pair.get("quality_weight", 1.0), 0.0, 1.0)) for pair in detail_pairs],
            dtype=np.float64,
        )
        if float(np.sum(quality_chunk_weights)) <= 1e-12:
            quality_chunk_weights = np.ones(len(detail_pairs), dtype=np.float64)

        if args.take_sampling == "balanced":
            chunk_weights = quality_chunk_weights
        else:
            chunk_weights = np.array(
                [
                    (pair["max_chunk_start"] - pair["min_chunk_start"] + 1)
                    for pair in detail_pairs
                ],
                dtype=np.float64,
            )
            chunk_weights *= quality_chunk_weights
        chunk_weights /= float(np.sum(chunk_weights) + 1e-12)

        validation_chunk_count = int(round(args.detail_chunks_per_epoch * args.validation_fraction))
        validation_chunk_count = max(
            len(detail_pairs) if args.validation_fraction > 0 else 0,
            min(validation_chunk_count, max(32, len(detail_pairs))),
        )
        validation_refs = []
        for pair_choice, pair in enumerate(detail_pairs):
            val_min = max(args.context_radius, int(pair["validation_start"]))
            val_max = int(pair["usable_end"]) - args.detail_chunk_samples
            if val_max >= val_min:
                validation_refs.append((pair_choice, int((val_min + val_max) // 2)))
        while len(validation_refs) < validation_chunk_count:
            pair_choice = int(rng.choice(len(detail_pairs), p=chunk_weights))
            pair = detail_pairs[pair_choice]
            val_min = max(args.context_radius, int(pair["validation_start"]))
            val_max = int(pair["usable_end"]) - args.detail_chunk_samples
            if val_max < val_min:
                continue
            chunk_start = int(rng.integers(val_min, val_max + 1))
            validation_refs.append((pair_choice, chunk_start))

        print(
            "Chunks: "
            f"train_per_epoch={args.detail_chunks_per_epoch} "
            f"validation={len(validation_refs)} "
            f"chunk_samples={args.detail_chunk_samples} "
            f"features={input_feature_count}"
        )
        print(
            "Detail weights: "
            f"transient={args.transient_loss_weight} "
            f"highfreq={args.highfreq_loss_weight} "
            f"envelope={args.envelope_loss_weight} "
            f"esr={float(getattr(args, 'esr_loss_weight', 0.0))} "
            f"spectral={float(getattr(args, 'spectral_loss_weight', 0.0))}"
        )
        print(f"Take sampling: {args.take_sampling}")
        if quality_gate_enabled:
            sampled = sorted(
                [
                    (
                        float(chunk_weights[index]),
                        int(pair["index"]),
                        Path(str(pair["di_path"])).name,
                        float(pair.get("quality_weight", 1.0)),
                    )
                    for index, pair in enumerate(detail_pairs)
                ],
                reverse=True,
            )
            print("Quality-weighted chunk share:")
            for share, pair_index, name, quality_weight in sampled[:8]:
                print(f"  Pair {pair_index}: {share * 100:4.1f}% quality={quality_weight:.2f} {name}")

        def build_detail_batch(pair: dict, chunk_start: int) -> tuple[np.ndarray, np.ndarray]:
            sample_indices = np.arange(
                chunk_start,
                chunk_start + args.detail_chunk_samples,
                dtype=np.int64,
            )
            x_batch_np = build_amp_window_features(
                pair["di_aligned"],
                sample_indices,
                context_radius=args.context_radius,
                feature_mean=feature_mean,
                feature_std=feature_std,
                conditioning_features=pair["conditioning_features"],
                input_scale=input_scale,
                prepared_audio=pair["feature_audio"],
            )
            y_batch_np = (pair["target_aligned"][sample_indices] / target_scale).reshape(-1, 1).astype(np.float32)
            return x_batch_np, y_batch_np

        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            accepted_steps = 0

            for _ in range(args.detail_chunks_per_epoch):
                pair_choice = int(rng.choice(len(detail_pairs), p=chunk_weights))
                pair = detail_pairs[pair_choice]
                chunk_start = int(rng.integers(pair["min_chunk_start"], pair["max_chunk_start"] + 1))
                x_batch_np, y_batch_np = build_detail_batch(pair, chunk_start)
                loss_value, _ = stable_training_step(mx.array(x_batch_np), mx.array(y_batch_np))
                if loss_value is not None:
                    epoch_loss += loss_value
                    accepted_steps += 1

            train_loss = epoch_loss / max(1, accepted_steps)
            if validation_refs:
                validation_losses = []
                for pair_choice, chunk_start in validation_refs:
                    x_val_np, y_val_np = build_detail_batch(detail_pairs[pair_choice], int(chunk_start))
                    val_loss = loss_fn(params, mx.array(x_val_np), mx.array(y_val_np))
                    mx.eval(val_loss)
                    validation_losses.append(float(val_loss.item()))
                val_loss_float = float(np.mean(validation_losses))
            else:
                val_loss_float = train_loss

            improved, should_stop = update_checkpoint(epoch, val_loss_float)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_loss_float,
                    "learning_rate": learning_rate,
                    "best_checkpoint": improved,
                }
            )
            if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
                marker = " best" if improved else ""
                print(
                    f"Epoch {epoch:03d}: train_loss={train_loss:.6f} validation_loss={val_loss_float:.6f} "
                    f"lr={learning_rate:.7f}{marker}"
                )
            if should_stop:
                print(f"Early stopping at epoch {epoch}; restoring best epoch {best_epoch}.")
                break

    if best_params_np is None:
        mx.eval(*params.values())
        best_params_np = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
        best_epoch = len(history)
        best_validation_loss = float(history[-1]["validation_loss"]) if history else float("inf")
    params = {key: mx.array(value) for key, value in best_params_np.items()}
    mx.eval(*params.values())
    params_np = best_params_np
    print(f"Restored best checkpoint: epoch {best_epoch} validation_loss={best_validation_loss:.6f}")
    conditioning_feature_names = []
    if args.conditioning_mode == "source-stats":
        conditioning_feature_names.extend(AMP_SOURCE_CONDITIONING_FEATURES)
        if mic_position_conditioning_enabled:
            conditioning_feature_names.extend(AMP_MIC_POSITION_FEATURES)
    conditioning_feature_names.extend([f"rig_{value}" for value in known_rig_fingerprints])
    default_mic_position = str(args.mic_position or primary_pair.get("mic_position", ""))
    known_mic_positions = sorted(
        {
            str(pair.get("mic_position", ""))
            for pair in training_pairs
            if str(pair.get("mic_position", "")).strip()
        }
        | ({str(args.mic_position)} if str(args.mic_position).strip() else set())
    )
    amp_tone_anchor = build_all_recordings_amp_tone_anchor(
        training_pairs,
        sample_rate=model_rate,
        strength=float(getattr(args, "amp_anchor_strength", 1.0)),
        smoothing_bins=int(getattr(args, "amp_anchor_smoothing_bins", AMP_TONE_ANCHOR_SMOOTHING_BINS)),
        max_gain_db=float(getattr(args, "amp_anchor_max_gain_db", AMP_TONE_ANCHOR_MAX_GAIN_DB)),
    )
    metadata = {
        "model_version": MLX_AMP_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Direct nonlinear neural amp model from clean DI waveform windows to SM57 amp/cab waveform.",
        "sample_rate_hz": int(model_rate),
        "default_render_sample_rate_hz": int(args.render_sample_rate or model_rate),
        "alignment_lag_samples": float(primary_pair["lag"]),
        "target_polarity_after_alignment": int(primary_pair["polarity"]),
        "context_radius": int(args.context_radius),
        "input_feature_count": int(input_feature_count),
        "hidden_dim": int(args.hidden_dim),
        "target_scale": float(target_scale),
        "input_scale": float(input_scale or 1.0),
        "preserve_input_level": bool(preserve_training_levels),
        "conditioning_mode": str(args.conditioning_mode),
        "conditioning_feature_names": conditioning_feature_names,
        "conditioning_feature_count": int(len(conditioning_feature_names)),
        "mic_position_conditioning": bool(mic_position_conditioning_enabled),
        "default_mic_position": default_mic_position,
        "known_mic_positions": known_mic_positions,
        "default_rig_fingerprint": default_rig_fingerprint,
        "known_rig_fingerprints": known_rig_fingerprints,
        "rig_groups": rig_group_report,
        "hybrid_cabinet_profile": str(getattr(args, "hybrid_cabinet_profile", "") or ""),
        "cab_lowpass_hz": float(args.cab_lowpass_hz),
        "cab_highpass_hz": float(args.cab_highpass_hz),
        "cab_presence_db": float(args.cab_presence_db),
        "cab_air_db": float(args.cab_air_db),
        "amp_tone_anchor": amp_tone_anchor,
        "feature_mean": [float(value) for value in feature_mean],
        "feature_std": [float(max(value, 1e-6)) for value in feature_std],
        "training": {
            "training_pair_count": int(len(training_pairs)),
            "training_pairs": [
                {
                    "index": int(pair["index"]),
                    "di": str(pair["di_path"]),
                    "target": str(pair["target_path"]),
                    "take_name": str(pair.get("take_name", "")),
                    "source": str(pair.get("source", "")),
                    "mic_position": str(pair.get("mic_position", "")),
                    "rig_identity": dict(pair.get("rig_identity", {})),
                    "rig_fingerprint": str(pair.get("rig_fingerprint", "")),
                    "take_metadata": dict(pair.get("take_metadata", {})),
                    "recording_levels": dict(pair.get("recording_levels", {})),
                    "alignment_lag_samples": float(pair["lag"]),
                    "target_polarity_after_alignment": int(pair["polarity"]),
                    "usable_seconds": float(pair["usable_end"] / model_rate),
                    "heldout_start_sample": int(pair["validation_start"]),
                    "heldout_seconds": float((pair["usable_end"] - pair["validation_start"]) / model_rate),
                    "conditioning_features": [float(value) for value in pair["conditioning_features"]],
                    "quality_weight": float(pair.get("quality_weight", 1.0)),
                    "quality_report": dict(pair.get("quality_report", {})),
                }
                for pair in training_pairs
            ],
            "quality_gate": {
                "enabled": bool(quality_gate_enabled),
                "exclude_bad": bool(quality_exclude_bad),
                "min_weight": float(quality_min_weight),
                "excluded_pair_count": int(len(quality_excluded_pairs) if quality_exclude_bad else 0),
                "excluded_pairs": [
                    {
                        "index": int(pair["index"]),
                        "di": str(pair["di_path"]),
                        "target": str(pair["target_path"]),
                        "quality_weight": float(pair.get("quality_weight", 0.0)),
                        "issues": list(dict(pair.get("quality_report", {})).get("issues", [])),
                    }
                    for pair in quality_excluded_pairs
                ]
                if quality_exclude_bad
                else [],
            },
            "loss_mode": str(args.loss_mode),
            "conditioning_mode": str(args.conditioning_mode),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
            "gradient_clip_norm": float(gradient_clip_norm),
            "robust_loss_delta": float(getattr(args, "robust_loss_delta", 0.08)),
            "max_loss_spike_ratio": float(max_loss_spike_ratio),
            "skipped_unstable_steps": int(skipped_unstable_steps),
            "early_stopping_patience": int(early_stopping_patience),
            "lr_patience": int(lr_patience),
            "lr_decay": float(lr_decay),
            "max_train_samples": int(args.max_train_samples),
            "max_training_seconds": float(args.max_training_seconds),
            "validation_fraction": float(args.validation_fraction),
            "take_sampling": str(args.take_sampling),
            "detail_chunk_samples": int(args.detail_chunk_samples),
            "detail_chunks_per_epoch": int(args.detail_chunks_per_epoch),
            "transient_loss_weight": float(args.transient_loss_weight),
            "highfreq_loss_weight": float(args.highfreq_loss_weight),
            "envelope_loss_weight": float(args.envelope_loss_weight),
            "esr_loss_weight": float(getattr(args, "esr_loss_weight", 0.0)),
            "spectral_loss_weight": float(getattr(args, "spectral_loss_weight", 0.0)),
            "history": history,
        },
        "portfolio_note": "MLX neural amp capture prototype. Best results need longer diverse DI/SM57 training takes.",
    }
    requested_model_path = Path(args.model)
    candidate_model_path = requested_model_path.with_name(
        f"{requested_model_path.stem}.candidate{requested_model_path.suffix}"
    )
    existing_model_path = requested_model_path if requested_model_path.exists() else None
    save_mlx_amp_model(candidate_model_path, metadata, params_np)
    hybrid_cabinet_value = getattr(args, "hybrid_cabinet_profile", None)
    hybrid_cabinet_profile = Path(hybrid_cabinet_value) if hybrid_cabinet_value else None

    output_rate, rendered, _ = render_mlx_amp_model(
        di_audio=primary_pair["di_aligned"],
        sample_rate=model_rate,
        model_path=candidate_model_path,
        chunk_samples=args.chunk_samples,
        cab_lowpass_hz=args.cab_lowpass_hz,
        cab_highpass_hz=args.cab_highpass_hz,
        cab_presence_db=args.cab_presence_db,
        cab_air_db=args.cab_air_db,
        render_sample_rate=args.render_sample_rate,
        model_input_trim_db=args.model_input_trim_db,
        render_limiter=args.render_limiter,
        output_peak_dbfs=args.output_peak_dbfs,
        mic_position=primary_pair["mic_position"],
        source_hint_path=primary_pair["di_path"],
        cabinet_profile=hybrid_cabinet_profile,
        cabinet_mix=float(getattr(args, "hybrid_cabinet_mix", 1.0)),
    )
    write_wav_float(args.output, output_rate, rendered)

    primary_target_for_render = resample_if_needed(
        primary_pair["target_aligned"],
        model_rate,
        output_rate,
    )
    target_audition = normalize_for_audition(primary_target_for_render[: len(rendered)], peak=0.70)
    rendered_audition = normalize_for_audition(rendered, peak=0.70)
    min_compare = min(len(target_audition), len(rendered_audition))
    metrics = {
        "match_correlation": correlation(target_audition[:min_compare], rendered_audition[:min_compare]),
        "spectral_error_db": spectral_error_db(target_audition[:min_compare], rendered_audition[:min_compare], output_rate),
        "audition_peak_dbfs": peak_dbfs(rendered),
    }
    metadata["render_validation"] = metrics

    per_take_metrics = []
    per_take_output_dir = args.per_take_output_dir
    skip_per_take_validation = bool(getattr(args, "skip_per_take_validation", False))
    if per_take_output_dir and not skip_per_take_validation:
        per_take_output_dir.mkdir(parents=True, exist_ok=True)

    validation_pairs = training_pairs
    existing_per_take_metrics = []
    existing_model_usable = existing_model_path is not None
    existing_model_error = ""
    for pair in validation_pairs:
        if pair["index"] == primary_pair["index"]:
            pair_rate = output_rate
            pair_rendered = rendered
        else:
            pair_rate, pair_rendered, _ = render_mlx_amp_model(
                di_audio=pair["di_aligned"],
                sample_rate=model_rate,
                model_path=candidate_model_path,
                chunk_samples=args.chunk_samples,
                cab_lowpass_hz=args.cab_lowpass_hz,
                cab_highpass_hz=args.cab_highpass_hz,
                cab_presence_db=args.cab_presence_db,
                cab_air_db=args.cab_air_db,
                render_sample_rate=args.render_sample_rate,
                model_input_trim_db=args.model_input_trim_db,
                render_limiter=args.render_limiter,
                output_peak_dbfs=args.output_peak_dbfs,
                mic_position=pair["mic_position"],
                source_hint_path=pair["di_path"],
                cabinet_profile=hybrid_cabinet_profile,
                cabinet_mix=float(getattr(args, "hybrid_cabinet_mix", 1.0)),
            )

        pair_target_for_render = resample_if_needed(pair["target_aligned"], model_rate, pair_rate)
        pair_di_for_render = resample_if_needed(pair["di_aligned"], model_rate, pair_rate)
        heldout_start = int(round(int(pair["validation_start"]) * pair_rate / model_rate))
        heldout_end = int(round(int(pair["usable_end"]) * pair_rate / model_rate))
        heldout_end = min(heldout_end, len(pair_di_for_render), len(pair_target_for_render), len(pair_rendered))
        heldout_start = int(np.clip(heldout_start, 0, max(0, heldout_end - 1)))
        heldout_di = pair_di_for_render[heldout_start:heldout_end]
        heldout_target = pair_target_for_render[heldout_start:heldout_end]
        heldout_render = pair_rendered[heldout_start:heldout_end]
        pair_metrics = heldout_amp_model_metrics(
            heldout_di,
            heldout_target,
            heldout_render,
            pair_rate,
            max_spectral_error_db=float(getattr(args, "max_heldout_spectral_error_db", AMP_HELDOUT_MAX_MEAN_SPECTRAL_ERROR_DB)),
            min_correlation=float(getattr(args, "min_heldout_correlation", AMP_HELDOUT_MIN_MEAN_CORRELATION)),
            max_level_error_db=float(getattr(args, "max_heldout_level_error_db", 3.0)),
            min_improvement_db=float(getattr(args, "amp_tone_guard_min_improvement_db", AMP_TONE_GUARD_MIN_IMPROVEMENT_DB)),
            min_movement_db=float(getattr(args, "amp_tone_guard_min_movement_db", AMP_TONE_GUARD_MIN_MOVEMENT_DB)),
        )
        pair_metrics.update(
            {
                "index": int(pair["index"]),
                "di": str(pair["di_path"]),
                "target": str(pair["target_path"]),
                "rig_fingerprint": str(pair.get("rig_fingerprint", "")),
                "heldout_start_sample": heldout_start,
                "heldout_end_sample": heldout_end,
                "audition_peak_dbfs": peak_dbfs(pair_rendered),
            }
        )
        per_take_metrics.append(pair_metrics)

        if existing_model_usable and existing_model_path is not None:
            try:
                _, existing_render, _ = render_mlx_amp_model(
                    di_audio=pair["di_aligned"],
                    sample_rate=model_rate,
                    model_path=existing_model_path,
                    chunk_samples=args.chunk_samples,
                    cab_lowpass_hz=args.cab_lowpass_hz,
                    cab_highpass_hz=args.cab_highpass_hz,
                    cab_presence_db=args.cab_presence_db,
                    cab_air_db=args.cab_air_db,
                    render_sample_rate=pair_rate,
                    model_input_trim_db=args.model_input_trim_db,
                    render_limiter=args.render_limiter,
                    output_peak_dbfs=args.output_peak_dbfs,
                    mic_position=pair["mic_position"],
                    source_hint_path=pair["di_path"],
                    cabinet_profile=hybrid_cabinet_profile,
                    cabinet_mix=float(getattr(args, "hybrid_cabinet_mix", 1.0)),
                )
                existing_heldout = existing_render[heldout_start:heldout_end]
                previous = heldout_amp_model_metrics(
                    heldout_di,
                    heldout_target,
                    existing_heldout,
                    pair_rate,
                    max_spectral_error_db=float(getattr(args, "max_heldout_spectral_error_db", AMP_HELDOUT_MAX_MEAN_SPECTRAL_ERROR_DB)),
                    min_correlation=float(getattr(args, "min_heldout_correlation", AMP_HELDOUT_MIN_MEAN_CORRELATION)),
                    max_level_error_db=float(getattr(args, "max_heldout_level_error_db", 3.0)),
                )
                previous["index"] = int(pair["index"])
                existing_per_take_metrics.append(previous)
            except (OSError, ValueError, KeyError, SystemExit) as exc:
                existing_model_usable = False
                existing_model_error = str(exc)
                existing_per_take_metrics = []

        pair_target = normalize_for_audition(pair_target_for_render[: len(pair_rendered)], peak=0.70)
        pair_render_audition = normalize_for_audition(pair_rendered, peak=0.70)
        pair_compare_len = min(len(pair_target), len(pair_render_audition))

        if per_take_output_dir and not skip_per_take_validation:
            safe_stem = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in pair["di_path"].stem
            )
            write_wav_float(
                per_take_output_dir / f"pair_{pair['index']:02d}_{safe_stem}_mlx_amp.wav",
                pair_rate,
                pair_rendered,
            )
            silence = np.zeros(pair_rate, dtype=np.float64)
            comparison = np.concatenate(
                [
                    pair_target[:pair_compare_len],
                    silence,
                    pair_render_audition[:pair_compare_len],
                ]
            )
            write_wav_float(
                per_take_output_dir / f"pair_{pair['index']:02d}_{safe_stem}_mic_then_mlx_amp.wav",
                pair_rate,
                comparison,
            )

    metadata["per_take_render_validation"] = per_take_metrics
    primary_heldout = next(
        (item for item in per_take_metrics if int(item["index"]) == int(primary_pair["index"])),
        per_take_metrics[0],
    )
    metadata["render_validation"] = primary_heldout
    candidate_aggregate = aggregate_heldout_amp_metrics(per_take_metrics)
    existing_aggregate = (
        aggregate_heldout_amp_metrics(existing_per_take_metrics)
        if existing_model_usable and len(existing_per_take_metrics) == len(per_take_metrics)
        else None
    )
    metadata["aggregate_render_validation"] = candidate_aggregate
    metadata["existing_model_validation"] = {
        "path": str(existing_model_path) if existing_model_path else None,
        "available": bool(existing_aggregate is not None),
        "error": existing_model_error or None,
        "aggregate": existing_aggregate,
        "per_take": existing_per_take_metrics,
    }
    promotion = amp_model_promotion_decision(
        candidate=candidate_aggregate,
        existing=existing_aggregate,
        max_mean_spectral_error_db=float(
            getattr(args, "max_heldout_mean_spectral_error_db", AMP_HELDOUT_MAX_MEAN_SPECTRAL_ERROR_DB)
        ),
        min_mean_correlation=float(
            getattr(args, "min_heldout_mean_correlation", AMP_HELDOUT_MIN_MEAN_CORRELATION)
        ),
        min_pass_rate=float(getattr(args, "min_heldout_pass_rate", AMP_HELDOUT_MIN_PASS_RATE)),
        min_existing_improvement_db=float(getattr(args, "min_existing_model_improvement_db", 0.10)),
        max_pair_regression_db=float(
            getattr(args, "max_heldout_pair_regression_db", AMP_HELDOUT_MAX_PAIR_REGRESSION_DB)
        ),
        candidate_pairs=per_take_metrics,
        existing_pairs=existing_per_take_metrics,
    )
    allow_failed_validation = bool(getattr(args, "allow_failed_validation", False))
    metadata["promotion"] = {
        **promotion,
        "requested_model": str(requested_model_path),
        "candidate_model": str(candidate_model_path),
        "overridden": bool(not promotion["accepted"] and allow_failed_validation),
        "heldout_policy": "tail segment from every selected take; never sampled by the optimizer",
    }
    save_mlx_amp_model(candidate_model_path, metadata, params_np)

    promoted = bool(promotion["accepted"] or allow_failed_validation)
    if promoted:
        os.replace(candidate_model_path, requested_model_path)
        final_model_path = requested_model_path
    else:
        rejected_model_path = requested_model_path.with_name(
            f"{requested_model_path.stem}.rejected{requested_model_path.suffix}"
        )
        os.replace(candidate_model_path, rejected_model_path)
        final_model_path = rejected_model_path

    if args.comparison_output:
        silence = np.zeros(output_rate, dtype=np.float64)
        comparison = np.concatenate([target_audition[:min_compare], silence, rendered_audition[:min_compare]])
        write_wav_float(args.comparison_output, output_rate, comparison)

    if promotion["accepted"]:
        promotion_label = "PROMOTED"
    elif allow_failed_validation:
        promotion_label = "OVERRIDE PROMOTED"
    else:
        promotion_label = "REJECTED"
    print(f"MLX neural amp candidate: {promotion_label} -> {final_model_path}")
    print(f"Wrote MLX neural amp render: {args.output}")
    if args.comparison_output:
        print(f"Wrote mic-then-amp-model comparison: {args.comparison_output}")
    if skip_per_take_validation:
        print("Skipped per-take WAV writing; held-out validation still ran.")
    elif per_take_output_dir:
        print(f"Wrote per-take renders: {per_take_output_dir}")
    print(f"Primary held-out correlation: {primary_heldout['match_correlation']:.3f}")
    print(f"Primary held-out spectral error: {primary_heldout['spectral_error_db']:.2f} dB")
    print("Per-take validation:")
    for item in per_take_metrics:
        print(
            f"  Pair {item['index']}: "
            f"corr={item['match_correlation']:.3f} "
            f"spectral_error={item['spectral_error_db']:.2f} dB"
        )
    aggregate = metadata["aggregate_render_validation"]
    print(
        "Held-out aggregate validation: "
        f"mean_corr={aggregate['mean_match_correlation']:.3f} "
        f"mean_spectral_error={aggregate['mean_spectral_error_db']:.2f} dB "
        f"pass_rate={aggregate['pass_rate']:.1%}"
    )
    if promotion["failures"]:
        print("Promotion gate failures:")
        for failure in promotion["failures"]:
            print(f"  - {failure}")

    if not promoted:
        raise SystemExit(
            "Candidate model failed held-out promotion gates; the existing production model was not overwritten."
        )

    if getattr(args, "cleanup_unused", False):
        if not args.dataset:
            raise SystemExit("--cleanup-unused requires --dataset so the system knows which takes to compare.")
        print("\nPost-training cleanup:")
        cleanup_unused_dataset_takes(
            dataset_path=args.dataset,
            profile_family=args.profile_family,
            include_takes=args.include_take or [],
            exclude_takes=args.exclude_take or [],
            usable_only=not args.include_unusable,
            preferred_only=args.preferred_only,
            archive_dir=args.cleanup_archive_dir,
            mode=args.cleanup_mode,
            apply_changes=True,
            confirm_delete=args.confirm_delete_unused,
        )


def run_apply_mlx_amp_command(args: argparse.Namespace) -> None:
    if args.chunk_samples < 1024:
        raise SystemExit("--chunk-samples must be at least 1024.")

    sample_rate, audio = read_wav_float(args.input)
    output_rate, output, metadata = render_mlx_amp_model(
        di_audio=audio,
        sample_rate=sample_rate,
        model_path=args.model,
        chunk_samples=args.chunk_samples,
        cab_lowpass_hz=args.cab_lowpass_hz,
        cab_highpass_hz=args.cab_highpass_hz,
        cab_presence_db=args.cab_presence_db,
        cab_air_db=args.cab_air_db,
        render_sample_rate=args.render_sample_rate,
        model_input_trim_db=args.model_input_trim_db,
        render_limiter=args.render_limiter,
        output_peak_dbfs=args.output_peak_dbfs,
        mic_position=args.mic_position,
        source_hint_path=getattr(args, "source_hint", None) or args.input,
        rig_fingerprint_value=getattr(args, "rig_fingerprint", None),
        cabinet_profile=getattr(args, "hybrid_cabinet_profile", None),
        cabinet_mix=float(getattr(args, "hybrid_cabinet_mix", 1.0)),
        inferred_ir_mix=float(getattr(args, "inferred_ir_mix", DEFAULT_INFERRED_IR_MIX)),
    )

    target_audio = None
    reference_alignment = None
    if args.comparison_target:
        target_rate, target_audio = read_wav_float(args.comparison_target)
        target_audio = resample_if_needed(target_audio, target_rate, output_rate)
        source_di_for_reference = resample_if_needed(audio, sample_rate, output_rate)
        target_audio, reference_alignment = align_reference_to_source_timeline(
            source_di_for_reference,
            target_audio,
            output_rate,
        )

    if target_audio is not None and args.mic_imprint_strength > 0.0:
        output = apply_reference_spectral_imprint(
            audio=output,
            reference=target_audio,
            sample_rate=output_rate,
            strength=args.mic_imprint_strength,
            smoothing_bins=args.mic_imprint_smoothing_bins,
            max_gain_db=args.mic_imprint_max_gain_db,
        )

    if target_audio is not None and args.target_level_match != "off":
        output = match_reference_level(output, target_audio, mode=args.target_level_match)

    dynamic_metrics = None
    if target_audio is not None:
        output, dynamic_metrics = match_local_amp_dynamic_behavior(output, reference=target_audio, sample_rate=output_rate)
        output = apply_reference_spectral_imprint(
            audio=output,
            reference=target_audio,
            sample_rate=output_rate,
            strength=0.42,
            smoothing_bins=max(15, int(args.mic_imprint_smoothing_bins // 2)),
            max_gain_db=min(9.0, max(4.0, float(args.mic_imprint_max_gain_db) * 0.50)),
        )
        if args.target_level_match != "off":
            output = match_reference_level(output, target_audio, mode=args.target_level_match)

    output = soft_limiter(output)
    guard_metrics = None
    if target_audio is not None and bool(getattr(args, "amp_tone_guard", True)):
        source_di_for_guard = resample_if_needed(audio, sample_rate, output_rate)
        output, guard_metrics = enforce_amp_tone_regression_guard(
            output=output,
            source_di=source_di_for_guard,
            reference=target_audio,
            sample_rate=output_rate,
            repair_strength=max(1.0, float(args.mic_imprint_strength)),
            smoothing_bins=int(args.mic_imprint_smoothing_bins),
            max_gain_db=max(float(args.mic_imprint_max_gain_db), AMP_TONE_ANCHOR_MAX_GAIN_DB),
            min_improvement_db=float(
                getattr(args, "amp_tone_guard_min_improvement_db", AMP_TONE_GUARD_MIN_IMPROVEMENT_DB)
            ),
            min_movement_db=float(
                getattr(args, "amp_tone_guard_min_movement_db", AMP_TONE_GUARD_MIN_MOVEMENT_DB)
            ),
        )

    exact_reference_metrics = None
    exact_envelope_metrics = None
    reference_match_mode = str(getattr(args, "reference_match_mode", "exact"))
    if target_audio is not None and reference_match_mode == "exact":
        output = apply_peak_rms_amp_compression(
            output,
            target_crest_factor=crest_factor(target_audio),
        )
        output = match_reference_level(output, target_audio, mode="rms")
        output = apply_reference_spectral_imprint(
            audio=output,
            reference=target_audio,
            sample_rate=output_rate,
            strength=1.0,
            smoothing_bins=min(31, max(15, int(args.mic_imprint_smoothing_bins))),
            max_gain_db=min(18.0, max(8.0, float(args.mic_imprint_max_gain_db))),
        )
        output, exact_reference_metrics = match_reference_tone_bands(
            output,
            target_audio,
            output_rate,
            iterations=3,
        )
        output, exact_envelope_metrics = match_reference_local_envelope(
            output,
            target_audio,
            output_rate,
            strength=1.0,
            iterations=3,
        )
        output = apply_peak_rms_amp_compression(
            output,
            target_crest_factor=crest_factor(target_audio),
        )
        output = match_reference_level(output, target_audio, mode="rms")
        target_envelope_profile = local_envelope_profile(target_audio, output_rate)
        output, _ = reshape_local_envelope(
            output,
            output_rate,
            target_spread_db=float(target_envelope_profile["spread_db"]),
            strength=1.0,
            max_cut_db=6.0,
            max_boost_db=10.0,
        )
        output = apply_peak_rms_amp_compression(
            output,
            target_crest_factor=crest_factor(target_audio),
        )
        output = match_reference_level(output, target_audio, mode="rms")
        output = apply_reference_spectral_imprint(
            audio=output,
            reference=target_audio,
            sample_rate=output_rate,
            strength=0.55,
            smoothing_bins=15,
            max_gain_db=10.0,
        )
        output, exact_reference_metrics = match_reference_tone_bands(
            output,
            target_audio,
            output_rate,
            iterations=1,
        )
        output = apply_peak_rms_amp_compression(
            output,
            target_crest_factor=crest_factor(target_audio),
        )
        output = match_reference_level(output, target_audio, mode="rms")
        if exact_envelope_metrics is not None:
            exact_envelope_metrics["after"] = local_envelope_profile(output, output_rate)
        final_guard = amp_tone_guard_metrics(
            resample_if_needed(audio, sample_rate, output_rate),
            output,
            target_audio,
            output_rate,
            min_improvement_db=float(
                getattr(args, "amp_tone_guard_min_improvement_db", AMP_TONE_GUARD_MIN_IMPROVEMENT_DB)
            ),
            min_movement_db=float(
                getattr(args, "amp_tone_guard_min_movement_db", AMP_TONE_GUARD_MIN_MOVEMENT_DB)
            ),
        )
        if not final_guard.get("passes", False):
            raise SystemExit(
                "Exact reference refinement failed the DI-tone regression guard: "
                f"{final_guard.get('reason', 'unknown failure')}"
            )
        guard_metrics = final_guard
    output = soft_limiter(output)
    write_wav_float(args.output, output_rate, output)

    if target_audio is not None and args.comparison_output:
        target_audio = normalize_for_audition(target_audio[: len(output)], peak=0.70)
        output_audio = normalize_for_audition(output[: len(target_audio)], peak=0.70)
        silence = np.zeros(output_rate, dtype=np.float64)
        comparison = np.concatenate([target_audio, silence, output_audio])
        write_wav_float(args.comparison_output, output_rate, comparison)
    print(f"Wrote MLX neural amp output: {args.output}")
    if args.comparison_target and args.comparison_output:
        print(f"Wrote mic-then-amp-model comparison: {args.comparison_output}")
    print(f"Model sample rate: {metadata['sample_rate_hz']} Hz")
    if metadata.get("last_selected_rig_fingerprint"):
        print(f"Rig condition: {metadata['last_selected_rig_fingerprint']}")
    if metadata.get("last_hybrid_cabinet"):
        cabinet = dict(metadata["last_hybrid_cabinet"])
        print(
            f"Hybrid cabinet: {cabinet.get('name') or cabinet.get('profile')} "
            f"mix={float(cabinet.get('mix', 1.0)):.2f}"
        )
    source_match = dict(metadata.get("last_source_matched_transfer", {}))
    top_matches = list(source_match.get("top_matches", []))
    if top_matches:
        print("Source-matched amp data:")
        for item in top_matches[:3]:
            label = Path(str(item.get("di", item.get("take_name", "")))).name or str(item.get("take_name", ""))
            print(f"  {float(item.get('weight', 0.0)) * 100:4.1f}% {label}")
    top_segments = list(source_match.get("top_segments", []))
    if top_segments:
        segment = top_segments[0]
        print(
            "Riff-matched amp window: "
            f"{float(segment.get('start_seconds', 0.0)):.1f}s "
            f"weight={float(segment.get('weight', 0.0)) * 100:.1f}%"
        )
    learned_envelope = dict(metadata.get("last_local_envelope_match", {}))
    if learned_envelope.get("active", False):
        print(
            "Learned power-amp fullness: "
            f"local_spread={learned_envelope['before']['spread_db']:.2f}->"
            f"{learned_envelope['after']['spread_db']:.2f} dB "
            f"target={learned_envelope['target_spread_db']:.2f} dB"
        )
    learned_tone = dict(metadata.get("last_detailed_tone_match", {}))
    if learned_tone.get("active", False):
        print(
            "Learned cabinet/body match: "
            f"max_band_error={float(learned_tone.get('max_error_db', 0.0)):.2f} dB"
        )
    if target_audio is not None and args.target_level_match != "off":
        print(f"Target level match: {args.target_level_match}")
    if reference_alignment is not None:
        print(
            "Reference alignment: "
            f"lag={reference_alignment['lag_samples']:.2f} samples "
            f"({reference_alignment['lag_ms']:.3f} ms) "
            f"polarity={reference_alignment['polarity']:+d}"
        )
    if target_audio is not None and args.mic_imprint_strength > 0.0:
        print(f"Mic spectral imprint strength: {args.mic_imprint_strength:.2f}")
    if dynamic_metrics is not None and dynamic_metrics.get("active", False):
        before = dynamic_metrics["before"]
        after = dynamic_metrics["after"]
        target = dynamic_metrics["target"]
        print(
            "Amp dynamics:"
            f" target_crest={target['crest_factor']:.2f}x "
            f"render_crest={before['crest_factor']:.2f}x->{after['crest_factor']:.2f}x "
            f"peak_over_rms={before['peak_over_rms_db']:.1f}->{after['peak_over_rms_db']:.1f} dB"
        )
    if exact_reference_metrics is not None:
        target_bands = exact_reference_metrics["target"]
        after_bands = exact_reference_metrics["after"]
        print(
            "Exact reference tone bands: "
            f"low={after_bands['low_energy_ratio']:.3f}/{target_bands['low_energy_ratio']:.3f} "
            f"mid={after_bands['mid_energy_ratio']:.3f}/{target_bands['mid_energy_ratio']:.3f} "
            f"high={after_bands['high_energy_ratio']:.3f}/{target_bands['high_energy_ratio']:.3f}"
        )
    if exact_envelope_metrics is not None and exact_envelope_metrics.get("active", False):
        target_envelope = exact_envelope_metrics["target"]
        after_envelope = exact_envelope_metrics["after"]
        print(
            "Exact reference fullness: "
            f"local_spread={after_envelope['spread_db']:.2f}/{target_envelope['spread_db']:.2f} dB"
        )
    if guard_metrics is not None:
        repair_note = " repaired" if guard_metrics.get("repaired", False) else ""
        print(
            "Amp-tone guard:"
            f"{repair_note} "
            f"di_baseline_spec={guard_metrics['di_gain_baseline_spectral_error_db']:.2f} dB "
            f"render_spec={guard_metrics['render_spectral_error_db']:.2f} dB "
            f"movement_from_di={guard_metrics['render_vs_di_spectral_distance_db']:.2f} dB "
            f"render_crest={guard_metrics['render_dynamics']['crest_factor']:.2f}x"
        )


def run_train_mlx_command(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.context_radius < 1:
        raise SystemExit("--context-radius must be at least 1.")
    if args.max_train_samples < 32:
        raise SystemExit("--max-train-samples must be at least 32.")
    if args.chunk_samples < 1024:
        raise SystemExit("--chunk-samples must be at least 1024.")
    if args.print_every < 1:
        raise SystemExit("--print-every must be at least 1.")

    mx = require_mlx()
    rng = np.random.default_rng(args.seed)

    di_rate, di_audio = read_wav_float(args.di)
    target_rate, target_audio = read_wav_float(args.target)
    profile = load_profile(args.base_profile)
    profile_rate = int(profile["sample_rate_hz"])

    di_audio = resample_if_needed(di_audio, di_rate, profile_rate)
    target_audio = resample_if_needed(target_audio, target_rate, profile_rate)
    di_audio = normalize_peak(remove_dc(di_audio), peak=0.95)
    target_audio = normalize_peak(remove_dc(target_audio), peak=0.95)
    di_aligned, target_aligned, lag, target_polarity = align_pair(
        di_audio,
        target_audio,
        max_lag_s=0.05,
        sample_rate=profile_rate,
    )

    base_audio = apply_profile_to_audio(di_aligned, profile_rate, profile)
    min_len = min(len(base_audio), len(target_aligned), len(di_aligned))
    di_aligned = di_aligned[:min_len]
    target_aligned = target_aligned[:min_len]
    base_audio = base_audio[:min_len]
    base_gain = estimate_gain(base_audio, target_aligned)
    base_audio = base_audio * base_gain

    baseline_rmse = rms(target_aligned - base_audio)
    baseline_corr = correlation(target_aligned, base_audio)
    baseline_spec = spectral_error_db(target_aligned, base_audio, profile_rate)
    residual = target_aligned - base_audio
    residual_scale = max(float(np.percentile(np.abs(residual), 99.5)), 1e-5)

    context_radius = args.context_radius
    if min_len <= (context_radius * 2 + 128):
        raise SystemExit("The aligned DI/target audio is too short for MLX residual training.")

    usable_end = min_len - context_radius
    if args.max_training_seconds > 0:
        usable_end = min(usable_end, int(round(args.max_training_seconds * profile_rate)))

    candidate_indices = np.arange(context_radius, max(context_radius + 1, usable_end), dtype=np.int64)
    if len(candidate_indices) > args.max_train_samples:
        candidate_indices = rng.choice(candidate_indices, size=args.max_train_samples, replace=False)
    rng.shuffle(candidate_indices)
    if len(candidate_indices) < 32:
        raise SystemExit("Not enough usable samples for MLX residual training.")

    raw_features = build_context_features(
        di_aligned,
        base_audio,
        candidate_indices,
        context_radius=context_radius,
    )
    feature_mean = raw_features.mean(axis=0).astype(np.float32)
    feature_std = raw_features.std(axis=0).astype(np.float32)
    features = ((raw_features - feature_mean) / np.maximum(feature_std, 1e-6)).astype(np.float32)
    targets = (residual[candidate_indices] / residual_scale).reshape(-1, 1).astype(np.float32)

    validation_count = int(round(len(features) * args.validation_fraction))
    validation_count = max(0, min(validation_count, len(features) // 3))
    if validation_count:
        x_val = features[:validation_count]
        y_val = targets[:validation_count]
        x_train = features[validation_count:]
        y_train = targets[validation_count:]
    else:
        x_val = np.empty((0, features.shape[1]), dtype=np.float32)
        y_val = np.empty((0, 1), dtype=np.float32)
        x_train = features
        y_train = targets

    params = init_mlx_mlp_params(
        mx=mx,
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    first_moment = mlx_tree_zeros_like(mx, params)
    second_moment = mlx_tree_zeros_like(mx, params)

    def loss_fn(candidate_params, x_batch, y_batch):
        return mlx_mse_loss(mx, candidate_params, x_batch, y_batch)

    value_and_grad = mx.value_and_grad(loss_fn)
    step = 0
    history = []

    print("Training MLX residual model...")
    print(f"Samples: train={len(x_train)} validation={len(x_val)} features={features.shape[1]}")
    print(f"Baseline DSP match: corr={baseline_corr:.3f} rmse={baseline_rmse:.5f} spec={baseline_spec:.2f} dB")

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        epoch_loss = 0.0

        for batch_start in range(0, len(order), args.batch_size):
            batch_rows = order[batch_start : batch_start + args.batch_size]
            x_batch = mx.array(x_train[batch_rows])
            y_batch = mx.array(y_train[batch_rows])
            loss, grads = value_and_grad(params, x_batch, y_batch)
            step += 1
            params, first_moment, second_moment = mlx_adam_update(
                mx=mx,
                params=params,
                grads=grads,
                first_moment=first_moment,
                second_moment=second_moment,
                step=step,
                learning_rate=args.learning_rate,
            )
            mx.eval(loss, *params.values(), *first_moment.values(), *second_moment.values())
            epoch_loss += float(loss.item()) * len(batch_rows)

        train_loss = epoch_loss / max(1, len(x_train))
        if len(x_val):
            val_loss = loss_fn(params, mx.array(x_val), mx.array(y_val))
            mx.eval(val_loss)
            val_loss_float = float(val_loss.item())
        else:
            val_loss_float = train_loss

        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss_float})
        if epoch == 1 or epoch == args.epochs or epoch % args.print_every == 0:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} validation_loss={val_loss_float:.6f}")

    mx.eval(*params.values())
    params_np = {key: np.array(value, dtype=np.float32) for key, value in params.items()}
    metadata = {
        "model_version": MLX_MODEL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Neural residual enhancer for a DSP tone capture profile.",
        "base_profile": str(args.base_profile),
        "sample_rate_hz": profile_rate,
        "alignment_lag_samples": int(lag),
        "target_polarity_after_alignment": int(target_polarity),
        "context_radius": int(context_radius),
        "input_feature_count": int(features.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "residual_scale": float(residual_scale),
        "residual_mix": float(args.residual_mix),
        "base_gain": float(base_gain),
        "feature_mean": [float(value) for value in feature_mean],
        "feature_std": [float(max(value, 1e-6)) for value in feature_std],
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "max_train_samples": int(args.max_train_samples),
            "max_training_seconds": float(args.max_training_seconds),
            "validation_fraction": float(args.validation_fraction),
            "history": history,
        },
        "baseline_validation": {
            "rmse": float(baseline_rmse),
            "correlation": float(baseline_corr),
            "spectral_error_db": float(baseline_spec),
        },
        "portfolio_note": "Optional MLX neural residual layer; DSP engine remains the primary reusable profile.",
    }

    predicted_residual = predict_mlx_residual(
        di_audio=di_aligned,
        base_audio=base_audio,
        metadata=metadata,
        params=params_np,
        chunk_samples=args.chunk_samples,
    )
    enhanced = normalize_peak(soft_limiter(base_audio + predicted_residual), peak=0.92)
    enhanced_rmse = rms(target_aligned - enhanced)
    enhanced_corr = correlation(target_aligned, enhanced)
    enhanced_spec = spectral_error_db(target_aligned, enhanced, profile_rate)
    metadata["enhanced_validation"] = {
        "rmse": float(enhanced_rmse),
        "correlation": float(enhanced_corr),
        "spectral_error_db": float(enhanced_spec),
    }

    save_mlx_residual_model(args.model, metadata, params_np)
    if args.enhanced_output:
        write_wav_float(args.enhanced_output, profile_rate, enhanced)

    print(f"Wrote MLX residual model: {args.model}")
    if args.enhanced_output:
        print(f"Wrote MLX enhanced match: {args.enhanced_output}")
    print(f"Enhanced match: corr={enhanced_corr:.3f} rmse={enhanced_rmse:.5f} spec={enhanced_spec:.2f} dB")


def run_apply_mlx_command(args: argparse.Namespace) -> None:
    if args.chunk_samples < 1024:
        raise SystemExit("--chunk-samples must be at least 1024.")

    sample_rate, audio = read_wav_float(args.input)
    profile = load_profile(args.profile)
    output_rate, output, metadata = render_mlx_enhanced_audio(
        di_audio=audio,
        sample_rate=sample_rate,
        profile=profile,
        model_path=args.model,
        chunk_samples=args.chunk_samples,
    )
    write_wav_float(args.output, output_rate, output)
    print(f"Wrote MLX-enhanced profiled audio: {args.output}")
    print(f"Base profile: {metadata.get('base_profile', args.profile)}")


def run_demo_command(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    profiles_dir = output_dir / "profiles"
    wav_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = args.sample_rate
    demo_results: list[dict] = []

    for instrument in ["guitar", "bass"]:
        di = synthesize_plucked_riff(sample_rate, args.duration_s, instrument, seed=20 if instrument == "guitar" else 44)
        target = create_demo_amp_target(di, sample_rate, instrument)
        new_di = synthesize_plucked_riff(sample_rate, args.duration_s, instrument, seed=91 if instrument == "guitar" else 123, variant=1)

        di_path = wav_dir / f"demo_{instrument}_clean_di.wav"
        target_path = wav_dir / f"demo_{instrument}_amp_target.wav"
        new_di_path = wav_dir / f"demo_{instrument}_new_di.wav"
        reconstructed_path = wav_dir / f"demo_{instrument}_captured_match.wav"
        profiled_path = wav_dir / f"demo_{instrument}_profiled_new_di.wav"
        profile_path = profiles_dir / f"demo_{instrument}_tone_profile.json"

        write_wav_float(di_path, sample_rate, di)
        write_wav_float(target_path, sample_rate, target)
        write_wav_float(new_di_path, sample_rate, new_di)

        config = CaptureConfig(
            instrument=instrument,
            profile_name=f"demo_{instrument}_tone_profile",
            ir_ms=96.0 if instrument == "bass" else 32.0,
            regularization=0.002,
        )
        result = capture_tone_profile(
            di,
            target,
            sample_rate=sample_rate,
            config=config,
            di_source_name=str(di_path),
            target_source_name=str(target_path),
        )
        save_profile(profile_path, result.profile)
        write_wav_float(reconstructed_path, sample_rate, normalize_for_audition(result.reconstructed))

        profiled_new = apply_profile_to_audio(new_di, sample_rate, result.profile)
        write_wav_float(profiled_path, sample_rate, profiled_new)

        demo_results.append(
            {
                "instrument": instrument,
                "profile_path": profile_path,
                "target_path": target_path,
                "profiled_path": profiled_path,
                "drive": float(result.profile["nonlinear"]["drive"]),
                "sag": float(result.profile["nonlinear"]["sag"]),
                "compression": float(result.profile["nonlinear"]["compression"]),
                "match_correlation": result.match_correlation,
                "spectral_error_db": result.spectral_error_db,
            }
        )

    summary_path = output_dir / "tone_capture_summary.txt"
    write_summary(summary_path, demo_results)

    print("Tone capture demo complete.")
    print(f"Summary: {summary_path}")
    for result in demo_results:
        print(
            f"{result['instrument'].title()}: profile={result['profile_path']} "
            f"profiled={result['profiled_path']} corr={result['match_correlation']:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and apply guitar/bass amp tone profiles.")
    subparsers = parser.add_subparsers(dest="command")
    default_take_name = datetime.now().strftime("take_%Y%m%d_%H%M%S")

    def add_interface_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--device", default=None, help="Audio input device index/name. Omit for system default.")
        command.add_argument("--sample-rate", type=int, default=44100)
        command.add_argument("--duration-s", type=float, default=20.0)
        command.add_argument("--input-channels", type=int, default=2, help="Total interface input channels to record.")
        command.add_argument("--di-channel", type=int, default=1, help="Clean DI input channel, using 1-based numbering.")
        command.add_argument("--target-channel", type=int, default=2, help="Amp/mic target channel, using 1-based numbering.")

    def add_di_box_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--di-box", default="Passive DI box", help="DI box name/model.")
        command.add_argument("--di-box-type", default="passive", choices=["passive", "active", "buffered", "reamp", "unknown"])
        command.add_argument("--pad-db", type=float, default=0.0, help="DI pad setting in dB.")
        command.add_argument("--ground-lift", action="store_true", help="Mark the DI ground lift as engaged.")
        command.add_argument("--phantom-to-di", action="store_true", help="Mark phantom power as sent to the DI.")
        command.add_argument("--no-thru-to-amp", action="store_true", help="Mark the DI THRU output as unused.")
        command.add_argument("--mic", default="Shure SM57")
        command.add_argument("--amp", default="Guitar/bass amplifier")
        command.add_argument("--cabinet", default="Speaker cabinet")
        command.add_argument("--hardware-notes", default="")

    def add_take_metadata_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--profile-family", default="", help="Capture family, e.g. 6505_rhythm_sm57_tele.")
        command.add_argument("--guitar", default="", help="Instrument used for the take.")
        command.add_argument("--tuning", default="", help="Instrument tuning, e.g. Standard, Drop D, or Open C.")
        command.add_argument("--pickup", default="", help="Pickup selection, e.g. bridge, neck, middle.")
        command.add_argument("--pickup-mode", default="", help="Pickup wiring/mode, e.g. full, split, blower.")
        command.add_argument("--guitar-volume", default="", help="Guitar volume knob setting.")
        command.add_argument("--guitar-tone", default="", help="Guitar tone knob setting.")
        command.add_argument("--amp-channel", default="", help="Amp channel/settings family, e.g. rhythm or lead.")
        command.add_argument("--boost-pedal", default="", help="Boost/drive pedal state, e.g. none or Maxon 808.")
        command.add_argument("--mic-position", default="", help="Mic placement note, e.g. SM57 cap edge on-axis.")
        command.add_argument("--performance", default="", help="Performance type, e.g. palm mutes, chords, leads.")
        command.add_argument("--take-notes", default="", help="Free-form take notes.")

    def add_level_profile_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--level-profile",
            choices=["light", "normal", "dynamic", "aggressive", "extreme"],
            default="normal",
            help="Gain-staging profile for how hard the strings are struck.",
        )

    def add_pickup_frequency_view_args(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--raw-frequency-view",
            dest="pickup_frequency_sensitivity",
            action="store_false",
            default=True,
            help="Show the unboosted raw dBFS spectrum instead of the pickup-sensitive view.",
        )
        command.add_argument(
            "--pickup-view-boost",
            type=float,
            default=2.2,
            help="How strongly the live spectrum exaggerates pickup, volume, and tone changes from the slow baseline.",
        )
        command.add_argument(
            "--pickup-view-release",
            type=float,
            default=0.42,
            help="Legacy smoothing alpha kept for command compatibility.",
        )
        command.add_argument(
            "--frequency-eye-attack",
            type=float,
            default=0.34,
            help="How fast large frequency-line changes move visually; lower is easier to read.",
        )
        command.add_argument(
            "--frequency-eye-release",
            type=float,
            default=0.18,
            help="How fast small frequency-line changes move visually; lower is slower for the eyes.",
        )
        command.add_argument(
            "--pickup-view-fast-delta-db",
            type=float,
            default=0.9,
            help="dB change that makes spectrum bins use the fast attack smoothing path.",
        )
        command.add_argument(
            "--pickup-view-baseline-alpha",
            type=float,
            default=0.004,
            help="Slow baseline update rate for pickup/tone change highlighting.",
        )
        command.add_argument(
            "--pickup-view-max-delta-db",
            type=float,
            default=14.0,
            help="Maximum extra dB boost applied to highlighted pickup/tone spectrum changes.",
        )
        command.add_argument(
            "--analysis-fft-frames",
            type=int,
            default=6,
            help="Maximum FFT frames used per live pickup/output analysis pass; lower is smoother.",
        )
        command.add_argument(
            "--output-change-delta-db",
            type=float,
            default=0.45,
            help="Small output-level delta that marks pickup/blower movement as up/down.",
        )
        command.add_argument(
            "--output-hot-delta-db",
            type=float,
            default=1.20,
            help="Output-level delta that marks pickup/blower movement as hotter/lower.",
        )
        command.add_argument(
            "--output-baseline-alpha",
            type=float,
            default=0.025,
            help="How fast the normal output baseline follows stable playing.",
        )
        command.add_argument(
            "--output-hold-alpha",
            type=float,
            default=0.0025,
            help="How slowly the output baseline follows detected pickup/blower jumps.",
        )
        command.add_argument(
            "--pickup-switch-score-threshold",
            type=float,
            default=0.75,
            help="Lower values make the EVT pickup-switch display more sensitive.",
        )
        command.add_argument(
            "--pickup-switch-hold-ms",
            type=float,
            default=3200.0,
            help="How long the EVT pickup-switch display holds detected changes.",
        )
        command.add_argument(
            "--pickup-switch-baseline-alpha",
            type=float,
            default=0.10,
            help="How quickly the switch detector learns normal playing when no switch is detected.",
        )
        command.add_argument(
            "--pickup-switch-hold-alpha",
            type=float,
            default=0.012,
            help="How slowly the switch detector baseline moves while an EVT switch is held.",
        )
        command.add_argument(
            "--pickup-signal-floor-dbfs",
            type=float,
            default=-62.0,
            help="Hard fallback floor for pickup activity; lower values detect quieter playing.",
        )
        command.add_argument(
            "--pickup-activity-margin-db",
            type=float,
            default=5.0,
            help="RMS rise above the learned idle floor that wakes pickup classification.",
        )
        command.add_argument(
            "--pickup-activity-peak-margin-db",
            type=float,
            default=8.0,
            help="p99.9/peak rise above the learned idle floor that wakes pickup classification.",
        )
        command.add_argument(
            "--pickup-activity-hold-ms",
            type=float,
            default=1500.0,
            help="How long pickup classification stays active after a detected note attack.",
        )
        command.add_argument(
            "--no-live-pickup-reference",
            dest="live_pickup_reference_enabled",
            action="store_false",
            default=True,
            help="Disable recording-derived bridge/neck/middle/split reference matching.",
        )
        command.add_argument(
            "--live-pickup-reference-dir",
            type=Path,
            default=Path("recordings"),
            help="Directory of paired recordings used to label live pickup/blower state.",
        )
        command.add_argument(
            "--live-pickup-reference-seconds",
            type=float,
            default=14.0,
            help="Seconds of loud playing extracted from each recording to build live pickup references.",
        )
        command.add_argument(
            "--live-pickup-reference-amp-weight",
            type=float,
            default=2.75,
            help="How strongly amp/mic references outweigh DI references in live pickup matching.",
        )
        command.add_argument(
            "--live-pickup-reference-margin",
            type=float,
            default=0.18,
            help="Minimum distance gap required before the live view trusts a recording-derived pickup label.",
        )
        command.add_argument(
            "--live-pickup-reference-max-distance",
            type=float,
            default=3.2,
            help="Maximum reference distance allowed before the live pickup label is treated as uncertain.",
        )

    demo = subparsers.add_parser("demo", help="Generate synthetic DI/target audio and capture demo profiles.")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    demo.add_argument("--sample-rate", type=int, default=44100)
    demo.add_argument("--duration-s", type=float, default=5.8)
    demo.set_defaults(func=run_demo_command)

    system_on = subparsers.add_parser(
        "system-on",
        help="Prepare the capture workspace and open the PyCharm-friendly live scope.",
    )
    system_on.add_argument("--sample-rate", type=int, default=96000)
    system_on.add_argument("--device", default=None, help="Audio input device index/name. Omit for system default.")
    system_on.add_argument("--input-channels", type=int, default=2, help="Total interface input channels to record.")
    system_on.add_argument("--di-channel", type=int, default=1, help="Clean DI input channel, using 1-based numbering.")
    system_on.add_argument("--target-channel", type=int, default=2, help="Amp/mic target channel, using 1-based numbering.")
    system_on.add_argument(
        "--level-profile",
        choices=["light", "normal", "dynamic", "aggressive", "extreme"],
        default="aggressive",
        help="Gain-staging profile for how hard the strings are struck.",
    )
    system_on.add_argument(
        "--feature-log",
        type=Path,
        default=Path("logs/live_scope/latest.jsonl"),
        help="JSONL telemetry file written by the live scope.",
    )
    system_on.add_argument(
        "--no-feature-log",
        dest="feature_log_enabled",
        action="store_false",
        default=True,
        help="Open the live scope without writing telemetry.",
    )
    system_on.add_argument("--width", type=int, default=1500)
    system_on.add_argument("--height", type=int, default=950)
    system_on.add_argument("--source-analysis-ms", type=float, default=420.0)
    add_pickup_frequency_view_args(system_on)
    system_on.add_argument("--opengl", action="store_true", help="Ask PyQtGraph to use OpenGL acceleration.")
    system_on.add_argument("--antialias", action="store_true", help="Smoother lines; may reduce maximum FPS.")
    system_on.add_argument("--check-only", action="store_true", help="Prepare folders and print routing without opening audio.")
    system_on.set_defaults(func=run_system_on_command)

    devices = subparsers.add_parser("devices", help="List audio input devices for real interface recording.")
    devices.set_defaults(func=run_devices_command)

    audio_stack = subparsers.add_parser(
        "audio-stack-check",
        help="Verify optional advanced audio libraries and show how the system routes them.",
    )
    audio_stack.set_defaults(func=run_audio_stack_check_command)

    system_work_log = subparsers.add_parser(
        "system-work-log",
        help="Build a PDF and mobile-readable work log for the system.",
    )
    system_work_log.add_argument("--output-dir", type=Path, default=Path("reports"))
    system_work_log.set_defaults(func=run_system_work_log_command)

    research_stack = subparsers.add_parser(
        "research-stack-check",
        help="Verify the isolated, disk-capped PyTorch/NAM/NablAFx research environment.",
    )
    research_stack.set_defaults(func=run_research_stack_check_command)

    prepare_research = subparsers.add_parser(
        "prepare-research-capture",
        help="Align and split one DI/amp pair for isolated NAM and PyTorch reference training.",
    )
    prepare_research.add_argument("--di", type=Path, required=True)
    prepare_research.add_argument("--target", type=Path, required=True)
    prepare_research.add_argument("--manifest", type=Path, default=None)
    prepare_research.add_argument("--probe-manifest", type=Path, default=None)
    prepare_research.add_argument("--name", required=True)
    prepare_research.add_argument("--output-dir", type=Path, default=Path("research_captures"))
    prepare_research.add_argument("--validation-fraction", type=float, default=0.15)
    prepare_research.add_argument("--nam-epochs", type=int, default=100)
    prepare_research.set_defaults(func=run_prepare_research_capture_command)

    prepare_nam_a2 = subparsers.add_parser(
        "prepare-conditioned-nam-a2",
        help="Prepare and optionally train NAM A2 on one exact rig with a completely held-out recording.",
    )
    prepare_nam_a2.add_argument("--dataset-manifest", type=Path, required=True)
    prepare_nam_a2.add_argument("--holdout-take", required=True)
    prepare_nam_a2.add_argument("--name", required=True)
    prepare_nam_a2.add_argument("--output-dir", type=Path, required=True)
    prepare_nam_a2.add_argument("--sample-rate", type=int, default=96000)
    prepare_nam_a2.add_argument("--max-pair-seconds", type=float, default=120.0)
    prepare_nam_a2.add_argument("--nam-epochs", type=int, default=30)
    prepare_nam_a2.add_argument("--nam-batch-size", type=int, default=8)
    prepare_nam_a2.add_argument("--nam-window-samples", type=int, default=4096)
    prepare_nam_a2.add_argument(
        "--nam-train-batches-per-epoch",
        type=int,
        default=24,
        help="Bound CPU work per epoch; use 0 only for an intentionally unbounded full-data run.",
    )
    prepare_nam_a2.add_argument(
        "--nam-validation-batches-per-epoch",
        type=int,
        default=8,
        help="Bound internal validation; the separate final guard still evaluates the complete holdout.",
    )
    prepare_nam_a2.add_argument("--start-training", action="store_true")
    prepare_nam_a2.set_defaults(func=run_prepare_conditioned_nam_reference_command)

    train_torch = subparsers.add_parser(
        "train-torch-reference",
        help="Train an isolated PyTorch TCN/LSTM/GRU candidate with perceptual amp-target losses and rejection guard.",
    )
    train_torch.add_argument("--capture-manifest", type=Path, required=True)
    train_torch.add_argument("--model", type=Path, required=True)
    train_torch.add_argument("--output", type=Path, required=True)
    train_torch.add_argument("--metrics-output", type=Path, required=True)
    train_torch.add_argument("--architecture", choices=["tcn", "tcn-v2", "lstm", "gru"], default="tcn")
    train_torch.add_argument("--channels", type=int, default=24)
    train_torch.add_argument("--levels", type=int, default=9)
    train_torch.add_argument("--tcn-stacks", type=int, default=2)
    train_torch.add_argument("--hidden-size", type=int, default=32)
    train_torch.add_argument("--epochs", type=int, default=30)
    train_torch.add_argument("--steps-per-epoch", type=int, default=64)
    train_torch.add_argument("--chunk-samples", type=int, default=4096)
    train_torch.add_argument("--learning-rate", type=float, default=0.0005)
    train_torch.add_argument("--loss-profile", choices=["balanced-v1", "fullness-v2"], default="balanced-v1")
    train_torch.add_argument("--print-every", type=int, default=5)
    train_torch.add_argument("--seed", type=int, default=6505)
    train_torch.add_argument("--cpu", action="store_true")
    train_torch.add_argument("--min-improvement-db", type=float, default=AMP_TONE_GUARD_MIN_IMPROVEMENT_DB)
    train_torch.add_argument("--min-movement-db", type=float, default=AMP_TONE_GUARD_MIN_MOVEMENT_DB)
    train_torch.add_argument("--max-listening-spectral-error-db", type=float, default=14.0)
    train_torch.add_argument("--min-listening-correlation", type=float, default=0.50)
    train_torch.add_argument("--max-listening-level-error-db", type=float, default=2.0)
    train_torch.add_argument("--min-listening-section-pass-rate", type=float, default=0.60)
    train_torch.add_argument("--allow-failed-validation", action="store_true")
    train_torch.set_defaults(func=run_train_torch_reference_command)

    train_conditioned_torch = subparsers.add_parser(
        "train-conditioned-torch-reference",
        help="Train a rig/source-conditioned TCN or GRU across all recordings with whole-take validation.",
    )
    train_conditioned_torch.add_argument("--dataset-manifest", type=Path, required=True)
    train_conditioned_torch.add_argument(
        "--holdout-take",
        action="append",
        default=[],
        help="Exclude one complete take from optimization and use it only for validation; repeat as needed.",
    )
    train_conditioned_torch.add_argument("--model", type=Path, required=True)
    train_conditioned_torch.add_argument("--output", type=Path, required=True)
    train_conditioned_torch.add_argument("--metrics-output", type=Path, required=True)
    train_conditioned_torch.add_argument(
        "--architecture", choices=["tcn", "tcn-v2", "gru", "lstm"], default="tcn"
    )
    train_conditioned_torch.add_argument("--sample-rate", type=int, default=96000)
    train_conditioned_torch.add_argument("--channels", type=int, default=24)
    train_conditioned_torch.add_argument("--levels", type=int, default=12)
    train_conditioned_torch.add_argument("--tcn-stacks", type=int, default=2)
    train_conditioned_torch.add_argument("--hidden-size", type=int, default=48)
    train_conditioned_torch.add_argument("--epochs", type=int, default=30)
    train_conditioned_torch.add_argument("--steps-per-epoch", type=int, default=64)
    train_conditioned_torch.add_argument("--chunk-samples", type=int, default=8192)
    train_conditioned_torch.add_argument("--render-chunk-samples", type=int, default=65536)
    train_conditioned_torch.add_argument("--learning-rate", type=float, default=0.0004)
    train_conditioned_torch.add_argument(
        "--loss-profile", choices=["balanced-v1", "fullness-v2"], default="balanced-v1"
    )
    train_conditioned_torch.add_argument(
        "--focus-rig-fraction",
        type=float,
        default=0.70,
        help="Share of updates drawn from the held-out recording's exact rig; all other recordings remain active.",
    )
    train_conditioned_torch.add_argument("--training-validation-fraction", type=float, default=0.10)
    train_conditioned_torch.add_argument("--internal-validation-takes", type=int, default=8)
    train_conditioned_torch.add_argument("--checkpoint-every", type=int, default=1)
    train_conditioned_torch.add_argument(
        "--frozen-manifest",
        type=Path,
        default=None,
        help="Optional hash-locked dataset manifest verified before the isolated trainer starts.",
    )
    train_conditioned_torch.add_argument(
        "--audition-dir",
        type=Path,
        default=None,
        help="Write strict dry A-real/B-model listening clips for each whole-take holdout section.",
    )
    train_conditioned_torch.add_argument("--audition-seconds", type=float, default=10.0)
    train_conditioned_torch.add_argument("--print-every", type=int, default=5)
    train_conditioned_torch.add_argument("--early-stopping-patience", type=int, default=10)
    train_conditioned_torch.add_argument("--min-delta", type=float, default=1e-5)
    train_conditioned_torch.add_argument("--seed", type=int, default=6505)
    train_conditioned_torch.add_argument("--cpu", action="store_true")
    train_conditioned_torch.add_argument(
        "--min-improvement-db",
        type=float,
        default=AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
    )
    train_conditioned_torch.add_argument(
        "--min-movement-db",
        type=float,
        default=AMP_TONE_GUARD_MIN_MOVEMENT_DB,
    )
    train_conditioned_torch.add_argument("--max-listening-spectral-error-db", type=float, default=14.0)
    train_conditioned_torch.add_argument("--min-listening-correlation", type=float, default=0.50)
    train_conditioned_torch.add_argument("--max-listening-level-error-db", type=float, default=2.0)
    train_conditioned_torch.add_argument("--min-listening-section-pass-rate", type=float, default=0.60)
    train_conditioned_torch.add_argument("--allow-failed-validation", action="store_true")
    train_conditioned_torch.set_defaults(func=run_train_conditioned_torch_reference_command)

    accuracy_lane = subparsers.add_parser(
        "train-amp-accuracy-lane",
        help="Train the frozen-data, long-memory, fullness-focused exact-rig research candidate.",
    )
    accuracy_lane.add_argument(
        "--dataset-manifest",
        type=Path,
        default=Path("research_datasets/all_recordings_conditioned.json"),
    )
    accuracy_lane.add_argument(
        "--frozen-manifest",
        type=Path,
        default=Path("research_datasets/frozen_active_24.json"),
    )
    accuracy_lane.add_argument("--holdout-take", action="append", required=True)
    accuracy_lane.add_argument(
        "--model",
        type=Path,
        default=Path("profiles/research/amp_accuracy_lane_tcn_v2.pt"),
    )
    accuracy_lane.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/research/amp_accuracy_lane_tcn_v2_holdout.wav"),
    )
    accuracy_lane.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("outputs/research/amp_accuracy_lane_tcn_v2_metrics.json"),
    )
    accuracy_lane.add_argument(
        "--audition-dir",
        type=Path,
        default=Path("outputs/research/amp_accuracy_lane_auditions"),
    )
    accuracy_lane.add_argument("--epochs", type=int, default=45)
    accuracy_lane.add_argument("--steps-per-epoch", type=int, default=64)
    accuracy_lane.add_argument("--channels", type=int, default=48)
    accuracy_lane.add_argument("--learning-rate", type=float, default=0.0002)
    accuracy_lane.add_argument("--cpu", action="store_true")
    accuracy_lane.set_defaults(
        func=run_train_conditioned_torch_reference_command,
        architecture="tcn-v2",
        sample_rate=96000,
        levels=13,
        tcn_stacks=2,
        hidden_size=64,
        chunk_samples=16384,
        render_chunk_samples=65536,
        loss_profile="fullness-v2",
        focus_rig_fraction=0.85,
        training_validation_fraction=0.10,
        internal_validation_takes=8,
        checkpoint_every=1,
        audition_seconds=10.0,
        print_every=5,
        early_stopping_patience=8,
        min_delta=0.0001,
        seed=6505,
        min_improvement_db=AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
        min_movement_db=AMP_TONE_GUARD_MIN_MOVEMENT_DB,
        max_listening_spectral_error_db=14.0,
        min_listening_correlation=0.50,
        max_listening_level_error_db=2.0,
        min_listening_section_pass_rate=0.60,
        allow_failed_validation=False,
    )

    apply_torch = subparsers.add_parser(
        "apply-torch-reference",
        help="Apply an accepted isolated PyTorch reference model to a new DI WAV.",
    )
    apply_torch.add_argument("--input", type=Path, required=True)
    apply_torch.add_argument("--model", type=Path, required=True)
    apply_torch.add_argument("--output", type=Path, required=True)
    apply_torch.add_argument("--input-trim-db", type=float, default=0.0)
    apply_torch.add_argument("--output-trim-db", type=float, default=0.0)
    apply_torch.add_argument("--cpu", action="store_true")
    apply_torch.add_argument("--allow-rejected", action="store_true", help=argparse.SUPPRESS)
    apply_torch.add_argument("--rig-fingerprint", default="")
    apply_torch.add_argument("--guitar", default="")
    apply_torch.add_argument("--tuning", default="")
    apply_torch.add_argument("--pickup", default="")
    apply_torch.add_argument("--pickup-mode", default="")
    apply_torch.add_argument("--guitar-volume", default="")
    apply_torch.add_argument("--guitar-tone", default="")
    apply_torch.set_defaults(func=run_apply_torch_reference_command)

    apply_nam = subparsers.add_parser(
        "apply-nam-reference",
        help="Render an exported NAM or NAM A2 model in the isolated research environment.",
    )
    apply_nam.add_argument("--input", type=Path, required=True)
    apply_nam.add_argument("--model", type=Path, required=True)
    apply_nam.add_argument("--output", type=Path, required=True)
    apply_nam.add_argument("--input-trim-db", type=float, default=0.0)
    apply_nam.add_argument("--output-trim-db", type=float, default=0.0)
    apply_nam.add_argument("--render-chunk-samples", type=int, default=65536)
    apply_nam.set_defaults(func=run_apply_nam_reference_command)

    hybrid_compare = subparsers.add_parser(
        "hybrid-model-compare",
        help="Rank MLX, PyTorch, and NAM renders with auraloss and the DI-gain-only regression guard.",
    )
    hybrid_compare.add_argument("--di", type=Path, required=True)
    hybrid_compare.add_argument("--target", type=Path, required=True)
    hybrid_compare.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate render in NAME=PATH form; repeat for MLX, PyTorch, or NAM.",
    )
    hybrid_compare.add_argument("--output", type=Path, required=True)
    hybrid_compare.add_argument("--min-improvement-db", type=float, default=AMP_TONE_GUARD_MIN_IMPROVEMENT_DB)
    hybrid_compare.add_argument("--min-movement-db", type=float, default=AMP_TONE_GUARD_MIN_MOVEMENT_DB)
    hybrid_compare.add_argument("--max-listening-spectral-error-db", type=float, default=14.0)
    hybrid_compare.add_argument("--min-listening-correlation", type=float, default=0.50)
    hybrid_compare.add_argument("--max-listening-level-error-db", type=float, default=2.0)
    hybrid_compare.add_argument("--min-listening-section-pass-rate", type=float, default=0.60)
    hybrid_compare.add_argument("--audition-dir", type=Path, default=None)
    hybrid_compare.add_argument("--audition-seconds", type=float, default=10.0)
    hybrid_compare.set_defaults(func=run_hybrid_model_compare_command)

    conditioned_dataset = subparsers.add_parser(
        "build-conditioned-dataset",
        help="Index every complete recording pair with exact-rig and guitar/pickup/tuning condition labels.",
    )
    conditioned_dataset.add_argument("--dataset", type=Path, required=True)
    conditioned_dataset.add_argument("--output", type=Path, required=True)
    conditioned_dataset.add_argument(
        "--allow-mixed-rigs",
        action="store_true",
        help="Keep multiple rigs only as explicit condition groups; never silently merge their targets.",
    )
    conditioned_dataset.set_defaults(func=run_build_conditioned_dataset_command)

    frozen_dataset = subparsers.add_parser(
        "freeze-conditioned-dataset",
        help="Hash-lock an approved conditioned dataset without duplicating its recording audio.",
    )
    frozen_dataset.add_argument("--dataset-manifest", type=Path, required=True)
    frozen_dataset.add_argument("--output", type=Path, required=True)
    frozen_dataset.set_defaults(func=run_freeze_conditioned_dataset_command)

    rig_probe_generate = subparsers.add_parser(
        "rig-probe-generate",
        help="Generate a calibrated multilevel test signal for a fixed pedal/amp/cab/mic rig.",
    )
    rig_probe_generate.add_argument(
        "--output",
        type=Path,
        default=Path("rig_captures/probes/rig_probe_96k.wav"),
    )
    rig_probe_generate.add_argument("--manifest", type=Path, default=None)
    rig_probe_generate.add_argument("--sample-rate", type=int, default=96000)
    rig_probe_generate.add_argument(
        "--peak-dbfs",
        type=float,
        default=-18.0,
        help="Digital probe peak before reamp calibration. Start conservatively.",
    )
    rig_probe_generate.add_argument("--seed", type=int, default=6505)
    rig_probe_generate.set_defaults(func=run_rig_probe_generate_command)

    rig_probe_record = subparsers.add_parser(
        "rig-probe-record",
        help="Play a controlled probe through a reamp path and record the fixed rig return.",
    )
    rig_probe_record.add_argument("--probe", type=Path, required=True)
    rig_probe_record.add_argument("--probe-manifest", type=Path, default=None)
    rig_probe_record.add_argument("--capture-name", required=True)
    rig_probe_record.add_argument("--output-dir", type=Path, default=Path("rig_captures"))
    rig_probe_record.add_argument("--input-device", default=None)
    rig_probe_record.add_argument("--output-device", default=None)
    rig_probe_record.add_argument("--input-channels", type=int, default=2)
    rig_probe_record.add_argument("--target-channel", type=int, default=2)
    rig_probe_record.add_argument("--output-channels", type=int, default=2)
    rig_probe_record.add_argument("--output-channel", type=int, default=1)
    rig_probe_record.add_argument("--send-trim-db", type=float, default=0.0)
    rig_probe_record.add_argument(
        "--input-impedance-kohm",
        type=float,
        default=1000.0,
        help="Physical interface guitar-input impedance used for the source DI path.",
    )
    rig_probe_record.add_argument("--blocksize", type=int, default=1024)
    rig_probe_record.add_argument("--latency", choices=["low", "high"], default="high")
    rig_probe_record.add_argument(
        "--capture-type",
        choices=["amp-cab", "amp-preamp", "pedal-only"],
        default="amp-cab",
        help="HeadRush-inspired clone category used for routing safety, labeling, and later cabinet separation.",
    )
    rig_probe_record.add_argument("--pedal", default="none")
    rig_probe_record.add_argument("--amp", default="Peavey 6505 Mini Head")
    rig_probe_record.add_argument("--amp-settings", default="rhythm channel")
    rig_probe_record.add_argument("--cabinet", default="Egnater Tweaker 1x12 Celestion")
    rig_probe_record.add_argument("--mic", default="Shure SM57")
    rig_probe_record.add_argument("--mic-position", default="SM57 close, directly in front of speaker")
    rig_probe_record.add_argument("--send-calibration-dbu", type=float, default=None)
    rig_probe_record.add_argument("--return-calibration-dbu", type=float, default=None)
    rig_probe_record.add_argument("--notes", default="")
    rig_probe_record.add_argument(
        "--clone-control",
        action="append",
        default=[],
        help="Document a physical control as NAME=VALUE. Repeat for gain, bass, middle, treble, master, or pedal controls.",
    )
    rig_probe_record.add_argument("--dry-run", action="store_true")
    rig_probe_record.add_argument(
        "--confirm-reamp-routing",
        action="store_true",
        help="Required before probe playback; confirms interface output feeds a reamp box, never a speaker output.",
    )
    rig_probe_record.set_defaults(func=run_rig_probe_record_command)

    train_rig_capture = subparsers.add_parser(
        "train-rig-capture",
        help="Train one calibrated causal MLX model from a controlled fixed-rig probe/return pair.",
    )
    train_rig_capture.add_argument("--probe", type=Path, required=True)
    train_rig_capture.add_argument("--target", type=Path, required=True)
    train_rig_capture.add_argument("--probe-manifest", type=Path, default=None)
    train_rig_capture.add_argument("--capture-manifest", type=Path, default=None)
    train_rig_capture.add_argument("--model", type=Path, required=True)
    train_rig_capture.add_argument("--output", type=Path, required=True)
    train_rig_capture.add_argument("--comparison-output", type=Path, default=None)
    train_rig_capture.add_argument("--render-sample-rate", type=int, default=96000)
    train_rig_capture.add_argument("--oversample-factor", type=int, choices=[1, 2, 4], default=2)
    train_rig_capture.add_argument(
        "--levels",
        type=int,
        default=13,
        help="Causal dilation depth; 13 gives about 85 ms of memory at the default 192 kHz model rate.",
    )
    train_rig_capture.add_argument("--channels", type=int, default=16)
    train_rig_capture.add_argument("--chunk-samples", type=int, default=4096)
    train_rig_capture.add_argument("--render-chunk-samples", type=int, default=32768)
    train_rig_capture.add_argument("--batch-chunks", type=int, default=2)
    train_rig_capture.add_argument("--chunks-per-epoch", type=int, default=48)
    train_rig_capture.add_argument("--validation-chunks", type=int, default=8)
    train_rig_capture.add_argument("--epochs", type=int, default=100)
    train_rig_capture.add_argument("--learning-rate", type=float, default=0.0005)
    train_rig_capture.add_argument("--min-learning-rate", type=float, default=0.00002)
    train_rig_capture.add_argument("--lr-decay", type=float, default=0.5)
    train_rig_capture.add_argument("--lr-patience", type=int, default=4)
    train_rig_capture.add_argument("--early-stopping-patience", type=int, default=12)
    train_rig_capture.add_argument("--min-delta", type=float, default=0.0001)
    train_rig_capture.add_argument("--gradient-clip-norm", type=float, default=1.0)
    train_rig_capture.add_argument("--max-validation-esr", type=float, default=1.0)
    train_rig_capture.add_argument("--min-validation-correlation", type=float, default=0.15)
    train_rig_capture.add_argument("--max-validation-spectral-error-db", type=float, default=10.0)
    train_rig_capture.add_argument("--min-amp-movement-db", type=float, default=1.5)
    train_rig_capture.add_argument("--comparison-seconds", type=float, default=10.0)
    train_rig_capture.add_argument("--allow-failed-validation", action="store_true")
    train_rig_capture.add_argument("--print-every", type=int, default=5)
    train_rig_capture.add_argument("--seed", type=int, default=6505)
    train_rig_capture.set_defaults(func=run_train_rig_capture_command)

    refine_rig_capture = subparsers.add_parser(
        "refine-rig-capture",
        help="Refine an accepted controlled rig model with aligned real-guitar DI and amp/mic playing.",
    )
    refine_rig_capture.add_argument("--model", type=Path, required=True, help="Accepted controlled rig model.")
    refine_rig_capture.add_argument("--di", type=Path, required=True, help="Clean DI from the refinement take.")
    refine_rig_capture.add_argument("--target", type=Path, required=True, help="Simultaneous amp/cab/mic target.")
    refine_rig_capture.add_argument("--output-model", type=Path, required=True)
    refine_rig_capture.add_argument("--output", type=Path, required=True)
    refine_rig_capture.add_argument("--comparison-output", type=Path, default=None)
    refine_rig_capture.add_argument(
        "--input-trim-db",
        type=float,
        default=None,
        help="DI-to-reamp calibration trim. Omit to measure it automatically from the base model and target return.",
    )
    refine_rig_capture.add_argument("--output-sample-rate", type=int, default=96000)
    refine_rig_capture.add_argument("--chunk-samples", type=int, default=4096)
    refine_rig_capture.add_argument("--render-chunk-samples", type=int, default=32768)
    refine_rig_capture.add_argument("--batch-chunks", type=int, default=2)
    refine_rig_capture.add_argument("--chunks-per-epoch", type=int, default=32)
    refine_rig_capture.add_argument("--validation-chunks", type=int, default=8)
    refine_rig_capture.add_argument("--validation-fraction", type=float, default=0.20)
    refine_rig_capture.add_argument("--epochs", type=int, default=30)
    refine_rig_capture.add_argument("--learning-rate", type=float, default=0.00010)
    refine_rig_capture.add_argument("--min-learning-rate", type=float, default=0.00001)
    refine_rig_capture.add_argument("--lr-decay", type=float, default=0.5)
    refine_rig_capture.add_argument("--lr-patience", type=int, default=3)
    refine_rig_capture.add_argument("--early-stopping-patience", type=int, default=8)
    refine_rig_capture.add_argument("--min-delta", type=float, default=0.00005)
    refine_rig_capture.add_argument("--gradient-clip-norm", type=float, default=0.75)
    refine_rig_capture.add_argument("--max-validation-esr", type=float, default=1.0)
    refine_rig_capture.add_argument("--min-validation-correlation", type=float, default=0.15)
    refine_rig_capture.add_argument("--max-validation-spectral-error-db", type=float, default=10.0)
    refine_rig_capture.add_argument("--min-amp-movement-db", type=float, default=1.5)
    refine_rig_capture.add_argument("--min-esr-improvement", type=float, default=0.01)
    refine_rig_capture.add_argument("--min-spectral-improvement-db", type=float, default=0.15)
    refine_rig_capture.add_argument("--min-correlation-improvement", type=float, default=0.005)
    refine_rig_capture.add_argument("--comparison-seconds", type=float, default=10.0)
    refine_rig_capture.add_argument("--allow-failed-validation", action="store_true")
    refine_rig_capture.add_argument("--print-every", type=int, default=2)
    refine_rig_capture.add_argument("--seed", type=int, default=6505)
    refine_rig_capture.set_defaults(func=run_refine_rig_capture_command)

    build_cabinet_variant = subparsers.add_parser(
        "build-cabinet-variant",
        help="Build a measured cabinet/microphone correction from repeated controlled probe returns.",
    )
    build_cabinet_variant.add_argument("--probe", type=Path, required=True)
    build_cabinet_variant.add_argument("--probe-manifest", type=Path, default=None)
    build_cabinet_variant.add_argument("--reference-target", type=Path, required=True)
    build_cabinet_variant.add_argument("--variant-target", type=Path, required=True)
    build_cabinet_variant.add_argument("--profile", type=Path, required=True)
    build_cabinet_variant.add_argument("--comparison-output", type=Path, default=None)
    build_cabinet_variant.add_argument("--name", required=True)
    build_cabinet_variant.add_argument("--reference-cabinet", default="reference cabinet")
    build_cabinet_variant.add_argument("--reference-microphone", default="Shure SM57")
    build_cabinet_variant.add_argument("--reference-mic-position", default="reference position")
    build_cabinet_variant.add_argument("--variant-cabinet", default="same cabinet")
    build_cabinet_variant.add_argument("--variant-microphone", default="Shure SM57")
    build_cabinet_variant.add_argument("--variant-mic-position", default="variant position")
    build_cabinet_variant.add_argument("--variant-mic-axis", choices=["on-axis", "off-axis"], default="on-axis")
    build_cabinet_variant.add_argument("--fft-size", type=int, default=262144)
    build_cabinet_variant.add_argument("--fir-length", type=int, default=8192)
    build_cabinet_variant.add_argument("--smoothing-bins", type=int, default=129)
    build_cabinet_variant.add_argument("--max-correction-db", type=float, default=12.0)
    build_cabinet_variant.add_argument("--level-mode", choices=["tone-only", "preserve"], default="tone-only")
    build_cabinet_variant.add_argument("--min-spectral-improvement-db", type=float, default=0.50)
    build_cabinet_variant.add_argument("--comparison-seconds", type=float, default=10.0)
    build_cabinet_variant.add_argument("--allow-failed-validation", action="store_true")
    build_cabinet_variant.set_defaults(func=run_build_cabinet_variant_command)

    separated_cabinet = subparsers.add_parser(
        "build-separated-cabinet",
        help="Derive a guarded cabinet/speaker/SM57 stage from matched amp-preamp and amp-cab probe captures.",
    )
    separated_cabinet.add_argument("--preamp-capture-manifest", type=Path, required=True)
    separated_cabinet.add_argument("--amp-cab-capture-manifest", type=Path, required=True)
    separated_cabinet.add_argument("--profile", type=Path, required=True)
    separated_cabinet.add_argument("--comparison-output", type=Path, default=None)
    separated_cabinet.add_argument("--name", required=True)
    separated_cabinet.add_argument("--mic-axis", choices=["on-axis", "off-axis"], default="on-axis")
    separated_cabinet.add_argument("--fft-size", type=int, default=262144)
    separated_cabinet.add_argument("--fir-length", type=int, default=2048)
    separated_cabinet.add_argument("--smoothing-bins", type=int, default=129)
    separated_cabinet.add_argument("--max-correction-db", type=float, default=18.0)
    separated_cabinet.add_argument("--level-mode", choices=["tone-only", "preserve"], default="tone-only")
    separated_cabinet.add_argument("--min-spectral-improvement-db", type=float, default=0.50)
    separated_cabinet.add_argument("--comparison-seconds", type=float, default=10.0)
    separated_cabinet.add_argument("--allow-failed-validation", action="store_true")
    separated_cabinet.set_defaults(func=run_build_separated_cabinet_command)

    apply_cabinet_variant = subparsers.add_parser(
        "apply-cabinet-variant",
        help="Apply an accepted measured cabinet/microphone variant to a rig-model WAV.",
    )
    apply_cabinet_variant.add_argument("--input", type=Path, required=True)
    apply_cabinet_variant.add_argument("--profile", type=Path, required=True)
    apply_cabinet_variant.add_argument("--output", type=Path, required=True)
    apply_cabinet_variant.add_argument("--mix", type=float, default=1.0)
    apply_cabinet_variant.add_argument("--low-cut-hz", type=float, default=0.0)
    apply_cabinet_variant.add_argument("--high-cut-hz", type=float, default=0.0)
    apply_cabinet_variant.add_argument("--output-trim-db", type=float, default=0.0)
    apply_cabinet_variant.add_argument("--limiter", choices=["soft", "off"], default="soft")
    apply_cabinet_variant.set_defaults(func=run_apply_cabinet_variant_command)

    apply_virtual_studio = subparsers.add_parser(
        "apply-virtual-studio",
        help="Render a Two notes-inspired dual measured-mic cabinet studio around an existing rig WAV.",
    )
    apply_virtual_studio.add_argument("--input", type=Path, required=True)
    apply_virtual_studio.add_argument("--output", type=Path, required=True)
    apply_virtual_studio.add_argument(
        "--mic-a",
        type=Path,
        default=None,
        help="Measured cabinet/mic variant for mic A; omit for the rig's captured reference mic.",
    )
    apply_virtual_studio.add_argument("--mic-b", type=Path, default=None)
    apply_virtual_studio.add_argument("--mic-a-variant-mix", type=float, default=1.0)
    apply_virtual_studio.add_argument("--mic-morph", "--center", dest="mic_morph", type=float, default=0.5)
    apply_virtual_studio.add_argument("--mic-a-level-db", type=float, default=0.0)
    apply_virtual_studio.add_argument("--mic-b-level-db", type=float, default=0.0)
    apply_virtual_studio.add_argument("--mic-a-pan", type=float, default=-0.15)
    apply_virtual_studio.add_argument("--mic-b-pan", type=float, default=0.15)
    apply_virtual_studio.add_argument("--variphi-ms", type=float, default=0.0)
    apply_virtual_studio.add_argument("--invert-mic-b", action="store_true")
    apply_virtual_studio.add_argument(
        "--room-preset",
        choices=["off", "tight", "studio", "live"],
        default="off",
    )
    apply_virtual_studio.add_argument("--distance", type=float, default=0.0)
    apply_virtual_studio.add_argument("--room-mix", type=float, default=0.20)
    apply_virtual_studio.add_argument("--speaker-overload", type=float, default=0.0)
    apply_virtual_studio.add_argument("--low-cut-hz", type=float, default=0.0)
    apply_virtual_studio.add_argument("--high-cut-hz", type=float, default=0.0)
    apply_virtual_studio.add_argument("--output-trim-db", type=float, default=0.0)
    apply_virtual_studio.add_argument("--limiter", choices=["soft", "off"], default="soft")
    apply_virtual_studio.set_defaults(func=run_apply_virtual_studio_command)

    apply_rig_capture = subparsers.add_parser(
        "apply-rig-capture",
        help="Apply a calibrated causal rig model to a DI without automatic peak normalization.",
    )
    apply_rig_capture.add_argument("--input", type=Path, required=True)
    apply_rig_capture.add_argument("--model", type=Path, required=True)
    apply_rig_capture.add_argument("--output", type=Path, required=True)
    apply_rig_capture.add_argument("--input-trim-db", type=float, default=0.0)
    apply_rig_capture.add_argument(
        "--ignore-model-input-calibration",
        action="store_true",
        help="Ignore the DI input trim learned during real-guitar refinement.",
    )
    apply_rig_capture.add_argument("--output-trim-db", type=float, default=0.0)
    apply_rig_capture.add_argument(
        "--cabinet-variant",
        type=Path,
        default=None,
        help="Optional measured cabinet/microphone variant profile applied after the rig model.",
    )
    apply_rig_capture.add_argument("--cabinet-mix", type=float, default=1.0)
    apply_rig_capture.add_argument("--cabinet-low-cut-hz", type=float, default=0.0)
    apply_rig_capture.add_argument("--cabinet-high-cut-hz", type=float, default=0.0)
    apply_rig_capture.add_argument(
        "--virtual-mic-b",
        type=Path,
        default=None,
        help="Optional second accepted cabinet/mic variant for the virtual studio.",
    )
    apply_rig_capture.add_argument("--virtual-mic-morph", type=float, default=0.5)
    apply_rig_capture.add_argument("--virtual-mic-a-level-db", type=float, default=0.0)
    apply_rig_capture.add_argument("--virtual-mic-b-level-db", type=float, default=0.0)
    apply_rig_capture.add_argument("--virtual-mic-a-pan", type=float, default=0.0)
    apply_rig_capture.add_argument("--virtual-mic-b-pan", type=float, default=0.0)
    apply_rig_capture.add_argument("--virtual-variphi-ms", type=float, default=0.0)
    apply_rig_capture.add_argument("--virtual-invert-mic-b", action="store_true")
    apply_rig_capture.add_argument(
        "--virtual-room-preset",
        choices=["off", "tight", "studio", "live"],
        default="off",
    )
    apply_rig_capture.add_argument("--virtual-distance", type=float, default=0.0)
    apply_rig_capture.add_argument("--virtual-room-mix", type=float, default=0.20)
    apply_rig_capture.add_argument("--virtual-speaker-overload", type=float, default=0.0)
    apply_rig_capture.add_argument("--output-sample-rate", type=int, default=96000)
    apply_rig_capture.add_argument("--chunk-samples", type=int, default=32768)
    apply_rig_capture.add_argument("--limiter", choices=["soft", "off"], default="soft")
    apply_rig_capture.set_defaults(func=run_apply_rig_capture_command)

    build_performance_rig = subparsers.add_parser(
        "build-performance-rig",
        help="Bundle accepted models and guarded speaker/cabinet/runtime controls into a portable performance rig.",
    )
    build_performance_rig.add_argument("--preset", type=Path, required=True)
    build_performance_rig.add_argument("--name", required=True)
    build_performance_rig.add_argument("--model", type=Path, required=True)
    build_performance_rig.add_argument(
        "--secondary-model",
        type=Path,
        default=None,
        help="Optional second accepted controlled model for morph or parallel-path processing.",
    )
    build_performance_rig.add_argument("--model-morph", type=float, default=0.0)
    build_performance_rig.add_argument("--model-path-mode", choices=["morph", "parallel"], default="morph")
    build_performance_rig.add_argument("--primary-level-db", type=float, default=0.0)
    build_performance_rig.add_argument("--secondary-level-db", type=float, default=0.0)
    build_performance_rig.add_argument("--input-trim-db", type=float, default=0.0)
    build_performance_rig.add_argument(
        "--input-impedance-kohm",
        type=float,
        default=None,
        help="Physical interface impedance used for the captured DI. Defaults to capture metadata, then 1000 kOhm.",
    )
    build_performance_rig.add_argument("--gate-threshold-dbfs", type=float, default=-100.0)
    build_performance_rig.add_argument("--bass-db", type=float, default=0.0)
    build_performance_rig.add_argument("--middle-db", type=float, default=0.0)
    build_performance_rig.add_argument("--treble-db", type=float, default=0.0)
    build_performance_rig.add_argument("--cabinet-variant", type=Path, default=None)
    build_performance_rig.add_argument(
        "--cabinet-variant-b",
        type=Path,
        default=None,
        help="Optional second accepted measured mic endpoint for continuous position morphing.",
    )
    build_performance_rig.add_argument("--mic-position-morph", type=float, default=0.0)
    build_performance_rig.add_argument(
        "--cabinet-ir",
        type=Path,
        default=None,
        help="Full cabinet IR for an Amp / Pre-Amp capture. Refused for Amp & Cab models to prevent cabinet stacking.",
    )
    build_performance_rig.add_argument("--cabinet-ir-samples", type=int, choices=[1024, 2048], default=2048)
    build_performance_rig.add_argument("--cabinet-mix", type=float, default=1.0)
    build_performance_rig.add_argument("--cabinet-low-cut-hz", type=float, default=0.0)
    build_performance_rig.add_argument("--cabinet-high-cut-hz", type=float, default=0.0)
    build_performance_rig.add_argument(
        "--speaker-impedance-curve",
        choices=["flat", "open-back-1x12", "closed-back-1x12", "closed-back-4x12", "custom"],
        default="flat",
        help="Cabinet-linked resonance approximation for Amp / Pre-Amp captures; flat is neutral.",
    )
    build_performance_rig.add_argument("--speaker-resonance-hz", type=float, default=100.0)
    build_performance_rig.add_argument("--speaker-resonance-db", type=float, default=0.0)
    build_performance_rig.add_argument("--speaker-resonance-q", type=float, default=1.2)
    build_performance_rig.add_argument("--speaker-presence-hz", type=float, default=2800.0)
    build_performance_rig.add_argument("--speaker-presence-db", type=float, default=0.0)
    build_performance_rig.add_argument("--speaker-presence-q", type=float, default=0.8)
    build_performance_rig.add_argument("--dynamic-speaker-drive", type=float, default=0.0)
    build_performance_rig.add_argument("--cone-cry", type=float, default=0.0)
    build_performance_rig.add_argument("--cone-cry-hz", type=float, default=2600.0)
    build_performance_rig.add_argument("--speaker-reference-level-dbfs", type=float, default=-18.0)
    build_performance_rig.add_argument(
        "--destination",
        choices=["studio-frfr", "headphones", "amp-input", "amp-return", "power-amp-guitar-cab"],
        default="studio-frfr",
    )
    build_performance_rig.add_argument(
        "--snapshots-json",
        type=Path,
        default=None,
        help="Optional JSON object of named runtime-setting snapshots copied into the portable preset.",
    )
    build_performance_rig.add_argument("--output-sample-rate", type=int, default=96000)
    build_performance_rig.add_argument("--output-trim-db", type=float, default=0.0)
    build_performance_rig.add_argument("--normalize", choices=["off", "peak"], default="off")
    build_performance_rig.add_argument("--normalize-peak-dbfs", type=float, default=-1.0)
    build_performance_rig.add_argument("--limiter", choices=["soft", "off"], default="soft")
    build_performance_rig.add_argument("--notes", default="")
    build_performance_rig.set_defaults(func=run_build_performance_rig_command)

    apply_performance_rig = subparsers.add_parser(
        "apply-performance-rig",
        help="Render a clean DI through a portable performance-rig preset.",
    )
    apply_performance_rig.add_argument("--input", type=Path, required=True)
    apply_performance_rig.add_argument("--preset", type=Path, required=True)
    apply_performance_rig.add_argument("--output", type=Path, required=True)
    apply_performance_rig.add_argument("--input-trim-db", type=float, default=0.0)
    apply_performance_rig.add_argument("--output-trim-db", type=float, default=0.0)
    apply_performance_rig.add_argument("--model-morph", type=float, default=None)
    apply_performance_rig.add_argument("--snapshot", default=None)
    apply_performance_rig.add_argument(
        "--destination",
        choices=["studio-frfr", "headphones", "amp-input", "amp-return", "power-amp-guitar-cab"],
        default=None,
    )
    apply_performance_rig.add_argument("--source-input-impedance-kohm", type=float, default=None)
    apply_performance_rig.add_argument("--allow-input-impedance-mismatch", action="store_true")
    apply_performance_rig.add_argument("--output-sample-rate", type=int, default=None)
    apply_performance_rig.add_argument("--chunk-samples", type=int, default=32768)
    apply_performance_rig.set_defaults(func=run_apply_performance_rig_command)

    denoise_preview = subparsers.add_parser(
        "denoise-preview",
        help="Render an explicit noisereduce preview WAV without changing training data.",
    )
    denoise_preview.add_argument("--input", type=Path, required=True, help="Input WAV path.")
    denoise_preview.add_argument("--output", type=Path, required=True, help="Output denoised preview WAV path.")
    denoise_preview.add_argument("--stationary", action="store_true", help="Use stationary spectral gating.")
    denoise_preview.add_argument("--prop-decrease", type=float, default=0.55, help="Noise reduction amount from 0 to 1.")
    denoise_preview.add_argument("--no-level-match", dest="level_match", action="store_false", default=True)
    denoise_preview.add_argument("--peak", type=float, default=0.70, help="Audition peak normalization.")
    denoise_preview.set_defaults(func=run_denoise_preview_command)

    pedalboard_preview = subparsers.add_parser(
        "pedalboard-preview",
        help="Render an explicit pedalboard effect-chain preview without changing training data.",
    )
    pedalboard_preview.add_argument("--input", type=Path, required=True, help="Input WAV path.")
    pedalboard_preview.add_argument("--output", type=Path, required=True, help="Output effected preview WAV path.")
    pedalboard_preview.add_argument("--preset", choices=["transparent", "tighten", "drive-check"], default="tighten")
    pedalboard_preview.add_argument("--no-level-match", dest="level_match", action="store_false", default=True)
    pedalboard_preview.add_argument("--peak", type=float, default=0.70, help="Audition peak normalization.")
    pedalboard_preview.set_defaults(func=run_pedalboard_preview_command)

    hardware_plan = subparsers.add_parser("hardware-plan", help="Write a DI box/audio interface routing manifest.")
    hardware_plan.add_argument("--take-name", default=default_take_name)
    hardware_plan.add_argument("--output", type=Path, default=Path("hardware/di_interface_plan.json"))
    add_interface_args(hardware_plan)
    add_di_box_args(hardware_plan)
    add_take_metadata_args(hardware_plan)
    hardware_plan.set_defaults(func=run_hardware_plan_command)

    level_check = subparsers.add_parser("level-check", help="Record a short no-file gain-staging check.")
    level_check.add_argument("--duration-s", type=float, default=8.0)
    level_check.add_argument("--sample-rate", type=int, default=44100)
    level_check.add_argument("--device", default=None, help="Audio input device index/name. Omit for system default.")
    level_check.add_argument("--input-channels", type=int, default=2, help="Total interface input channels to record.")
    level_check.add_argument("--di-channel", type=int, default=1, help="Clean DI input channel, using 1-based numbering.")
    level_check.add_argument("--target-channel", type=int, default=2, help="Amp/mic target channel, using 1-based numbering.")
    add_di_box_args(level_check)
    add_level_profile_args(level_check)
    level_check.set_defaults(func=run_level_check_command)

    live_scope = subparsers.add_parser(
        "live-scope",
        help="Show the default high-performance PyQtGraph live waveform/spectrum/tone-diff scope.",
    )
    live_scope.add_argument("--sample-rate", type=int, default=96000)
    live_scope.add_argument("--duration-s", type=float, default=1.0, help=argparse.SUPPRESS)
    live_scope.add_argument("--device", default=None, help="Audio input device index/name. Omit for system default.")
    live_scope.add_argument("--input-channels", type=int, default=2, help="Total interface input channels to record.")
    live_scope.add_argument("--di-channel", type=int, default=1, help="Clean DI input channel, using 1-based numbering.")
    live_scope.add_argument("--target-channel", type=int, default=2, help="Amp/mic target channel, using 1-based numbering.")
    live_scope.add_argument("--view", choices=["both", "waveform", "spectrum"], default="both")
    live_scope.add_argument("--block-ms", type=float, default=4.0, help="Audio callback block size in milliseconds.")
    live_scope.add_argument("--window-ms", type=float, default=120.0, help="Visible waveform history in milliseconds.")
    live_scope.add_argument("--refresh-ms", type=int, default=8, help="Graph refresh interval in milliseconds.")
    live_scope.add_argument("--fft-size", type=int, default=4096)
    live_scope.add_argument("--display-points", type=int, default=1200)
    live_scope.add_argument("--min-freq", type=float, default=20.0)
    live_scope.add_argument("--max-freq", type=float, default=12000.0)
    live_scope.add_argument("--min-db", type=float, default=-110.0)
    live_scope.add_argument("--max-db", type=float, default=0.0)
    live_scope.add_argument("--diff-min-db", type=float, default=-36.0)
    live_scope.add_argument("--diff-max-db", type=float, default=36.0)
    live_scope.add_argument("--spectrum-smoothing-bins", type=int, default=7)
    live_scope.add_argument("--tone-diff-smoothing-bins", type=int, default=71)
    live_scope.add_argument("--smoothing-attack", type=float, default=0.92)
    live_scope.add_argument("--smoothing-release", type=float, default=0.16)
    live_scope.add_argument(
        "--visual-smoothing",
        choices=["off", "light", "medium", "heavy", "studio", "ultra", "fluid", "hyperfluid"],
        default="medium",
    )
    live_scope.add_argument("--amplitude-range", type=float, default=1.0)
    live_scope.add_argument("--waveform-layout", choices=["stacked", "overlay"], default="stacked")
    live_scope.add_argument("--waveform-linewidth", type=float, default=1.05)
    live_scope.add_argument("--clip-guard", type=float, default=0.95)
    live_scope.add_argument("--metrics-window-ms", type=float, default=80.0)
    live_scope.add_argument(
        "--source-analysis-ms",
        type=float,
        default=420.0,
        help="Longer window for pickup/output/blower tone analysis.",
    )
    live_scope.add_argument("--metrics-max-delay-ms", type=float, default=10.0)
    live_scope.add_argument("--responsive", action="store_true", help="Use lower latency graph settings for pick attack.")
    live_scope.add_argument("--linear-frequency", dest="log_frequency", action="store_false")
    live_scope.add_argument(
        "--feature-log",
        type=Path,
        default=None,
        help="Optional JSONL output path for MLX-readable live graph telemetry.",
    )
    live_scope.add_argument(
        "--feature-log-interval-ms",
        type=float,
        default=120.0,
        help="How often to write feature telemetry frames when --feature-log is enabled.",
    )
    live_scope.add_argument("--width", type=int, default=1500)
    live_scope.add_argument("--height", type=int, default=950)
    add_pickup_frequency_view_args(live_scope)
    live_scope.add_argument("--opengl", action="store_true", help="Ask PyQtGraph to use OpenGL acceleration.")
    live_scope.add_argument("--antialias", action="store_true", help="Smoother lines; may reduce maximum FPS.")
    live_scope.add_argument("--hide-waveform-details", dest="show_waveform_details", action="store_false")
    live_scope.add_argument("--hide-tone-diff", dest="show_tone_diff", action="store_false")
    live_scope.add_argument("--hide-levels", dest="show_levels", action="store_false")
    live_scope.add_argument("--hide-metrics", dest="show_metrics", action="store_false")
    add_level_profile_args(live_scope)
    live_scope.set_defaults(
        func=run_live_scope_qt_command,
        log_frequency=True,
        show_waveform_details=True,
        show_tone_diff=True,
        show_levels=True,
        show_metrics=True,
    )

    live_scope_legacy = subparsers.add_parser(
        "live-scope-legacy",
        help="Show the older Matplotlib live waveform and frequency graph.",
    )
    live_scope_legacy.add_argument("--sample-rate", type=int, default=44100)
    live_scope_legacy.add_argument("--duration-s", type=float, default=1.0, help=argparse.SUPPRESS)
    live_scope_legacy.add_argument("--device", default=None, help="Audio input device index/name. Omit for system default.")
    live_scope_legacy.add_argument("--input-channels", type=int, default=2, help="Total interface input channels to record.")
    live_scope_legacy.add_argument("--di-channel", type=int, default=1, help="Clean DI input channel, using 1-based numbering.")
    live_scope_legacy.add_argument("--target-channel", type=int, default=2, help="Amp/mic target channel, using 1-based numbering.")
    live_scope_legacy.add_argument("--view", choices=["both", "waveform", "spectrum"], default="both")
    live_scope_legacy.add_argument("--block-ms", type=float, default=25.0, help="Audio callback block size in milliseconds.")
    live_scope_legacy.add_argument("--window-ms", type=float, default=120.0, help="Visible waveform history in milliseconds.")
    live_scope_legacy.add_argument("--refresh-ms", type=int, default=40, help="Graph refresh interval in milliseconds.")
    live_scope_legacy.add_argument("--fft-size", type=int, default=4096)
    live_scope_legacy.add_argument("--min-freq", type=float, default=20.0)
    live_scope_legacy.add_argument("--max-freq", type=float, default=12000.0)
    live_scope_legacy.add_argument("--min-db", type=float, default=-110.0)
    live_scope_legacy.add_argument("--max-db", type=float, default=0.0)
    live_scope_legacy.add_argument("--diff-min-db", type=float, default=-36.0)
    live_scope_legacy.add_argument("--diff-max-db", type=float, default=36.0)
    live_scope_legacy.add_argument("--spectrum-smoothing-bins", type=int, default=9)
    live_scope_legacy.add_argument("--tone-diff-smoothing-bins", type=int, default=31)
    live_scope_legacy.add_argument(
        "--visual-smoothing",
        choices=["off", "light", "medium", "heavy", "studio", "ultra", "fluid", "hyperfluid"],
        default="medium",
    )
    live_scope_legacy.add_argument("--amplitude-range", type=float, default=1.0)
    live_scope_legacy.add_argument("--waveform-layout", choices=["stacked", "overlay"], default="stacked")
    live_scope_legacy.add_argument("--waveform-linewidth", type=float, default=1.05)
    live_scope_legacy.add_argument("--clip-guard", type=float, default=0.95)
    live_scope_legacy.add_argument("--metrics-window-ms", type=float, default=80.0)
    live_scope_legacy.add_argument("--metrics-max-delay-ms", type=float, default=10.0)
    live_scope_legacy.add_argument("--responsive", action="store_true", help="Use lower latency graph settings for pick attack.")
    live_scope_legacy.add_argument("--linear-frequency", dest="log_frequency", action="store_false")
    live_scope_legacy.add_argument("--hide-waveform-details", dest="show_waveform_details", action="store_false")
    live_scope_legacy.add_argument("--hide-tone-diff", dest="show_tone_diff", action="store_false")
    live_scope_legacy.add_argument("--hide-levels", dest="show_levels", action="store_false")
    live_scope_legacy.add_argument("--hide-metrics", dest="show_metrics", action="store_false")
    add_level_profile_args(live_scope_legacy)
    live_scope_legacy.set_defaults(
        func=run_live_scope_command,
        log_frequency=True,
        show_waveform_details=True,
        show_tone_diff=True,
        show_levels=True,
        show_metrics=True,
    )

    live_scope_qt = subparsers.add_parser(
        "live-scope-qt",
        help="Show a faster PyQtGraph live waveform/spectrum/tone-diff scope.",
    )
    live_scope_qt.add_argument("--sample-rate", type=int, default=96000)
    live_scope_qt.add_argument("--duration-s", type=float, default=1.0, help=argparse.SUPPRESS)
    live_scope_qt.add_argument("--device", default=None, help="Audio input device index/name. Omit for system default.")
    live_scope_qt.add_argument("--input-channels", type=int, default=2, help="Total interface input channels to record.")
    live_scope_qt.add_argument("--di-channel", type=int, default=1, help="Clean DI input channel, using 1-based numbering.")
    live_scope_qt.add_argument("--target-channel", type=int, default=2, help="Amp/mic target channel, using 1-based numbering.")
    live_scope_qt.add_argument("--block-ms", type=float, default=4.0, help="Audio callback block size in milliseconds.")
    live_scope_qt.add_argument("--window-ms", type=float, default=120.0, help="Visible waveform history in milliseconds.")
    live_scope_qt.add_argument("--refresh-ms", type=int, default=8, help="Graph refresh interval in milliseconds.")
    live_scope_qt.add_argument("--fft-size", type=int, default=4096)
    live_scope_qt.add_argument("--display-points", type=int, default=1200)
    live_scope_qt.add_argument("--min-freq", type=float, default=20.0)
    live_scope_qt.add_argument("--max-freq", type=float, default=12000.0)
    live_scope_qt.add_argument("--min-db", type=float, default=-110.0)
    live_scope_qt.add_argument("--max-db", type=float, default=0.0)
    live_scope_qt.add_argument("--diff-min-db", type=float, default=-36.0)
    live_scope_qt.add_argument("--diff-max-db", type=float, default=36.0)
    live_scope_qt.add_argument("--spectrum-smoothing-bins", type=int, default=7)
    live_scope_qt.add_argument("--tone-diff-smoothing-bins", type=int, default=71)
    live_scope_qt.add_argument("--smoothing-attack", type=float, default=0.92)
    live_scope_qt.add_argument("--smoothing-release", type=float, default=0.16)
    live_scope_qt.add_argument("--amplitude-range", type=float, default=1.0)
    live_scope_qt.add_argument("--clip-guard", type=float, default=0.95)
    live_scope_qt.add_argument("--metrics-window-ms", type=float, default=36.0)
    live_scope_qt.add_argument(
        "--source-analysis-ms",
        type=float,
        default=420.0,
        help="Longer window for pickup/output/blower tone analysis.",
    )
    live_scope_qt.add_argument(
        "--feature-log",
        type=Path,
        default=None,
        help="Optional JSONL output path for MLX-readable live graph telemetry.",
    )
    live_scope_qt.add_argument(
        "--feature-log-interval-ms",
        type=float,
        default=120.0,
        help="How often to write feature telemetry frames when --feature-log is enabled.",
    )
    live_scope_qt.add_argument("--width", type=int, default=1500)
    live_scope_qt.add_argument("--height", type=int, default=950)
    live_scope_qt.add_argument("--responsive", action="store_true", help="Use very low latency graph settings.")
    live_scope_qt.add_argument("--linear-frequency", dest="log_frequency", action="store_false")
    add_pickup_frequency_view_args(live_scope_qt)
    live_scope_qt.add_argument("--opengl", action="store_true", help="Ask PyQtGraph to use OpenGL acceleration.")
    live_scope_qt.add_argument("--antialias", action="store_true", help="Smoother lines; may reduce maximum FPS.")
    add_level_profile_args(live_scope_qt)
    live_scope_qt.set_defaults(func=run_live_scope_qt_command, log_frequency=True)

    record = subparsers.add_parser("record", help="Record clean DI and amp/mic target from a two-channel interface.")
    record.add_argument("--take-name", default=default_take_name)
    record.add_argument("--output-dir", type=Path, default=Path("recordings"))
    record.add_argument("--dataset", type=Path, default=None, help="Optional dataset manifest to append this take to.")
    add_interface_args(record)
    add_di_box_args(record)
    add_take_metadata_args(record)
    add_level_profile_args(record)
    record.set_defaults(func=run_record_command)

    record_capture = subparsers.add_parser("record-capture", help="Record a two-channel hardware take and capture a profile.")
    record_capture.add_argument("--take-name", default=default_take_name)
    record_capture.add_argument("--output-dir", type=Path, default=Path("recordings"))
    record_capture.add_argument("--dataset", type=Path, default=None, help="Optional dataset manifest to append this take to.")
    record_capture.add_argument("--profile", type=Path, default=None, help="Output profile JSON path.")
    record_capture.add_argument("--reconstructed", type=Path, default=None, help="Optional reconstructed match WAV output.")
    record_capture.add_argument("--instrument", choices=["guitar", "bass"], default="guitar")
    record_capture.add_argument("--name", default="hardware_captured_tone")
    record_capture.add_argument("--ir-ms", type=float, default=32.0)
    record_capture.add_argument("--regularization", type=float, default=0.002)
    record_capture.add_argument("--no-bias-search", action="store_true")
    add_interface_args(record_capture)
    add_di_box_args(record_capture)
    add_take_metadata_args(record_capture)
    add_level_profile_args(record_capture)
    record_capture.set_defaults(func=run_record_capture_command)

    cleanup_unused = subparsers.add_parser(
        "cleanup-unused-takes",
        help="Dry-run, archive, or delete dataset takes not selected by current training filters.",
    )
    cleanup_unused.add_argument("--dataset", type=Path, required=True, help="Dataset manifest JSON to clean.")
    cleanup_unused.add_argument("--profile-family", default="", help="Only clean takes from this profile family.")
    cleanup_unused.add_argument(
        "--include-take",
        action="append",
        default=[],
        help="Treat only this take as selected for training. Repeat for multiple takes.",
    )
    cleanup_unused.add_argument(
        "--exclude-take",
        action="append",
        default=[],
        help="Treat this take as excluded from training and therefore cleanup-eligible.",
    )
    cleanup_unused.add_argument(
        "--preferred-only",
        action="store_true",
        help="Treat only preferred dataset takes as selected for training.",
    )
    cleanup_unused.add_argument(
        "--include-unusable",
        action="store_true",
        help="Treat unusable takes as eligible for training selection.",
    )
    cleanup_unused.add_argument(
        "--cleanup-mode",
        choices=["archive", "delete"],
        default="archive",
        help="Archive moves files out of the active set; delete permanently removes them.",
    )
    cleanup_unused.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("archived_unused_takes"),
        help="Where archived unused take files are moved.",
    )
    cleanup_unused.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform cleanup. Omit this for a dry run.",
    )
    cleanup_unused.add_argument(
        "--confirm-delete-unused",
        action="store_true",
        help="Required with --cleanup-mode delete and --apply.",
    )
    cleanup_unused.set_defaults(func=run_cleanup_unused_takes_command)

    capture = subparsers.add_parser("capture", help="Capture a tone profile from DI and target WAV files.")
    capture.add_argument("--di", type=Path, required=True, help="Clean DI WAV file.")
    capture.add_argument("--target", type=Path, required=True, help="Processed amp/cab target WAV file.")
    capture.add_argument("--profile", type=Path, required=True, help="Output profile JSON path.")
    capture.add_argument("--reconstructed", type=Path, default=None, help="Optional reconstructed match WAV output.")
    capture.add_argument("--manifest", type=Path, default=None, help="Optional DI/interface hardware manifest JSON.")
    capture.add_argument("--instrument", choices=["guitar", "bass"], default="guitar")
    capture.add_argument("--name", default="captured_tone")
    capture.add_argument("--ir-ms", type=float, default=32.0)
    capture.add_argument("--regularization", type=float, default=0.002)
    capture.add_argument("--no-bias-search", action="store_true")
    capture.set_defaults(func=run_capture_command)

    apply = subparsers.add_parser("apply", help="Apply a saved tone profile to a DI WAV file.")
    apply.add_argument("--input", type=Path, required=True, help="Input DI WAV file.")
    apply.add_argument("--profile", type=Path, required=True, help="Tone profile JSON path.")
    apply.add_argument("--output", type=Path, required=True, help="Output profiled WAV path.")
    apply.set_defaults(func=run_apply_command)

    tone_match = subparsers.add_parser("tone-match", help="Render an audible spectral tone-match from DI and mic target.")
    tone_match.add_argument("--di", type=Path, required=True, help="Clean DI WAV file.")
    tone_match.add_argument("--target", type=Path, required=True, help="Amp/mic target WAV file.")
    tone_match.add_argument("--profile", type=Path, default=None, help="Optional DSP profile JSON for nonlinear settings.")
    tone_match.add_argument("--output", type=Path, required=True, help="Output spectral tone-match WAV path.")
    tone_match.add_argument("--comparison-output", type=Path, default=None, help="Optional mic-then-match comparison WAV.")
    tone_match.add_argument("--fft-size", type=int, default=8192)
    tone_match.add_argument("--smoothing-bins", type=int, default=91)
    tone_match.add_argument("--amp-style", choices=["neutral", "mic-layer", "high-gain", "6505"], default="neutral")
    tone_match.add_argument("--drive-boost", type=float, default=1.0)
    tone_match.set_defaults(func=run_tone_match_command)

    mic_learn = subparsers.add_parser("mic-learn", help="Learn a nonlinear DI-to-SM57 model directly from a mic target.")
    mic_learn.add_argument("--di", type=Path, required=True, help="Clean DI WAV file.")
    mic_learn.add_argument("--target", type=Path, required=True, help="Amp/mic target WAV file.")
    mic_learn.add_argument("--model", type=Path, required=True, help="Output mic-learned model JSON path.")
    mic_learn.add_argument("--output", type=Path, required=True, help="Output mic-learned render WAV path.")
    mic_learn.add_argument("--comparison-output", type=Path, default=None, help="Optional mic-then-learned comparison WAV.")
    mic_learn.add_argument("--ir-ms", type=float, default=96.0)
    mic_learn.add_argument("--regularization", type=float, default=0.02)
    mic_learn.set_defaults(func=run_mic_learn_command)

    apply_mic = subparsers.add_parser("apply-mic-model", help="Apply a saved mic-learned model to a DI WAV.")
    apply_mic.add_argument("--input", type=Path, required=True, help="Input DI WAV file.")
    apply_mic.add_argument("--model", type=Path, required=True, help="Mic-learned model JSON path.")
    apply_mic.add_argument("--output", type=Path, required=True, help="Output WAV path.")
    apply_mic.set_defaults(func=run_apply_mic_model_command)

    train_mlx_bridge = subparsers.add_parser(
        "train-mlx-bridge",
        help="Train MLX to tighten the bridge between a clean DI and SM57 target.",
    )
    train_mlx_bridge.add_argument("--di", type=Path, required=True, help="Clean DI WAV file.")
    train_mlx_bridge.add_argument("--target", type=Path, required=True, help="Amp/mic target WAV file.")
    train_mlx_bridge.add_argument("--base-model", type=Path, required=True, help="Output mic-learned base model JSON path.")
    train_mlx_bridge.add_argument("--model", type=Path, required=True, help="Output MLX bridge model .npz path.")
    train_mlx_bridge.add_argument("--output", type=Path, required=True, help="Output MLX bridge render WAV path.")
    train_mlx_bridge.add_argument("--base-output", type=Path, default=None, help="Optional mic-learned base render WAV.")
    train_mlx_bridge.add_argument("--comparison-output", type=Path, default=None, help="Optional mic-then-MLX comparison WAV.")
    train_mlx_bridge.add_argument("--ir-ms", type=float, default=96.0)
    train_mlx_bridge.add_argument("--regularization", type=float, default=0.02)
    train_mlx_bridge.add_argument("--epochs", type=int, default=35)
    train_mlx_bridge.add_argument("--batch-size", type=int, default=2048)
    train_mlx_bridge.add_argument("--learning-rate", type=float, default=0.0012)
    train_mlx_bridge.add_argument("--hidden-dim", type=int, default=96)
    train_mlx_bridge.add_argument("--context-radius", type=int, default=48)
    train_mlx_bridge.add_argument("--max-train-samples", type=int, default=180000)
    train_mlx_bridge.add_argument("--max-training-seconds", type=float, default=22.0)
    train_mlx_bridge.add_argument("--validation-fraction", type=float, default=0.1)
    train_mlx_bridge.add_argument("--residual-mix", type=float, default=0.9)
    train_mlx_bridge.add_argument("--chunk-samples", type=int, default=65536)
    train_mlx_bridge.add_argument("--print-every", type=int, default=5)
    train_mlx_bridge.add_argument("--seed", type=int, default=2026)
    train_mlx_bridge.set_defaults(func=run_train_mlx_bridge_command)

    apply_mlx_bridge = subparsers.add_parser("apply-mlx-bridge", help="Apply a mic-learned base model plus MLX bridge model.")
    apply_mlx_bridge.add_argument("--input", type=Path, required=True, help="Input DI WAV file.")
    apply_mlx_bridge.add_argument("--base-model", type=Path, required=True, help="Mic-learned base model JSON path.")
    apply_mlx_bridge.add_argument("--model", type=Path, required=True, help="MLX bridge model .npz path.")
    apply_mlx_bridge.add_argument("--output", type=Path, required=True, help="Output WAV path.")
    apply_mlx_bridge.add_argument("--chunk-samples", type=int, default=65536)
    apply_mlx_bridge.set_defaults(func=run_apply_mlx_bridge_command)

    train_mlx_spectrum = subparsers.add_parser(
        "train-mlx-spectrum",
        help="Train MLX on full DI and SM57 frequency spectra frame by frame.",
    )
    train_mlx_spectrum.add_argument("--di", type=Path, required=True, help="Clean DI WAV file.")
    train_mlx_spectrum.add_argument("--target", type=Path, required=True, help="Amp/mic target WAV file.")
    train_mlx_spectrum.add_argument("--model", type=Path, required=True, help="Output MLX full-spectrum model .npz path.")
    train_mlx_spectrum.add_argument("--output", type=Path, required=True, help="Output rendered WAV path.")
    train_mlx_spectrum.add_argument("--comparison-output", type=Path, default=None, help="Optional mic-then-spectrum comparison WAV.")
    train_mlx_spectrum.add_argument("--fft-size", type=int, default=4096)
    train_mlx_spectrum.add_argument("--hop-size", type=int, default=1024)
    train_mlx_spectrum.add_argument("--hidden-dim", type=int, default=256)
    train_mlx_spectrum.add_argument("--epochs", type=int, default=90)
    train_mlx_spectrum.add_argument("--batch-size", type=int, default=128)
    train_mlx_spectrum.add_argument("--learning-rate", type=float, default=0.0008)
    train_mlx_spectrum.add_argument("--validation-fraction", type=float, default=0.12)
    train_mlx_spectrum.add_argument("--gain-scale-db", type=float, default=24.0)
    train_mlx_spectrum.add_argument("--max-gain-db", type=float, default=36.0)
    train_mlx_spectrum.add_argument("--training-smoothing-bins", type=int, default=9)
    train_mlx_spectrum.add_argument("--output-smoothing-bins", type=int, default=5)
    train_mlx_spectrum.add_argument("--smoothness-weight", type=float, default=0.001)
    train_mlx_spectrum.add_argument("--batch-frames", type=int, default=128)
    train_mlx_spectrum.add_argument("--print-every", type=int, default=10)
    train_mlx_spectrum.add_argument("--seed", type=int, default=6505)
    train_mlx_spectrum.set_defaults(func=run_train_mlx_spectrum_command)

    apply_mlx_spectrum = subparsers.add_parser(
        "apply-mlx-spectrum",
        help="Apply a saved full-spectrum MLX DI-to-SM57 model to a DI WAV.",
    )
    apply_mlx_spectrum.add_argument("--input", type=Path, required=True, help="Input DI WAV file.")
    apply_mlx_spectrum.add_argument("--model", type=Path, required=True, help="MLX full-spectrum model .npz path.")
    apply_mlx_spectrum.add_argument("--output", type=Path, required=True, help="Output WAV path.")
    apply_mlx_spectrum.add_argument("--batch-frames", type=int, default=128)
    apply_mlx_spectrum.set_defaults(func=run_apply_mlx_spectrum_command)

    train_all_recordings_amp = subparsers.add_parser(
        "train-all-recordings-amp",
        help="Train an amp-dominant model from all paired recordings, then render it onto a DI.",
    )
    train_all_recordings_amp.add_argument("--recordings-dir", type=Path, default=Path("recordings"))
    train_all_recordings_amp.add_argument(
        "--input",
        type=Path,
        default=None,
        help="DI WAV to render after training. Defaults to the newest paired clean DI recording.",
    )
    train_all_recordings_amp.add_argument(
        "--comparison-target",
        type=Path,
        default=None,
        help="Optional amp/mic WAV for final A/B and amp-favored spectral imprint. Defaults to the matching take target.",
    )
    train_all_recordings_amp.add_argument(
        "--model",
        type=Path,
        default=Path("profiles/sm57_amp_mlx_amp_all_recordings_amp_dominant.npz"),
    )
    train_all_recordings_amp.add_argument(
        "--training-output",
        type=Path,
        default=Path("outputs/sm57_amp_mlx_amp_all_recordings_amp_dominant_training_render.wav"),
        help="Training validation render from the first discovered pair.",
    )
    train_all_recordings_amp.add_argument(
        "--training-comparison-output",
        type=Path,
        default=Path("outputs/sm57_amp_mic_then_all_recordings_amp_dominant_training.wav"),
    )
    train_all_recordings_amp.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sm57_amp_all_recordings_amp_dominant_di_render.wav"),
        help="Final render of --input through the trained amp model.",
    )
    train_all_recordings_amp.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("outputs/sm57_amp_mic_then_all_recordings_amp_dominant_di_render.wav"),
    )
    train_all_recordings_amp.add_argument(
        "--per-take-output-dir",
        type=Path,
        default=Path("outputs/per_take_validation_all_recordings_amp_dominant"),
    )
    train_all_recordings_amp.add_argument(
        "--skip-per-take-validation",
        action="store_true",
        help="Skip writing per-take WAVs; held-out promotion scoring still runs for safety.",
    )
    train_all_recordings_amp.add_argument("--include-level-tests", action="store_true")
    train_all_recordings_amp.add_argument(
        "--usable-only",
        action="store_true",
        help="Only use discovered takes whose manifest marks them usable_for_training.",
    )
    train_all_recordings_amp.add_argument("--list-only", action="store_true", help="List discovered pairs without training.")
    train_all_recordings_amp.add_argument(
        "--no-quality-gate",
        dest="quality_gate",
        action="store_false",
        default=True,
        help="Disable automatic exclusion/downweighting of DI-like or outlier amp/mic takes.",
    )
    train_all_recordings_amp.add_argument(
        "--keep-quality-excluded",
        dest="quality_exclude_bad",
        action="store_false",
        default=True,
        help="Keep bad quality-gate takes in training, but still downweight and report them.",
    )
    train_all_recordings_amp.add_argument(
        "--quality-min-weight",
        type=float,
        default=AMP_QUALITY_MIN_WEIGHT,
        help="Quality weight below this value is excluded when the quality gate is enabled.",
    )
    train_all_recordings_amp.add_argument("--mic-position", default="")
    train_all_recordings_amp.add_argument(
        "--rig-policy",
        choices=["conditioned", "strict", "match-input", "largest"],
        default="conditioned",
        help="Condition mixed rigs explicitly, reject them, or select one exact-rig group.",
    )
    train_all_recordings_amp.add_argument(
        "--rig-fingerprint",
        default="",
        help="Train/apply one exact rig fingerprint reported by --list-only.",
    )
    train_all_recordings_amp.add_argument("--model-sample-rate", type=int, default=96000)
    train_all_recordings_amp.add_argument("--render-sample-rate", type=int, default=96000)
    train_all_recordings_amp.add_argument(
        "--context-radius",
        type=int,
        default=960,
        help="Half-window in samples; 960 provides about 20 ms total waveform memory at 96 kHz.",
    )
    train_all_recordings_amp.add_argument("--hidden-dim", type=int, default=384)
    train_all_recordings_amp.add_argument("--epochs", type=int, default=180)
    train_all_recordings_amp.add_argument("--batch-size", type=int, default=2048)
    train_all_recordings_amp.add_argument("--learning-rate", type=float, default=0.00025)
    train_all_recordings_amp.add_argument("--gradient-clip-norm", type=float, default=0.5)
    train_all_recordings_amp.add_argument("--early-stopping-patience", type=int, default=14)
    train_all_recordings_amp.add_argument("--lr-patience", type=int, default=4)
    train_all_recordings_amp.add_argument("--lr-decay", type=float, default=0.5)
    train_all_recordings_amp.add_argument("--min-learning-rate", type=float, default=0.00002)
    train_all_recordings_amp.add_argument("--min-delta", type=float, default=0.00001)
    train_all_recordings_amp.add_argument("--robust-loss-delta", type=float, default=0.08)
    train_all_recordings_amp.add_argument("--max-loss-spike-ratio", type=float, default=10.0)
    train_all_recordings_amp.add_argument(
        "--legacy-normalize-training-pairs",
        dest="preserve_training_levels",
        action="store_false",
        default=True,
        help="Use the old per-take peak normalization; normally leave this off so pickup/output level is learned.",
    )
    train_all_recordings_amp.add_argument("--max-train-samples", type=int, default=16000)
    train_all_recordings_amp.add_argument("--max-training-seconds", type=float, default=120.0)
    train_all_recordings_amp.add_argument("--validation-fraction", type=float, default=0.1)
    train_all_recordings_amp.add_argument(
        "--loss-mode",
        choices=["detail-spectral", "detail", "mse"],
        default="detail-spectral",
        help="detail-spectral adds ESR and log-FFT losses so the model learns amp/cab spectral shape.",
    )
    train_all_recordings_amp.add_argument("--detail-chunk-samples", type=int, default=2048)
    train_all_recordings_amp.add_argument("--detail-chunks-per-epoch", type=int, default=160)
    train_all_recordings_amp.add_argument("--transient-loss-weight", type=float, default=0.50)
    train_all_recordings_amp.add_argument("--highfreq-loss-weight", type=float, default=0.50)
    train_all_recordings_amp.add_argument("--envelope-loss-weight", type=float, default=0.18)
    train_all_recordings_amp.add_argument("--esr-loss-weight", type=float, default=0.35)
    train_all_recordings_amp.add_argument("--spectral-loss-weight", type=float, default=0.22)
    train_all_recordings_amp.add_argument("--cab-lowpass-hz", type=float, default=7800.0)
    train_all_recordings_amp.add_argument("--cab-highpass-hz", type=float, default=75.0)
    train_all_recordings_amp.add_argument("--cab-presence-db", type=float, default=3.5)
    train_all_recordings_amp.add_argument("--cab-air-db", type=float, default=0.8)
    train_all_recordings_amp.add_argument(
        "--hybrid-cabinet-profile",
        type=Path,
        default=None,
        help="Optional accepted measured cabinet/microphone profile applied after the nonlinear model.",
    )
    train_all_recordings_amp.add_argument("--hybrid-cabinet-mix", type=float, default=1.0)
    train_all_recordings_amp.add_argument("--amp-anchor-strength", type=float, default=1.0)
    train_all_recordings_amp.add_argument("--amp-anchor-smoothing-bins", type=int, default=AMP_TONE_ANCHOR_SMOOTHING_BINS)
    train_all_recordings_amp.add_argument("--amp-anchor-max-gain-db", type=float, default=AMP_TONE_ANCHOR_MAX_GAIN_DB)
    train_all_recordings_amp.add_argument("--model-input-trim-db", type=float, default=0.0)
    train_all_recordings_amp.add_argument("--output-peak-dbfs", type=float, default=-1.31)
    train_all_recordings_amp.add_argument("--mic-imprint-strength", type=float, default=1.0)
    train_all_recordings_amp.add_argument("--mic-imprint-smoothing-bins", type=int, default=AMP_TONE_ANCHOR_SMOOTHING_BINS)
    train_all_recordings_amp.add_argument("--mic-imprint-max-gain-db", type=float, default=AMP_TONE_ANCHOR_MAX_GAIN_DB)
    train_all_recordings_amp.add_argument(
        "--no-amp-tone-guard",
        dest="amp_tone_guard",
        action="store_false",
        default=True,
        help="Disable the final DI-gain-only regression guard.",
    )
    train_all_recordings_amp.add_argument(
        "--amp-tone-guard-min-improvement-db",
        type=float,
        default=AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
    )
    train_all_recordings_amp.add_argument(
        "--amp-tone-guard-min-movement-db",
        type=float,
        default=AMP_TONE_GUARD_MIN_MOVEMENT_DB,
    )
    train_all_recordings_amp.add_argument("--chunk-samples", type=int, default=65536)
    train_all_recordings_amp.add_argument("--max-heldout-spectral-error-db", type=float, default=9.0)
    train_all_recordings_amp.add_argument("--min-heldout-correlation", type=float, default=0.08)
    train_all_recordings_amp.add_argument("--max-heldout-level-error-db", type=float, default=3.0)
    train_all_recordings_amp.add_argument(
        "--max-heldout-mean-spectral-error-db",
        type=float,
        default=AMP_HELDOUT_MAX_MEAN_SPECTRAL_ERROR_DB,
    )
    train_all_recordings_amp.add_argument(
        "--min-heldout-mean-correlation",
        type=float,
        default=AMP_HELDOUT_MIN_MEAN_CORRELATION,
    )
    train_all_recordings_amp.add_argument("--min-heldout-pass-rate", type=float, default=AMP_HELDOUT_MIN_PASS_RATE)
    train_all_recordings_amp.add_argument("--min-existing-model-improvement-db", type=float, default=0.10)
    train_all_recordings_amp.add_argument(
        "--max-heldout-pair-regression-db",
        type=float,
        default=AMP_HELDOUT_MAX_PAIR_REGRESSION_DB,
    )
    train_all_recordings_amp.add_argument(
        "--allow-failed-validation",
        action="store_true",
        help="Debug only: promote a candidate that fails held-out production gates.",
    )
    train_all_recordings_amp.add_argument("--print-every", type=int, default=10)
    train_all_recordings_amp.add_argument("--seed", type=int, default=6505)
    train_all_recordings_amp.set_defaults(func=run_train_all_recordings_amp_command)

    train_mlx_amp = subparsers.add_parser(
        "train-mlx-amp",
        help="Train a direct nonlinear MLX amp model from DI waveform to SM57 waveform.",
    )
    train_mlx_amp.add_argument("--di", type=Path, default=None, help="Clean DI WAV file.")
    train_mlx_amp.add_argument("--target", type=Path, default=None, help="Amp/mic target WAV file.")
    train_mlx_amp.add_argument(
        "--mic-position",
        default="",
        help="Mic placement for direct --di/--target or extra pairs when no dataset metadata is available.",
    )
    train_mlx_amp.add_argument("--dataset", type=Path, default=None, help="Dataset manifest JSON to train from.")
    train_mlx_amp.add_argument(
        "--recordings-dir",
        type=Path,
        default=None,
        help="Auto-discover paired *_clean_di.wav and *_amp_mic_target.wav files from this recordings directory.",
    )
    train_mlx_amp.add_argument(
        "--include-level-tests",
        action="store_true",
        help="Include level_test paired WAV files when using --recordings-dir.",
    )
    train_mlx_amp.add_argument(
        "--recordings-usable-only",
        action="store_true",
        help="With --recordings-dir, only use takes whose hardware manifest marks them usable_for_training.",
    )
    train_mlx_amp.add_argument(
        "--rig-policy",
        choices=["conditioned", "strict", "match-input", "largest"],
        default="conditioned",
        help="Keep mixed rigs as explicit one-hot conditions or select/reject exact-rig groups.",
    )
    train_mlx_amp.add_argument("--rig-fingerprint", default="")
    train_mlx_amp.add_argument("--profile-family", default="", help="Only use dataset takes from this profile family.")
    train_mlx_amp.add_argument(
        "--include-take",
        action="append",
        default=[],
        help="Only include a specific dataset take name. Repeat for multiple takes.",
    )
    train_mlx_amp.add_argument(
        "--exclude-take",
        action="append",
        default=[],
        help="Exclude a specific dataset take name. Repeat for multiple takes.",
    )
    train_mlx_amp.add_argument(
        "--preferred-only",
        action="store_true",
        help="Only train from dataset takes marked ideal/preferred.",
    )
    train_mlx_amp.add_argument(
        "--include-unusable",
        action="store_true",
        help="Allow dataset takes marked not usable for training.",
    )
    train_mlx_amp.add_argument(
        "--cleanup-unused",
        action="store_true",
        help="After successful training, automatically clean dataset takes not selected by these filters.",
    )
    train_mlx_amp.add_argument(
        "--cleanup-mode",
        choices=["archive", "delete"],
        default="archive",
        help="Cleanup mode used with --cleanup-unused. Archive is recoverable; delete is permanent.",
    )
    train_mlx_amp.add_argument(
        "--cleanup-archive-dir",
        type=Path,
        default=Path("archived_unused_takes"),
        help="Where --cleanup-unused archives skipped take files.",
    )
    train_mlx_amp.add_argument(
        "--confirm-delete-unused",
        action="store_true",
        help="Required with --cleanup-unused --cleanup-mode delete.",
    )
    train_mlx_amp.add_argument(
        "--extra-pair",
        type=Path,
        nargs=2,
        action="append",
        metavar=("DI", "TARGET"),
        help="Additional clean DI and amp/mic target WAV pair. Repeat for multiple extra takes.",
    )
    train_mlx_amp.add_argument("--model", type=Path, required=True, help="Output MLX neural amp model .npz path.")
    train_mlx_amp.add_argument("--output", type=Path, required=True, help="Output rendered WAV path.")
    train_mlx_amp.add_argument("--comparison-output", type=Path, default=None, help="Optional mic-then-amp-model comparison WAV.")
    train_mlx_amp.add_argument("--model-sample-rate", type=int, default=48000)
    train_mlx_amp.add_argument(
        "--context-radius",
        type=int,
        default=480,
        help="Half-window in samples; 480 provides about 20 ms total waveform memory at 48 kHz.",
    )
    train_mlx_amp.add_argument("--hidden-dim", type=int, default=192)
    train_mlx_amp.add_argument("--epochs", type=int, default=80)
    train_mlx_amp.add_argument("--batch-size", type=int, default=2048)
    train_mlx_amp.add_argument("--learning-rate", type=float, default=0.00030)
    train_mlx_amp.add_argument("--gradient-clip-norm", type=float, default=0.5)
    train_mlx_amp.add_argument("--early-stopping-patience", type=int, default=14)
    train_mlx_amp.add_argument("--lr-patience", type=int, default=4)
    train_mlx_amp.add_argument("--lr-decay", type=float, default=0.5)
    train_mlx_amp.add_argument("--min-learning-rate", type=float, default=0.00002)
    train_mlx_amp.add_argument("--min-delta", type=float, default=0.00001)
    train_mlx_amp.add_argument("--robust-loss-delta", type=float, default=0.08)
    train_mlx_amp.add_argument("--max-loss-spike-ratio", type=float, default=10.0)
    train_mlx_amp.add_argument(
        "--legacy-normalize-training-pairs",
        dest="preserve_training_levels",
        action="store_false",
        default=True,
        help="Use old per-take peak normalization instead of preserving pickup and DI output levels.",
    )
    train_mlx_amp.add_argument("--max-train-samples", type=int, default=12000)
    train_mlx_amp.add_argument("--max-training-seconds", type=float, default=60.0)
    train_mlx_amp.add_argument("--validation-fraction", type=float, default=0.1)
    train_mlx_amp.add_argument(
        "--take-sampling",
        choices=["balanced", "length"],
        default="balanced",
        help="How detail-mode picks training chunks across multiple takes.",
    )
    train_mlx_amp.add_argument(
        "--conditioning-mode",
        choices=["source-stats", "none"],
        default="source-stats",
        help="Append DI source descriptors so the shared amp model can adapt across guitars/pickups.",
    )
    train_mlx_amp.add_argument(
        "--mic-position-conditioning",
        dest="mic_position_conditioning",
        action="store_true",
        default=True,
        help="Append coarse mic-position metadata to source-stats conditioning.",
    )
    train_mlx_amp.add_argument(
        "--no-mic-position-conditioning",
        dest="mic_position_conditioning",
        action="store_false",
        help="Disable mic-position metadata conditioning.",
    )
    train_mlx_amp.add_argument(
        "--loss-mode",
        choices=["detail-spectral", "detail", "mse"],
        default="detail-spectral",
        help="detail-spectral adds ESR and log-FFT losses; detail is waveform/transient/envelope only; mse is the older sample loss.",
    )
    train_mlx_amp.add_argument(
        "--no-quality-gate",
        dest="quality_gate",
        action="store_false",
        default=True,
        help="Disable automatic exclusion/downweighting of DI-like or outlier amp/mic takes.",
    )
    train_mlx_amp.add_argument(
        "--keep-quality-excluded",
        dest="quality_exclude_bad",
        action="store_false",
        default=True,
        help="Keep bad quality-gate takes in training, but still downweight and report them.",
    )
    train_mlx_amp.add_argument(
        "--quality-min-weight",
        type=float,
        default=AMP_QUALITY_MIN_WEIGHT,
        help="Quality weight below this value is excluded when the quality gate is enabled.",
    )
    train_mlx_amp.add_argument("--detail-chunk-samples", type=int, default=2048)
    train_mlx_amp.add_argument("--detail-chunks-per-epoch", type=int, default=96)
    train_mlx_amp.add_argument("--transient-loss-weight", type=float, default=0.45)
    train_mlx_amp.add_argument("--highfreq-loss-weight", type=float, default=0.35)
    train_mlx_amp.add_argument("--envelope-loss-weight", type=float, default=0.12)
    train_mlx_amp.add_argument("--esr-loss-weight", type=float, default=0.35)
    train_mlx_amp.add_argument("--spectral-loss-weight", type=float, default=0.22)
    train_mlx_amp.add_argument("--cab-lowpass-hz", type=float, default=6500.0)
    train_mlx_amp.add_argument("--cab-highpass-hz", type=float, default=75.0)
    train_mlx_amp.add_argument("--cab-presence-db", type=float, default=2.0)
    train_mlx_amp.add_argument("--cab-air-db", type=float, default=0.0)
    train_mlx_amp.add_argument("--hybrid-cabinet-profile", type=Path, default=None)
    train_mlx_amp.add_argument("--hybrid-cabinet-mix", type=float, default=1.0)
    train_mlx_amp.add_argument("--amp-anchor-strength", type=float, default=1.0)
    train_mlx_amp.add_argument("--amp-anchor-smoothing-bins", type=int, default=AMP_TONE_ANCHOR_SMOOTHING_BINS)
    train_mlx_amp.add_argument("--amp-anchor-max-gain-db", type=float, default=AMP_TONE_ANCHOR_MAX_GAIN_DB)
    train_mlx_amp.add_argument(
        "--render-sample-rate",
        type=int,
        default=None,
        help="Optional output WAV sample rate for renders, e.g. 96000 to match a 96 kHz interface session.",
    )
    train_mlx_amp.add_argument(
        "--model-input-trim-db",
        type=float,
        default=0.0,
        help="Trim DI level into the MLX amp model during rendering. Negative values reduce clipped/overdriven artifacts.",
    )
    train_mlx_amp.add_argument(
        "--render-limiter",
        choices=["soft", "off"],
        default="soft",
        help="Soft limits MLX renders before writing; off preserves the raw predicted waveform and relies on headroom.",
    )
    train_mlx_amp.add_argument(
        "--output-peak-dbfs",
        type=float,
        default=-1.31,
        help="Peak level used when normalizing MLX render WAVs before optional target matching.",
    )
    train_mlx_amp.add_argument(
        "--per-take-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-take rendered WAVs and mic-then-model comparisons.",
    )
    train_mlx_amp.add_argument(
        "--skip-per-take-validation",
        action="store_true",
        help="Skip writing per-take WAVs; held-out promotion scoring still runs for safety.",
    )
    train_mlx_amp.add_argument("--max-heldout-spectral-error-db", type=float, default=9.0)
    train_mlx_amp.add_argument("--min-heldout-correlation", type=float, default=0.08)
    train_mlx_amp.add_argument("--max-heldout-level-error-db", type=float, default=3.0)
    train_mlx_amp.add_argument(
        "--max-heldout-mean-spectral-error-db",
        type=float,
        default=AMP_HELDOUT_MAX_MEAN_SPECTRAL_ERROR_DB,
    )
    train_mlx_amp.add_argument(
        "--min-heldout-mean-correlation",
        type=float,
        default=AMP_HELDOUT_MIN_MEAN_CORRELATION,
    )
    train_mlx_amp.add_argument("--min-heldout-pass-rate", type=float, default=AMP_HELDOUT_MIN_PASS_RATE)
    train_mlx_amp.add_argument("--min-existing-model-improvement-db", type=float, default=0.10)
    train_mlx_amp.add_argument(
        "--max-heldout-pair-regression-db",
        type=float,
        default=AMP_HELDOUT_MAX_PAIR_REGRESSION_DB,
    )
    train_mlx_amp.add_argument("--allow-failed-validation", action="store_true")
    train_mlx_amp.add_argument("--chunk-samples", type=int, default=65536)
    train_mlx_amp.add_argument("--print-every", type=int, default=10)
    train_mlx_amp.add_argument("--seed", type=int, default=6505)
    train_mlx_amp.set_defaults(func=run_train_mlx_amp_command)

    apply_mlx_amp = subparsers.add_parser(
        "apply-mlx-amp",
        help="Apply a saved nonlinear MLX amp model to a DI WAV.",
    )
    apply_mlx_amp.add_argument("--input", type=Path, required=True, help="Input DI WAV file.")
    apply_mlx_amp.add_argument("--model", type=Path, required=True, help="MLX neural amp model .npz path.")
    apply_mlx_amp.add_argument("--output", type=Path, required=True, help="Output WAV path.")
    apply_mlx_amp.add_argument(
        "--mic-position",
        default=None,
        help="Requested conditioned mic placement. Omit to use the model default.",
    )
    apply_mlx_amp.add_argument(
        "--rig-fingerprint",
        default=None,
        help="Explicit rig condition from the model metadata; matching recorded DI paths select it automatically.",
    )
    apply_mlx_amp.add_argument(
        "--source-hint",
        type=Path,
        default=None,
        help="Original full-length DI path used to select the exact recorded take while rendering a short clip.",
    )
    apply_mlx_amp.add_argument("--chunk-samples", type=int, default=65536)
    apply_mlx_amp.add_argument("--cab-lowpass-hz", type=float, default=None)
    apply_mlx_amp.add_argument("--cab-highpass-hz", type=float, default=None)
    apply_mlx_amp.add_argument("--cab-presence-db", type=float, default=None)
    apply_mlx_amp.add_argument("--cab-air-db", type=float, default=None)
    apply_mlx_amp.add_argument("--hybrid-cabinet-profile", type=Path, default=None)
    apply_mlx_amp.add_argument("--hybrid-cabinet-mix", type=float, default=1.0)
    apply_mlx_amp.add_argument(
        "--inferred-ir-mix",
        type=float,
        default=DEFAULT_INFERRED_IR_MIX,
        help="Mix for phase-sensitive IR inferred from performances. Default 0 keeps close-mic renders dry; use measured cabinet profiles instead.",
    )
    apply_mlx_amp.add_argument(
        "--render-sample-rate",
        type=int,
        default=None,
        help="Optional output WAV sample rate, e.g. 96000 to export a 96 kHz render.",
    )
    apply_mlx_amp.add_argument(
        "--model-input-trim-db",
        type=float,
        default=0.0,
        help="Trim DI level into the MLX amp model. Try -3 to -8 dB if the render sounds clipped.",
    )
    apply_mlx_amp.add_argument(
        "--render-limiter",
        choices=["soft", "off"],
        default="soft",
        help="Soft limits MLX renders before writing; off can reduce clipped limiter character.",
    )
    apply_mlx_amp.add_argument(
        "--output-peak-dbfs",
        type=float,
        default=-1.31,
        help="Peak level used when normalizing the MLX render before optional target matching.",
    )
    apply_mlx_amp.add_argument(
        "--target-level-match",
        choices=["off", "peak", "rms"],
        default="off",
        help="When --comparison-target is provided, match the render level to the mic target.",
    )
    apply_mlx_amp.add_argument(
        "--mic-imprint-strength",
        type=float,
        default=0.0,
        help="When --comparison-target is provided, apply its averaged SM57 spectral curve to the render. Try 0.4-0.9.",
    )
    apply_mlx_amp.add_argument(
        "--reference-match-mode",
        choices=["balanced", "exact"],
        default="exact",
        help="Exact additionally locks close-mic tone bands and crest behavior to --comparison-target.",
    )
    apply_mlx_amp.add_argument("--mic-imprint-smoothing-bins", type=int, default=AMP_TONE_ANCHOR_SMOOTHING_BINS)
    apply_mlx_amp.add_argument("--mic-imprint-max-gain-db", type=float, default=AMP_TONE_ANCHOR_MAX_GAIN_DB)
    apply_mlx_amp.add_argument(
        "--no-amp-tone-guard",
        dest="amp_tone_guard",
        action="store_false",
        default=True,
        help="Disable the DI-gain-only regression guard when a comparison target is present.",
    )
    apply_mlx_amp.add_argument(
        "--amp-tone-guard-min-improvement-db",
        type=float,
        default=AMP_TONE_GUARD_MIN_IMPROVEMENT_DB,
    )
    apply_mlx_amp.add_argument(
        "--amp-tone-guard-min-movement-db",
        type=float,
        default=AMP_TONE_GUARD_MIN_MOVEMENT_DB,
    )
    apply_mlx_amp.add_argument("--comparison-target", type=Path, default=None, help="Optional real mic WAV for A/B comparison.")
    apply_mlx_amp.add_argument("--comparison-output", type=Path, default=None, help="Optional mic-then-render comparison WAV.")
    apply_mlx_amp.set_defaults(func=run_apply_mlx_amp_command)

    train_mlx = subparsers.add_parser("train-mlx", help="Train an optional MLX neural residual layer for a DSP profile.")
    train_mlx.add_argument("--di", type=Path, required=True, help="Clean DI WAV used for the capture.")
    train_mlx.add_argument("--target", type=Path, required=True, help="Amp/mic target WAV used for the capture.")
    train_mlx.add_argument("--base-profile", type=Path, required=True, help="Existing DSP tone profile JSON.")
    train_mlx.add_argument("--model", type=Path, required=True, help="Output MLX residual model .npz path.")
    train_mlx.add_argument("--enhanced-output", type=Path, default=None, help="Optional MLX-enhanced match WAV.")
    train_mlx.add_argument("--epochs", type=int, default=30)
    train_mlx.add_argument("--batch-size", type=int, default=2048)
    train_mlx.add_argument("--learning-rate", type=float, default=0.0015)
    train_mlx.add_argument("--hidden-dim", type=int, default=64)
    train_mlx.add_argument("--context-radius", type=int, default=16)
    train_mlx.add_argument("--max-train-samples", type=int, default=120000)
    train_mlx.add_argument("--max-training-seconds", type=float, default=18.0)
    train_mlx.add_argument("--validation-fraction", type=float, default=0.1)
    train_mlx.add_argument("--residual-mix", type=float, default=0.85)
    train_mlx.add_argument("--chunk-samples", type=int, default=65536)
    train_mlx.add_argument("--print-every", type=int, default=5)
    train_mlx.add_argument("--seed", type=int, default=1234)
    train_mlx.set_defaults(func=run_train_mlx_command)

    apply_mlx = subparsers.add_parser("apply-mlx", help="Apply a DSP profile plus an optional MLX residual model.")
    apply_mlx.add_argument("--input", type=Path, required=True, help="Input DI WAV file.")
    apply_mlx.add_argument("--profile", type=Path, required=True, help="DSP tone profile JSON path.")
    apply_mlx.add_argument("--model", type=Path, required=True, help="MLX residual model .npz path.")
    apply_mlx.add_argument("--output", type=Path, required=True, help="Output MLX-enhanced WAV path.")
    apply_mlx.add_argument("--chunk-samples", type=int, default=65536)
    apply_mlx.set_defaults(func=run_apply_mlx_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        args = parser.parse_args(["demo"])

    args.func(args)


if __name__ == "__main__":
    main()
