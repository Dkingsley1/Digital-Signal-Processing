import numpy as np
from scipy.io import wavfile

# Match your "Noir" settings
sample_rate = 96000  # 96kHz for your Audient
duration = 5.0  # seconds
amplitude = 0.5  # Avoid clipping


def generate_pink_noise(samples):
    # Generating white noise first
    white = np.random.uniform(-1, 1, samples)

    # Applying a filter to make it "Pink" (3dB per octave drop)
    # Spectral density is proportional to 1/f
    fd = np.fft.rfft(white)
    scaling = np.sqrt(np.arange(1, len(fd) + 1))
    pink_fd = fd / scaling
    pink = np.fft.irfft(pink_fd)
    return pink[:samples]


# 1. Generate the noise
samples = int(sample_rate * duration)
noise = generate_pink_noise(samples)

# 2. Normalize to prevent distortion
noise = noise / np.max(np.abs(noise)) * amplitude

# 3. Convert to 16-bit PCM (standard for most players)
# For a true 24-bit file, we'd use 'soundfile' library instead of wavfile
noise_int16 = (noise * 32767).astype(np.int16)

# 4. Save as a Hi-Res Mono file
wavfile.write("Calibration_Pink_Noise.wav", sample_rate, noise_int16)

print("Calibration file 'Calibration_Pink_Noise.wav' created at 96kHz!")
