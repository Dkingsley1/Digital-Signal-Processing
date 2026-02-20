import numpy as np
import soundfile as sf  # The "Pro" way to handle 24-bit

# Match your "Noir" settings
sample_rate = 96000  # 96kHz for your Audient
duration = 5.0       # seconds
amplitude = 0.5      # -6dB peak to keep those JBLs safe

def generate_pink_noise(samples):
    white = np.random.uniform(-1, 1, samples)
    # Frequency domain scaling for Pink Noise (1/f)
    fd = np.fft.rfft(white)
    scaling = np.sqrt(np.arange(1, len(fd) + 1))
    pink_fd = fd / scaling
    pink = np.fft.irfft(pink_fd)
    return pink[:samples]

# 1. Generate the noise
samples = int(sample_rate * duration)
noise = generate_pink_noise(samples)

# 2. Normalize to exactly -6dB (0.5 amplitude)
# This prevents any digital inter-sample clipping
noise = (noise / np.max(np.abs(noise))) * amplitude

# 3. Save as a true 24-bit Hi-Res file
# 'PCM_24' ensures it uses the full dynamic range of your Audient
sf.write("Noir_24bit_Calibration.wav", noise, sample_rate, subtype='PCM_24')

print(f"File created: 96kHz / 24-bit / {duration}s")
