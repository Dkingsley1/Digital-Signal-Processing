import sounddevice as sd
import numpy as np
from pedalboard import Pedalboard, Limiter

# --- CONFIG ---
RATE = 96000
CHUNK = 2048
DTYPE = "float32"

# --- WIDEN / HEADROOM ---
WIDEN = 2.2
PRE_GAIN = 0.5
LEFT_TRIM = 0.97   # tiny left channel trim

# --- EFFECT ENGINE ---
board = Pedalboard([
    Limiter(threshold_db=-6.0, release_ms=50.0)
])

def soft_clip(x, drive=1.1):
    return np.tanh(x * drive)

def widen_stereo(x, widen=WIDEN):
    if x.shape[1] != 2:
        return x
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])
    side *= widen
    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=1)

def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"Status Error: {status}")
    try:
        x = indata * PRE_GAIN
        x = widen_stereo(x)
        x[:, 0] *= LEFT_TRIM            # tame left channel
        x = soft_clip(x, drive=1.1)     # shave peaks
        y = board(x.T, RATE).T
        outdata[:] = np.clip(y, -0.98, 0.98)  # hard safety clamp
    except Exception:
        outdata.fill(0)

def run_engine():
    INPUT_ID = "BlackHole 2ch"
    OUTPUT_ID = "Audient iD14"

    print("--- 96kHz SIGNAL CHAIN STARTING ---")
    print(f"Input:  {INPUT_ID}")
    print(f"Output: {OUTPUT_ID}")

    try:
        with sd.Stream(
            device=(INPUT_ID, OUTPUT_ID),
            samplerate=RATE,
            blocksize=CHUNK,
            dtype=DTYPE,
            channels=2,
            callback=audio_callback
        ):
            print("LIVE: Widening system audio. Use Audient knob for volume.")
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"Final Error Check: {e}")

if __name__ == "__main__":
    run_engine()
