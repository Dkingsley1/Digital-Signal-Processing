import sounddevice as sd
import numpy as np
from pedalboard import Pedalboard, Limiter

# --- CONFIG ---
RATE = 96000
CHUNK = 2048
DTYPE = "float32"

# --- WIDTH / FOCUS / HEADROOM ---
WIDEN = 1.45      # keeps spread but preserves center
MID_GAIN = 1.06   # subtle vocal focus
PRE_GAIN = 0.6    # headroom before processing

# --- EFFECT ENGINE ---
board = Pedalboard([
    Limiter(threshold_db=-5.0, release_ms=50.0)
])

def soft_clip(x, drive=1.05):
    return np.tanh(x * drive)

def widen_and_focus(x, widen=WIDEN, mid_gain=MID_GAIN):
    """
    x shape: (frames, 2)
    Mid/Side processing:
    mid = (L + R)/2
    side = (L - R)/2
    """
    if x.shape[1] != 2:
        return x
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1])

    mid *= mid_gain
    side *= widen

    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=1)

def audio_callback(indata, outdata, frames, time_info, status):
    if status:
        print(f"Status Error: {status}")
    try:
        x = indata * PRE_GAIN
        x = widen_and_focus(x)
        x = soft_clip(x, drive=1.05)
        outdata[:] = board(x.T, RATE).T
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
            print("LIVE: Widening + Mid Focus. Use Audient knob for volume.")
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"Final Error Check: {e}")

if __name__ == "__main__":
    run_engine()
