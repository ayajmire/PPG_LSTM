# preprocess_external_ppg.py

import numpy as np
from scipy.signal import butter, filtfilt, resample

# === CONFIG ===
TARGET_FS = 100   # Hz
TARGET_DURATION = 2.1  # seconds
TARGET_LENGTH = int(TARGET_FS * TARGET_DURATION)  # 210

# === FILTERING ===
def lowpass_filter(signal, orig_fs=1000, cutoff=5):
    b, a = butter(4, cutoff / (0.5 * orig_fs), btype='low')
    return filtfilt(b, a, signal)

# === MAIN FUNCTION ===
def preprocess_ppg(raw_signal, orig_fs):
    """
    Args:
        raw_signal (np.ndarray): 1D array of raw PPG values
        orig_fs (int): Original sampling frequency (Hz)

    Returns:
        np.ndarray: Preprocessed signal of shape (210,)
    """
    # Step 1: Low-pass filter
    filtered = lowpass_filter(raw_signal, orig_fs)

    # Step 2: Resample to 100 Hz
    num_target_samples = TARGET_LENGTH
    resampled = resample(filtered, num_target_samples)

    # Step 3: Normalize
    normalized = (resampled - np.mean(resampled)) / (np.std(resampled) + 1e-6)

    return normalized

# === DEMO ===
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Simulate 2.1s of PPG at 1000 Hz
    t = np.linspace(0, 2.1, int(30 * 2.1))
    sim_ppg = 0.6 * np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 2.4 * t)
    sim_ppg += 0.05 * np.random.randn(len(t))  # Add noise

    processed = preprocess_ppg(sim_ppg, orig_fs=30)

    plt.plot(processed)
    plt.title("Preprocessed PPG (210 points @ 100 Hz)")
    plt.xlabel("Sample")
    plt.ylabel("Normalized amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
