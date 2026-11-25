
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz

# 1. Load EEG dataset
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "EEG_data.csv")

signal_cols = [f"X{i}" for i in range(1, 179)]
X = df[signal_cols].to_numpy(dtype=float)
y = df["y"].to_numpy()

# 2. Set sampling frequency
fs = 173.61
t = np.arange(X.shape[1]) / fs

# 3. Remove DC component
X_dc = X - np.mean(X, axis=1, keepdims=True)

# 4. Design Butterworth bandpass filter
lowcut = 1.0
highcut = 40.0
order = 4

sos = butter(
    order,
    [lowcut, highcut],
    btype="bandpass",
    fs=fs,
    output="sos"
)

# 5. Apply zero-phase filtering
X_filtered = sosfiltfilt(sos, X_dc, axis=1)

# 6. Select the same examples as before
normal_idx = np.where(y == 3)[0][0]
seizure_idx = np.where(y == 1)[0][0]

# 7. Create results folder
results = ROOT / "results"
results.mkdir(exist_ok=True)

# 8. Plot signals before and after preprocessing
fig, ax = plt.subplots(2, 2, figsize=(12, 7),
                       sharex=True, sharey="row")

ax[0, 0].plot(t, X[normal_idx], color="blue")
ax[0, 0].set_title("Non-seizure: Original")
ax[0, 0].set_ylabel("Amplitude")

ax[0, 1].plot(t, X_filtered[normal_idx], color="green")
ax[0, 1].set_title("Non-seizure: Filtered")

ax[1, 0].plot(t, X[seizure_idx], color="red")
ax[1, 0].set_title("Seizure: Original")
ax[1, 0].set_ylabel("Amplitude")
ax[1, 0].set_xlabel("Time (s)")

ax[1, 1].plot(t, X_filtered[seizure_idx], color="green")
ax[1, 1].set_title("Seizure: Filtered")
ax[1, 1].set_xlabel("Time (s)")

for a in ax.flat:
    a.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    results / "02_preprocessing_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# 9. Plot filter frequency response
freq, h = sosfreqz(sos, worN=4096, fs=fs)

# Effective response of forward-backward filtering
gain = 20 * np.log10(
    np.maximum(np.abs(h)**2, 1e-10)
)

plt.figure(figsize=(9, 4.5))
plt.plot(freq, gain, color="blue")
plt.axvline(lowcut, color="red", linestyle="--")
plt.axvline(highcut, color="red", linestyle="--")

plt.title("Butterworth Bandpass Filter Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(0, fs / 2)
plt.ylim(-100, 5)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results / "03_filter_response.png",
    dpi=300,
    bbox_inches="tight"
)

# 10. Save preprocessed data
np.savez_compressed(
    results / "preprocessed_eeg.npz",
    X=X_filtered,
    y=y,
    fs=fs
)

# 11. Print experiment information
print("=" * 50)
print("EEG Preprocessing Results")
print("=" * 50)
print("Sampling frequency:", fs, "Hz")
print("Original data shape:", X.shape)
print("Filtered data shape:", X_filtered.shape)
print("Filter type: Butterworth bandpass")
print("Filter order:", order)
print("Passband:", lowcut, "-", highcut, "Hz")

print("\nNon-seizure signal:")
print("Original mean:", np.mean(X[normal_idx]))
print("Filtered mean:", np.mean(X_filtered[normal_idx]))

print("\nSeizure signal:")
print("Original mean:", np.mean(X[seizure_idx]))
print("Filtered mean:", np.mean(X_filtered[seizure_idx]))

print("\nPreprocessing completed successfully!")
print("Figures and filtered data saved.")

plt.show()
