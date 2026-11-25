
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# 1. Load preprocessed EEG data
ROOT = Path(__file__).resolve().parent
results = ROOT / "results"

data = np.load(results / "preprocessed_eeg.npz")

X = data["X"]
y = data["y"]
fs = float(data["fs"])

# 2. Select the same EEG examples
non_seizure = X[y == 3][0]
seizure = X[y == 1][0]

N = X.shape[1]

# 3. Calculate FFT amplitude spectrum
freq = np.fft.rfftfreq(N, d=1 / fs)

def amplitude_spectrum(signal):
    amplitude = np.abs(np.fft.rfft(signal)) / N

    if N % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2

    return amplitude

fft_normal = amplitude_spectrum(non_seizure)
fft_seizure = amplitude_spectrum(seizure)

# 4. Plot FFT spectra
fig, ax = plt.subplots(
    2, 1, figsize=(10, 7),
    sharex=True, sharey=True
)

ax[0].plot(freq, fft_normal, color="blue")
ax[0].set_title("FFT Spectrum: Non-seizure")
ax[0].set_ylabel("Amplitude")

ax[1].plot(freq, fft_seizure, color="red")
ax[1].set_title("FFT Spectrum: Seizure")
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude")

for a in ax:
    a.set_xlim(0, 45)
    a.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    results / "04_fft_spectrum.png",
    dpi=300,
    bbox_inches="tight"
)

# 5. Calculate power spectral density
# for all EEG segments
f, psd = welch(
    X,
    fs=fs,
    window="hann",
    nperseg=N,
    axis=1
)

# 6. Calculate mean PSD for each class
psd_normal = np.mean(psd[y != 1], axis=0)
psd_seizure = np.mean(psd[y == 1], axis=0)

# 7. Plot mean PSD comparison
plt.figure(figsize=(11, 5.5))

plt.semilogy(
    f, psd_normal,
    color="blue",
    label="Non-seizure"
)

plt.semilogy(
    f, psd_seizure,
    color="red",
    label="Seizure"
)

# EEG frequency bands
bands = [
    ("Delta", 1, 4),
    ("Theta", 4, 8),
    ("Alpha", 8, 13),
    ("Beta", 13, 30),
    ("Gamma", 30, 40)
]

colors = [
    "lightblue",
    "lightgreen",
    "lightyellow",
    "orange",
    "plum"
]

for (name, low, high), color in zip(bands, colors):
    plt.axvspan(
        low, high,
        color=color,
        alpha=0.2
    )

    plt.text(
        (low + high) / 2,
        0.95,
        name,
        transform=plt.gca().get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9
    )

plt.title("Mean Power Spectral Density of EEG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (Amplitude squared / Hz)")
plt.xlim(0, 45)
plt.ylim(1e-3, 1e5)
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    results / "05_psd_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# 8. Print frequency analysis results
print("=" * 50)
print("EEG Frequency Analysis Results")
print("=" * 50)

print("Sampling frequency:", fs, "Hz")
print("Samples per segment:", N)
print("Frequency resolution:", fs / N, "Hz")

print("\nDominant frequency of selected examples:")

mask = (freq >= 1) & (freq <= 40)

print(
    "Non-seizure:",
    freq[mask][np.argmax(fft_normal[mask])],
    "Hz"
)

print(
    "Seizure:",
    freq[mask][np.argmax(fft_seizure[mask])],
    "Hz"
)

print("\nMean PSD comparison:")
print("Non-seizure segments:", np.sum(y != 1))
print("Seizure segments:", np.sum(y == 1))

print("\nFrequency analysis completed!")
print("Figures saved successfully.")

plt.show()
