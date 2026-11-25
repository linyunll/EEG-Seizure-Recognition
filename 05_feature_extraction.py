
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from scipy.stats import skew, kurtosis
from scipy.signal import welch

# 1. Load preprocessed EEG data
ROOT = Path(__file__).resolve().parent
results = ROOT / "results"

data = np.load(results / "preprocessed_eeg.npz")

X = data["X"]
y = data["y"]
fs = float(data["fs"])

# Binary labels: seizure = 1, non-seizure = 0
y_binary = (y == 1).astype(int)

print("EEG data shape:", X.shape)

# 2. Time-domain feature extraction
features = {}

features["Mean"] = np.mean(X, axis=1)
features["Std"] = np.std(X, axis=1)
features["Variance"] = np.var(X, axis=1)
features["Maximum"] = np.max(X, axis=1)
features["Minimum"] = np.min(X, axis=1)
features["Peak_to_Peak"] = np.ptp(X, axis=1)
features["RMS"] = np.sqrt(np.mean(X**2, axis=1))
features["Energy"] = np.sum(X**2, axis=1)
features["Skewness"] = skew(X, axis=1)
features["Kurtosis"] = kurtosis(X, axis=1)

print("Time-domain features extracted: 10")

# 3. Frequency-domain feature extraction
f, psd = welch(
    X,
    fs=fs,
    window="hann",
    nperseg=X.shape[1],
    axis=1
)

dfreq = f[1] - f[0]

bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 40)
}

total_mask = (f >= 1) & (f <= 40)

total_power = (
    np.sum(psd[:, total_mask], axis=1) * dfreq
)

for name, (low, high) in bands.items():

    if name == "Gamma":
        mask = (f >= low) & (f <= high)
    else:
        mask = (f >= low) & (f < high)

    band_power = (
        np.sum(psd[:, mask], axis=1) * dfreq
    )

    features[name + "_Power"] = band_power

    features[name + "_Relative"] = (
        band_power / (total_power + 1e-12)
    )

features["Dominant_Frequency"] = (
    f[total_mask][
        np.argmax(psd[:, total_mask], axis=1)
    ]
)

features["Total_Power"] = total_power

print("Frequency-domain features extracted: 12")

# 4. Wavelet feature extraction
coeffs = pywt.wavedec(
    X,
    wavelet="db4",
    level=4,
    axis=1
)

wavelet_names = [
    "A4", "D4", "D3", "D2", "D1"
]

for name, coeff in zip(wavelet_names, coeffs):

    features["Wavelet_" + name + "_Energy"] = (
        np.sum(coeff**2, axis=1)
    )

print("Wavelet features extracted: 5")

# 5. Construct feature dataset
feature_df = pd.DataFrame(features)

feature_names = feature_df.columns.tolist()

feature_df["Label"] = y_binary

# Check invalid feature values
if not np.isfinite(
    feature_df[feature_names].to_numpy()
).all():
    raise ValueError("Invalid feature values detected.")

# Save feature dataset
feature_df.to_csv(
    results / "eeg_features.csv",
    index=False
)

print("\n" + "=" * 50)
print("Feature Extraction Results")
print("=" * 50)

print("Number of EEG segments:", len(feature_df))
print("Number of features:", len(feature_names))
print("Feature dataset shape:", feature_df.shape)

print("\nFirst five rows:")
print(feature_df.head())

print("\nMissing values:",
      feature_df.isna().sum().sum())

# 6. Plot representative time-domain features
selected_features = [
    "Std", "RMS", "Peak_to_Peak", "Energy"
]

fig, axes = plt.subplots(
    2, 2, figsize=(11, 8)
)

for ax, name in zip(
    axes.flat, selected_features
):

    normal = feature_df.loc[
        feature_df["Label"] == 0, name
    ]

    seizure = feature_df.loc[
        feature_df["Label"] == 1, name
    ]

    bp = ax.boxplot(
        [normal, seizure],
        patch_artist=True,
        showfliers=False
    )

    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["Non-seizure", "Seizure"]
    )

    ax.set_title(name)
    ax.set_ylabel("Feature Value")
    ax.grid(True, alpha=0.3)

fig.suptitle(
    "Time-domain Feature Comparison"
)

plt.tight_layout()

plt.savefig(
    results / "06_time_features.png",
    dpi=300,
    bbox_inches="tight"
)

# 7. Plot mean frequency-band power
band_names = list(bands.keys())

normal_power = [
    feature_df.loc[
        feature_df["Label"] == 0,
        name + "_Power"
    ].mean()
    for name in band_names
]

seizure_power = [
    feature_df.loc[
        feature_df["Label"] == 1,
        name + "_Power"
    ].mean()
    for name in band_names
]

pos = np.arange(len(band_names))
width = 0.35

plt.figure(figsize=(10, 5.5))

plt.bar(
    pos - width / 2,
    normal_power,
    width,
    label="Non-seizure",
    color="royalblue"
)

plt.bar(
    pos + width / 2,
    seizure_power,
    width,
    label="Seizure",
    color="tomato"
)

plt.xticks(pos, band_names)
plt.yscale("log")

plt.title("Mean EEG Band Power Comparison")
plt.xlabel("Frequency Band")
plt.ylabel("Mean Band Power")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    results / "07_band_power.png",
    dpi=300,
    bbox_inches="tight"
)

# 8. Plot wavelet energy comparison
normal_energy = []
seizure_energy = []

for name in wavelet_names:

    column = "Wavelet_" + name + "_Energy"

    normal_energy.append(
        feature_df.loc[
            feature_df["Label"] == 0,
            column
        ].mean()
    )

    seizure_energy.append(
        feature_df.loc[
            feature_df["Label"] == 1,
            column
        ].mean()
    )

pos = np.arange(len(wavelet_names))

plt.figure(figsize=(10, 5.5))

plt.bar(
    pos - width / 2,
    normal_energy,
    width,
    label="Non-seizure",
    color="royalblue"
)

plt.bar(
    pos + width / 2,
    seizure_energy,
    width,
    label="Seizure",
    color="tomato"
)

plt.xticks(pos, wavelet_names)
plt.yscale("log")

plt.title("Wavelet Energy Comparison")
plt.xlabel("Wavelet Decomposition Level")
plt.ylabel("Mean Wavelet Energy")
plt.legend()
plt.grid(True, axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    results / "08_wavelet_energy.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nFeature extraction completed successfully!")
print("Feature dataset and figures saved.")

plt.show()
