
from pathlib import Path
import pandas as pd
import numpy as np

# 1. Set the dataset path
ROOT = Path(__file__).resolve().parent
data_path = ROOT / "data" / "EEG_data.csv"

# 2. Load the dataset
df = pd.read_csv(data_path)

# 3. Display basic information
print("=" * 50)
print("EEG Dataset Information")
print("=" * 50)

print("Dataset shape:", df.shape)
print("Number of samples:", df.shape[0])
print("Number of columns:", df.shape[1])

# 4. Display the first five rows
print("\nFirst five rows:")
print(df.head())

# 5. Check missing values
print("\nMissing values:", df.isnull().sum().sum())

# 6. Extract EEG signals and labels
signal_cols = [f"X{i}" for i in range(1, 179)]
X = df[signal_cols].to_numpy(dtype=float)
y = df["y"].to_numpy()

print("\nEEG signal shape:", X.shape)
print("Label shape:", y.shape)

# 7. Display class distribution
print("\nClass distribution:")
print(df["y"].value_counts().sort_index())

# 8. Convert labels to binary classes
y_binary = np.where(y == 1, 1, 0)

print("\nBinary class distribution:")
print("Non-seizure:", np.sum(y_binary == 0))
print("Seizure:", np.sum(y_binary == 1))

print("\nData loading completed!")


# 9. Plot class distribution
import matplotlib.pyplot as plt

results = ROOT / "results"
results.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

# Original five-class distribution
counts = df["y"].value_counts().sort_index()

ax[0].bar(
    counts.index.astype(str),
    counts.values,
    color="steelblue",
    edgecolor="black"
)

ax[0].set_title("Original Class Distribution")
ax[0].set_xlabel("Class Label")
ax[0].set_ylabel("Number of Samples")
ax[0].set_ylim(0, 2700)

for i, v in enumerate(counts.values):
    ax[0].text(i, v + 50, str(v), ha="center")

# Binary class distribution
binary_counts = [
    np.sum(y_binary == 0),
    np.sum(y_binary == 1)
]

ax[1].bar(
    ["Non-seizure", "Seizure"],
    binary_counts,
    color=["steelblue", "coral"],
    edgecolor="black"
)

ax[1].set_title("Binary Class Distribution")
ax[1].set_ylabel("Number of Samples")
ax[1].set_ylim(0, 10500)

for i, v in enumerate(binary_counts):
    ax[1].text(i, v + 150, str(v), ha="center")

plt.tight_layout()

plt.savefig(
    results / "00_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

print("Class distribution figure saved!")

plt.show()
