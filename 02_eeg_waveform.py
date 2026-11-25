
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "EEG_data.csv")

# 2. Extract EEG signals and labels
signal_cols = [f"X{i}" for i in range(1, 179)]
X = df[signal_cols].to_numpy(dtype=float)
y = df["y"].to_numpy()

# 3. Select representative examples
seizure = X[y == 1][0]
non_seizure = X[y == 3][0]

# 4. Create sampling index
n = np.arange(X.shape[1])

# 5. Plot EEG waveforms
fig, ax = plt.subplots(
    2, 1, figsize=(10, 6),
    sharex=True, sharey=True
)

ax[0].plot(n, non_seizure, color="blue")
ax[0].set_title("Non-seizure EEG Signal")
ax[0].set_ylabel("Amplitude")
ax[0].grid(True, alpha=0.3)

ax[1].plot(n, seizure, color="red")
ax[1].set_title("Seizure EEG Signal")
ax[1].set_xlabel("Sample Index")
ax[1].set_ylabel("Amplitude")
ax[1].grid(True, alpha=0.3)

plt.tight_layout()

# 6. Save the figure
results = ROOT / "results"
results.mkdir(exist_ok=True)

plt.savefig(
    results / "01_eeg_waveforms.png",
    dpi=300,
    bbox_inches="tight"
)

print("EEG waveform saved successfully!")

# 7. Display the figure
plt.show()
