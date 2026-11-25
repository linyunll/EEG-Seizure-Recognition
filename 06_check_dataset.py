
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

# 1. Find the original EEG dataset
csv_files = list((ROOT / "data").rglob("*.csv"))

df = None

for file in csv_files:
    temp = pd.read_csv(file)

    if "y" in temp.columns and "X178" in temp.columns:
        df = temp
        print("Dataset file:", file.name)
        break

if df is None:
    raise FileNotFoundError("Original EEG dataset not found.")

# 2. Check sample identifiers
sample_ids = df.iloc[:, 0].astype(str)

print("\n" + "=" * 50)
print("Dataset Group Analysis")
print("=" * 50)

print("\nTotal samples:", len(df))

print("\nFirst 20 sample identifiers:")
print(sample_ids.head(20).to_string(index=False))

# 3. Extract candidate recording identifiers
parts = sample_ids.str.extract(
    r"^X(\d+)\.(.+)$"
)

segment_ids = parts[0]
group_ids = parts[1]

print("\nUnmatched identifiers:",
      group_ids.isna().sum())

print("\nUnique candidate groups:",
      group_ids.nunique())

# 4. Check group size
group_sizes = group_ids.value_counts()

print("\nGroup size statistics:")
print(group_sizes.describe())

print("\nGroup size distribution:")
print(group_sizes.value_counts().sort_index())

# 5. Check labels within each group
check_df = pd.DataFrame({
    "Sample_ID": sample_ids,
    "Segment_ID": segment_ids,
    "Group_ID": group_ids,
    "Original_Label": df["y"]
})

label_counts = check_df.groupby(
    "Group_ID"
)["Original_Label"].nunique()

print("\nGroups containing multiple labels:",
      (label_counts > 1).sum())

print("\nFirst 10 candidate groups:")
print(group_sizes.head(10))

print("\nDataset group analysis completed!")
