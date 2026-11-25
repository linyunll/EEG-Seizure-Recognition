
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
    cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score


# ============================================================
# 1. Basic settings
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figures"

RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("Multi-domain Feature Comparison")
print("=" * 60)


# ============================================================
# 2. Load datasets
# ============================================================

feature_path = RESULT_DIR / "eeg_features.csv"

raw_candidates = [
    p for p in BASE_DIR.rglob("EEG_data.csv")
    if ".venv" not in p.parts
]

if len(raw_candidates) == 0:
    raise FileNotFoundError("EEG_data.csv not found!")

if len(raw_candidates) > 1:
    raise ValueError("Multiple EEG_data.csv files found!")

raw_path = raw_candidates[0]

raw_df = pd.read_csv(raw_path)
feature_df = pd.read_csv(feature_path)

print("Raw dataset file:", raw_path)
print("Raw dataset shape:", raw_df.shape)
print("Feature dataset shape:", feature_df.shape)

if len(raw_df) != len(feature_df):
    raise ValueError("Dataset sample numbers do not match!")

X = feature_df.drop(columns=["Label"])
y = feature_df["Label"].astype(int).to_numpy()

raw_labels = (raw_df["y"].astype(int) == 1).astype(int).to_numpy()

if not np.array_equal(raw_labels, y):
    raise ValueError("Dataset label order does not match!")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Invalid feature values found!")

print("Number of features:", X.shape[1])
print("Label consistency check: Passed")


# ============================================================
# 3. Extract recording groups (FIXED)
# ============================================================

sample_ids = raw_df.iloc[:, 0].astype(str).str.strip()

groups = sample_ids.str.extract(
    r"^X(\d+)\.(.+)$"
)[1]

unmatched = groups.isna().sum()

print("\n" + "=" * 60)
print("Dataset Group Check")
print("=" * 60)

print("Unmatched identifiers:", unmatched)

if unmatched > 0:
    print("Unmatched examples:")
    print(sample_ids[groups.isna()].head(10))
    raise ValueError("Unmatched sample identifiers!")

print("Unique groups:", groups.nunique())

group_df = pd.DataFrame({
    "Group": groups.to_numpy(),
    "Label": y
})

group_sizes = group_df.groupby("Group").size()
group_label_counts = group_df.groupby("Group")["Label"].nunique()

print("Minimum group size:", group_sizes.min())
print("Maximum group size:", group_sizes.max())

print(
    "Groups containing multiple labels:",
    (group_label_counts > 1).sum()
)

if groups.nunique() != 500:
    raise ValueError("Unexpected number of groups!")

if not (group_sizes == 23).all():
    raise ValueError("Unexpected group size!")

if (group_label_counts > 1).any():
    raise ValueError("Inconsistent labels within groups!")

print("Group check: Passed")

groups = groups.to_numpy()


# ============================================================
# 4. Group-based train/test split
# ============================================================

unique_groups = group_df.drop_duplicates(
    subset="Group"
).reset_index(drop=True)

train_groups, test_groups = train_test_split(
    unique_groups["Group"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=unique_groups["Label"]
)

train_mask = np.isin(groups, train_groups)
test_mask = np.isin(groups, test_groups)

X_train = X.loc[train_mask].reset_index(drop=True)
X_test = X.loc[test_mask].reset_index(drop=True)

y_train = y[train_mask]
y_test = y[test_mask]

groups_train = groups[train_mask]
groups_test = groups[test_mask]

overlap = set(groups_train) & set(groups_test)

if len(overlap) != 0:
    raise ValueError("Group leakage detected!")

print("\n" + "=" * 60)
print("Train/Test Split")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("Training groups:", len(set(groups_train)))
print("Testing groups:", len(set(groups_test)))

print("Overlapping groups:", len(overlap))

print("Training class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("Testing class distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# ============================================================
# 5. Define feature groups
# ============================================================

time_features = [
    "Mean",
    "Std",
    "Variance",
    "Maximum",
    "Minimum",
    "Peak_to_Peak",
    "RMS",
    "Energy",
    "Skewness",
    "Kurtosis"
]

frequency_features = [
    "Delta_Power",
    "Delta_Relative",
    "Theta_Power",
    "Theta_Relative",
    "Alpha_Power",
    "Alpha_Relative",
    "Beta_Power",
    "Beta_Relative",
    "Gamma_Power",
    "Gamma_Relative",
    "Dominant_Frequency",
    "Total_Power"
]

wavelet_features = [
    "Wavelet_A4_Energy",
    "Wavelet_D4_Energy",
    "Wavelet_D3_Energy",
    "Wavelet_D2_Energy",
    "Wavelet_D1_Energy"
]

feature_sets = {
    "Time": time_features,

    "Frequency": frequency_features,

    "Wavelet": wavelet_features,

    "Time+Frequency":
        time_features + frequency_features,

    "Time+Wavelet":
        time_features + wavelet_features,

    "Frequency+Wavelet":
        frequency_features + wavelet_features,

    "All Features":
        time_features + frequency_features + wavelet_features
}

print("\n" + "=" * 60)
print("Feature Combinations")
print("=" * 60)

for name, columns in feature_sets.items():

    missing = [
        col for col in columns
        if col not in X.columns
    ]

    if missing:
        raise ValueError(
            f"Missing features in {name}: {missing}"
        )

    print(f"{name}: {len(columns)} features")


# ============================================================
# 6. Define machine learning models
# ============================================================

models = {

    "SVM": Pipeline([
        ("scaler", StandardScaler()),

        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# 7. Five-fold stratified group cross-validation
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_splits = list(
    cv.split(
        X_train,
        y_train,
        groups=groups_train
    )
)

print("\n" + "=" * 60)
print("Cross-validation Group Check")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(
    cv_splits, start=1
):

    tr_groups = set(groups_train[tr_idx])
    val_groups = set(groups_train[val_idx])

    fold_overlap = tr_groups & val_groups

    if fold_overlap:
        raise ValueError("Cross-validation group leakage!")

    if len(np.unique(y_train[val_idx])) != 2:
        raise ValueError("Validation fold missing a class!")

    print(
        f"Fold {fold}: "
        f"Train={len(tr_idx)}, "
        f"Validation={len(val_idx)}, "
        f"Group overlap={len(fold_overlap)}"
    )


# ============================================================
# 8. Evaluation metrics
# ============================================================

scoring = {
    "Accuracy": "accuracy",

    "Precision": make_scorer(
        __import__("sklearn.metrics", fromlist=["precision_score"])
        .precision_score,
        zero_division=0
    ),

    "Recall": make_scorer(
        recall_score,
        pos_label=1,
        zero_division=0
    ),

    "Specificity": make_scorer(
        recall_score,
        pos_label=0,
        zero_division=0
    ),

    "F1": "f1",

    "AUC": "roc_auc"
}

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUC"
]


# ============================================================
# 9. Compare seven feature combinations
# ============================================================

results = []
fold_results = []

total_experiments = len(feature_sets) * len(models)
experiment_index = 0

print("\n" + "=" * 60)
print("Start Feature Comparison")
print("=" * 60)

for feature_name, columns in feature_sets.items():

    X_selected = X_train[columns]

    for model_name, model in models.items():

        experiment_index += 1

        print(
            f"\n[{experiment_index}/{total_experiments}] "
            f"{feature_name} + {model_name}",
            flush=True
        )

        scores = cross_validate(
            model,
            X_selected,
            y_train,
            cv=cv_splits,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise"
        )

        row = {
            "Feature_Set": feature_name,
            "Model": model_name,
            "Number_of_Features": len(columns)
        }

        for metric in metric_names:

            values = scores[f"test_{metric}"]

            row[metric] = np.mean(values)
            row[f"{metric}_Std"] = np.std(values)

        results.append(row)

        for fold in range(5):

            fold_row = {
                "Feature_Set": feature_name,
                "Model": model_name,
                "Fold": fold + 1
            }

            for metric in metric_names:
                fold_row[metric] = scores[
                    f"test_{metric}"
                ][fold]

            fold_results.append(fold_row)

        print(
            f"Accuracy: {row['Accuracy']:.4f} "
            f"+/- {row['Accuracy_Std']:.4f}"
        )

        print(f"Precision:   {row['Precision']:.4f}")
        print(f"Recall:      {row['Recall']:.4f}")
        print(f"Specificity: {row['Specificity']:.4f}")
        print(f"F1-score:    {row['F1']:.4f}")
        print(f"AUC:         {row['AUC']:.4f}")


# ============================================================
# 10. Save comparison results
# ============================================================

results_df = pd.DataFrame(results)
fold_results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    RESULT_DIR / "feature_comparison_results.csv",
    index=False
)

fold_results_df.to_csv(
    RESULT_DIR / "feature_comparison_fold_results.csv",
    index=False
)

print("\n" + "=" * 60)
print("Final Feature Comparison Results")
print("=" * 60)

display_columns = [
    "Feature_Set",
    "Model",
    "Number_of_Features",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUC"
]

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\nBest feature combination for each model:")

for model_name in models:

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    best = model_results.loc[
        model_results["F1"].idxmax()
    ]

    print(
        f"{model_name}: "
        f"{best['Feature_Set']}, "
        f"F1={best['F1']:.4f}, "
        f"Accuracy={best['Accuracy']:.4f}"
    )


# ============================================================
# 11. Figure 1: Accuracy comparison
# ============================================================

feature_names = list(feature_sets.keys())

positions = np.arange(len(feature_names))
width = 0.35

colors = {
    "SVM": "royalblue",
    "Random Forest": "tomato"
}

fig1, ax1 = plt.subplots(figsize=(13, 6))

for i, model_name in enumerate(models):

    data = results_df[
        results_df["Model"] == model_name
    ].set_index("Feature_Set").loc[feature_names]

    offset = (i - 0.5) * width

    bars = ax1.bar(
        positions + offset,
        data["Accuracy"],
        width,
        yerr=data["Accuracy_Std"],
        capsize=3,
        label=model_name,
        color=colors[model_name],
        alpha=0.9
    )

    for bar, value in zip(bars, data["Accuracy"]):

        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=9
        )

ax1.set_title("Classification Accuracy of Feature Combinations")
ax1.set_ylabel("Cross-validation Accuracy")
ax1.set_xlabel("Feature Combination")

ax1.set_xticks(positions)
ax1.set_xticklabels(feature_names, rotation=15)

ax1.set_ylim(0, 1.12)
ax1.legend(loc="lower right")
ax1.grid(axis="y", alpha=0.3)

fig1.tight_layout()

fig1.savefig(
    FIGURE_DIR / "08_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 12. Figure 2: F1-score comparison
# ============================================================

fig2, ax2 = plt.subplots(figsize=(13, 6))

for i, model_name in enumerate(models):

    data = results_df[
        results_df["Model"] == model_name
    ].set_index("Feature_Set").loc[feature_names]

    offset = (i - 0.5) * width

    bars = ax2.bar(
        positions + offset,
        data["F1"],
        width,
        yerr=data["F1_Std"],
        capsize=3,
        label=model_name,
        color=colors[model_name],
        alpha=0.9
    )

    for bar, value in zip(bars, data["F1"]):

        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            fontsize=9
        )

ax2.set_title("F1-score of Feature Combinations")
ax2.set_ylabel("Cross-validation F1-score")
ax2.set_xlabel("Feature Combination")

ax2.set_xticks(positions)
ax2.set_xticklabels(feature_names, rotation=15)

ax2.set_ylim(0, 1.12)
ax2.legend(loc="lower right")
ax2.grid(axis="y", alpha=0.3)

fig2.tight_layout()

fig2.savefig(
    FIGURE_DIR / "08_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 13. Figure 3: Performance heatmaps
# ============================================================

fig3, axes = plt.subplots(
    1, 2,
    figsize=(16, 7)
)

for ax, model_name in zip(axes, models):

    data = results_df[
        results_df["Model"] == model_name
    ].set_index("Feature_Set").loc[feature_names]

    matrix = data[metric_names].to_numpy()

    im = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_title(model_name)

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=30)

    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)

    for i in range(matrix.shape[0]):

        for j in range(matrix.shape[1]):

            value = matrix[i, j]

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.65 else "black",
                fontsize=10
            )

fig3.suptitle(
    "Multi-domain Feature Classification Performance",
    fontsize=16
)

fig3.subplots_adjust(
    left=0.12,
    right=0.90,
    bottom=0.15,
    top=0.88,
    wspace=0.45
)

cbar_ax = fig3.add_axes([0.92, 0.18, 0.015, 0.65])
fig3.colorbar(im, cax=cbar_ax, label="Score")

fig3.savefig(
    FIGURE_DIR / "08_performance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 14. Figure 4: Feature dimension vs performance
# ============================================================

fig4, ax4 = plt.subplots(figsize=(12, 7))

for model_name in models:

    data = results_df[
        results_df["Model"] == model_name
    ]

    ax4.scatter(
        data["Number_of_Features"],
        data["Accuracy"],
        s=100,
        label=model_name,
        color=colors[model_name],
        alpha=0.85
    )

    for _, row in data.iterrows():

        offset = (
            (5, 10)
            if model_name == "SVM"
            else (5, -15)
        )

        ax4.annotate(
            row["Feature_Set"],
            (
                row["Number_of_Features"],
                row["Accuracy"]
            ),
            textcoords="offset points",
            xytext=offset,
            fontsize=8
        )

ax4.set_title("Feature Dimension vs Classification Accuracy")
ax4.set_xlabel("Number of Features")
ax4.set_ylabel("Cross-validation Accuracy")

ax4.set_xticks([5, 10, 12, 15, 17, 22, 27])
ax4.set_xlim(3, 33)

lower_limit = max(
    0,
    results_df["Accuracy"].min() - 0.03
)

ax4.set_ylim(lower_limit, 1.025)

ax4.legend(loc="lower right")
ax4.grid(alpha=0.3)

fig4.tight_layout()

fig4.savefig(
    FIGURE_DIR / "08_dimension_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. Finish
# ============================================================

print("\n" + "=" * 60)
print("Feature comparison completed successfully!")
print("=" * 60)

print("Result files:")
print(RESULT_DIR / "feature_comparison_results.csv")
print(RESULT_DIR / "feature_comparison_fold_results.csv")

print("\nFigure files:")
print(FIGURE_DIR / "08_accuracy_comparison.png")
print(FIGURE_DIR / "08_f1_comparison.png")
print(FIGURE_DIR / "08_performance_heatmap.png")
print(FIGURE_DIR / "08_dimension_comparison.png")

plt.show()
