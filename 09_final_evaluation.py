
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)

# ============================================================
# 1. Basic settings
# ============================================================

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figures"

RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False

print("=" * 60)
print("Final EEG Classification Evaluation")
print("=" * 60)

# ============================================================
# 2. Load datasets
# ============================================================

raw_path = DATA_DIR / "EEG_data.csv"
feature_path = RESULT_DIR / "eeg_features.csv"

raw_df = pd.read_csv(raw_path)
feature_df = pd.read_csv(feature_path)

print("Raw dataset shape:", raw_df.shape)
print("Feature dataset shape:", feature_df.shape)

if len(raw_df) != len(feature_df):
    raise ValueError("Dataset sample numbers do not match!")

raw_labels = (raw_df["y"].to_numpy() == 1).astype(int)
y = feature_df["Label"].to_numpy().astype(int)

if not np.array_equal(raw_labels, y):
    raise ValueError("Dataset labels do not match!")

print("Label consistency check: Passed")

# ============================================================
# 3. Recover recording groups
# ============================================================

# Use the same identifier pattern as Step 08
pattern = r"^X(\d+)\.(.+)$"

id_column = None

for col in raw_df.columns:

    candidate = raw_df[col].astype(str).str.strip()

    if candidate.str.fullmatch(pattern).all():
        id_column = col
        break

if id_column is None:
    raise ValueError(
        "Cannot locate sample identifier column!"
    )

identifiers = (
    raw_df[id_column]
    .astype(str)
    .str.strip()
)

extracted = identifiers.str.extract(pattern)

# The second part is the recording group identifier
groups = extracted[1]

unmatched = groups.isna().sum()

if unmatched > 0:
    raise ValueError("Unmatched sample identifiers!")

group_df = pd.DataFrame({
    "Group": groups.to_numpy(),
    "Label": y
})

group_sizes = group_df.groupby("Group").size()

mixed_groups = (
    group_df.groupby("Group")["Label"].nunique() > 1
).sum()

print("\n" + "=" * 60)
print("Dataset Group Check")
print("=" * 60)

print("Identifier column:", id_column)
print("Unmatched identifiers:", unmatched)
print("Unique groups:", groups.nunique())
print("Minimum group size:", group_sizes.min())
print("Maximum group size:", group_sizes.max())
print("Groups containing multiple labels:", mixed_groups)

assert groups.nunique() == 500
assert (group_sizes == 23).all()
assert mixed_groups == 0

print("Group check: Passed")

# ============================================================
# 4. Group-level train/test split
# ============================================================

unique_groups = (
    group_df.groupby("Group")["Label"]
    .first()
    .reset_index()
)

train_groups, test_groups = train_test_split(
    unique_groups["Group"],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=unique_groups["Label"]
)

train_mask = groups.isin(train_groups).to_numpy()
test_mask = groups.isin(test_groups).to_numpy()

train_indices = np.flatnonzero(train_mask)
test_indices = np.flatnonzero(test_mask)

overlap = set(train_groups) & set(test_groups)

assert len(overlap) == 0
assert len(train_indices) == 9200
assert len(test_indices) == 2300
assert np.all(train_mask ^ test_mask)

y_train = y[train_indices]
y_test = y[test_indices]

print("\n" + "=" * 60)
print("Train/Test Split")
print("=" * 60)

print("Training samples:", len(train_indices))
print("Testing samples:", len(test_indices))
print("Training groups:", len(train_groups))
print("Testing groups:", len(test_groups))
print("Overlapping groups:", len(overlap))

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# Save final dataset split information
split_df = pd.DataFrame({
    "Sample_ID": identifiers,
    "Group": groups,
    "Label": y,
    "Dataset": np.where(
        train_mask,
        "Train",
        "Test"
    )
})

split_df.to_csv(
    RESULT_DIR / "09_dataset_split.csv",
    index=False
)

# ============================================================
# 5. Define feature combinations
# ============================================================

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

feature_sets = {
    "Frequency+Wavelet":
        frequency_features + wavelet_features,

    "All Features":
        time_features
        + frequency_features
        + wavelet_features
}

print("\n" + "=" * 60)
print("Selected Feature Sets")
print("=" * 60)

for name, columns in feature_sets.items():

    missing = [
        col for col in columns
        if col not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    print(f"{name}: {len(columns)} features")

# ============================================================
# 6. Define classification models
# ============================================================

def create_model(model_name):

    if model_name == "SVM":

        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ])

    elif model_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown model name!")


# ============================================================
# 7. Model evaluation function
# ============================================================

def evaluate_model(model, X_train, X_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)

    metrics = {
        "Accuracy": accuracy_score(
            y_test, y_pred
        ),

        "Precision": precision_score(
            y_test, y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test, y_pred,
            zero_division=0
        ),

        "Specificity": specificity,

        "F1": f1_score(
            y_test, y_pred,
            zero_division=0
        ),

        "AUC": roc_auc_score(
            y_test, y_score
        ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    fpr, tpr, _ = roc_curve(
        y_test,
        y_score
    )

    return metrics, cm, fpr, tpr, y_pred, y_score


# ============================================================
# 8. Train and evaluate four configurations
# ============================================================

results = []
evaluation_data = {}
prediction_tables = []

model_names = [
    "SVM",
    "Random Forest"
]

print("\n" + "=" * 60)
print("Start Final Model Evaluation")
print("=" * 60)

experiment_number = 0

for feature_name, columns in feature_sets.items():

    X = feature_df[columns].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            f"Invalid values in {feature_name}"
        )

    X_train = X[train_indices]
    X_test = X[test_indices]

    for model_name in model_names:

        experiment_number += 1

        print(
            f"\n[{experiment_number}/4] "
            f"{feature_name} + {model_name}"
        )

        model = create_model(model_name)

        metrics, cm, fpr, tpr, y_pred, y_score = (
            evaluate_model(
                model,
                X_train,
                X_test
            )
        )

        result = {
            "Feature_Set": feature_name,
            "Model": model_name,
            "Number_of_Features": len(columns)
        }

        result.update(metrics)
        results.append(result)

        evaluation_data[
            (feature_name, model_name)
        ] = {
            "Confusion_Matrix": cm,
            "FPR": fpr,
            "TPR": tpr,
            "AUC": metrics["AUC"],
            "Model": model
        }

        prediction_df = pd.DataFrame({
            "Sample_ID":
                identifiers.iloc[test_indices].to_numpy(),

            "Group":
                groups.iloc[test_indices].to_numpy(),

            "Feature_Set": feature_name,
            "Model": model_name,
            "True_Label": y_test,
            "Predicted_Label": y_pred,
            "Decision_Score": y_score
        })

        prediction_tables.append(prediction_df)

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "AUC"
        ]:

            print(
                f"{metric:12s}: "
                f"{metrics[metric]:.4f}"
            )

        print("\nConfusion matrix:")
        print(cm)

        print("\nClassification report:")

        print(classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-seizure",
                "Seizure"
            ],
            digits=4,
            zero_division=0
        ))

# ============================================================
# 9. Save evaluation results
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULT_DIR / "09_final_evaluation_results.csv",
    index=False
)

all_predictions = pd.concat(
    prediction_tables,
    ignore_index=True
)

all_predictions.to_csv(
    RESULT_DIR / "09_test_predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("Final Test Set Results")
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

# ============================================================
# 10. Figure 1: Confusion matrices
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 12),
    constrained_layout=True
)

for i, feature_name in enumerate(feature_sets):

    for j, model_name in enumerate(model_names):

        ax = axes[i, j]

        cm = evaluation_data[
            (feature_name, model_name)
        ]["Confusion_Matrix"]

        ax.imshow(
            cm,
            cmap="Blues",
            vmin=0,
            vmax=1840
        )

        threshold = cm.max() / 2

        for row in range(2):

            for col in range(2):

                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    fontsize=18,
                    color=(
                        "white"
                        if cm[row, col] > threshold
                        else "black"
                    )
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels([
            "Non-seizure",
            "Seizure"
        ])

        ax.set_yticklabels([
            "Non-seizure",
            "Seizure"
        ])

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        ax.set_title(
            f"{model_name}\n{feature_name}"
        )

fig.suptitle(
    "Final Test Set Confusion Matrices",
    fontsize=17
)


plt.savefig(
    FIGURE_DIR / "09_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================================
# 11. Figure 2: ROC curve comparison
# ============================================================

plt.figure(figsize=(10, 7))

colors = {
    "SVM": "royalblue",
    "Random Forest": "tomato"
}

line_styles = {
    "Frequency+Wavelet": "-",
    "All Features": "--"
}

for feature_name in feature_sets:

    for model_name in model_names:

        data = evaluation_data[
            (feature_name, model_name)
        ]

        plt.plot(
            data["FPR"],
            data["TPR"],
            color=colors[model_name],
            linestyle=line_styles[feature_name],
            linewidth=2,
            label=(
                f"{model_name} - {feature_name} "
                f"(AUC={data['AUC']:.4f})"
            )
        )

plt.plot(
    [0, 1],
    [0, 1],
    "k:",
    linewidth=1.5,
    label="Random Classifier"
)

plt.xlim(0, 1)
plt.ylim(0, 1.03)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Final Test Set ROC Curve Comparison")

plt.legend(
    loc="lower right",
    fontsize=10
)

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_roc_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================================
# 12. Figure 3: Classification performance comparison
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "AUC"
]

metric_labels = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1-score",
    "AUC"
]

x = np.arange(len(metric_names))
width = 0.19

plt.figure(figsize=(16, 7))

bar_colors = [
    "royalblue",
    "tomato",
    "skyblue",
    "orange"
]

for i, (_, row) in enumerate(results_df.iterrows()):

    values = [
        row[metric]
        for metric in metric_names
    ]

    offset = (i - 1.5) * width

    bars = plt.bar(
        x + offset,
        values,
        width,
        color=bar_colors[i],
        label=(
            f"{row['Model']} - "
            f"{row['Feature_Set']}"
        )
    )

    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xticks(x, metric_labels)
plt.ylim(0, 1.17)

plt.ylabel("Score")

plt.title(
    "Final Classification Performance Comparison"
)

plt.legend(
    loc="upper center",
    ncol=2,
    fontsize=10
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================================
# 13. Figure 4: Feature dimension comparison
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 6)
)

selected_metrics = [
    "Accuracy",
    "F1"
]

feature_labels = [
    "Frequency+Wavelet\n(17 features)",
    "All Features\n(27 features)"
]

x = np.arange(2)
width = 0.32

for ax, metric in zip(
    axes,
    selected_metrics
):

    for i, model_name in enumerate(model_names):

        subset = results_df[
            results_df["Model"] == model_name
        ].set_index("Feature_Set")

        values = [
            subset.loc[
                "Frequency+Wavelet",
                metric
            ],

            subset.loc[
                "All Features",
                metric
            ]
        ]

        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=model_name,
            color=colors[model_name]
        )

        for bar, value in zip(bars, values):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.01,
                f"{value:.4f}",
                ha="center",
                fontsize=11
            )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Test Set Score")
    ax.set_title(metric + " Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")

fig.suptitle(
    "17-dimensional vs 27-dimensional Features",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_feature_dimension_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================================
# 14. Figure 5: Feature importance
# ============================================================

rf_model = evaluation_data[
    ("Frequency+Wavelet", "Random Forest")
]["Model"]

selected_features = feature_sets[
    "Frequency+Wavelet"
]

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    RESULT_DIR / "09_feature_importance.csv",
    index=False
)

plot_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

plt.figure(figsize=(11, 8))

plt.barh(
    plot_df["Feature"],
    plot_df["Importance"],
    color="royalblue"
)

plt.xlabel("Feature Importance")

plt.title(
    "Random Forest Feature Importance\n"
    "Frequency + Wavelet Features"
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

# ============================================================
# 15. Final summary
# ============================================================

reduction = (27 - 17) / 27 * 100

print("\n" + "=" * 60)
print("Feature Dimension Reduction")
print("=" * 60)

print("Original feature dimension: 27")
print("Selected feature dimension: 17")
print(f"Feature reduction: {reduction:.2f}%")

print("\nPerformance of selected 17 features:")

selected_results = results_df[
    results_df["Feature_Set"]
    == "Frequency+Wavelet"
]

print(
    selected_results[display_columns]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("Final evaluation completed successfully!")
print("=" * 60)

print("\nResult files:")

for filename in [
    "09_dataset_split.csv",
    "09_final_evaluation_results.csv",
    "09_test_predictions.csv",
    "09_feature_importance.csv"
]:

    print(RESULT_DIR / filename)

print("\nFigure files:")

for filename in [
    "09_confusion_matrices.png",
    "09_roc_comparison.png",
    "09_performance_comparison.png",
    "09_feature_dimension_comparison.png",
    "09_feature_importance.png"
]:

    print(FIGURE_DIR / filename)

plt.show()
