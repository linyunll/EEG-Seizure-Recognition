
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
    classification_report,
    roc_curve,
    roc_auc_score
)

# ==================================================
# 1. Project paths
# ==================================================

ROOT = Path(__file__).resolve().parent

FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

FIG_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ==================================================
# 2. Load original dataset and feature dataset
# ==================================================

raw_df = None
feature_df = None

for file in ROOT.rglob("*.csv"):

    if ".venv" in file.parts:
        continue

    df = pd.read_csv(file)

    if "y" in df.columns and "X178" in df.columns:
        raw_df = df
        print("Original dataset:", file)

    if (
        "Label" in df.columns
        and "Std" in df.columns
        and "Wavelet_D1_Energy" in df.columns
        and df.shape == (11500, 28)
    ):
        feature_df = df
        print("Feature dataset:", file)

if raw_df is None:
    raise FileNotFoundError("Original EEG dataset not found.")

if feature_df is None:
    raise FileNotFoundError("Feature dataset not found.")

X = feature_df.drop(columns=["Label"])
y = feature_df["Label"].astype(int)

# Check dataset consistency
assert len(raw_df) == len(feature_df)

raw_labels = (raw_df["y"] == 1).astype(int)

assert np.array_equal(
    raw_labels.to_numpy(),
    y.to_numpy()
), "Dataset label order mismatch!"

assert np.isfinite(X.to_numpy()).all(), \
    "Invalid feature values found!"

print("\nFeature matrix shape:", X.shape)
print("Label shape:", y.shape)

# ==================================================
# 3. Extract recording group identifiers
# ==================================================

groups = raw_df.iloc[:, 0].astype(str).str.extract(
    r"^X(\d+)\.(.+)$"
)[1]

assert groups.notna().all()

group_df = pd.DataFrame({
    "Group": groups,
    "Label": y
})

assert group_df.groupby("Group").size().eq(23).all()
assert group_df.groupby("Group")["Label"].nunique().eq(1).all()

unique_groups = group_df.drop_duplicates(
    subset="Group"
)

# ==================================================
# 4. Stratified group-level data splitting
# ==================================================

train_groups, test_groups = train_test_split(
    unique_groups["Group"],
    test_size=0.2,
    random_state=42,
    stratify=unique_groups["Label"]
)

train_mask = groups.isin(train_groups)
test_mask = groups.isin(test_groups)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

assert set(train_groups).isdisjoint(set(test_groups))

print("\n" + "=" * 50)
print("Dataset Splitting Results")
print("=" * 50)

print("Training groups:", len(train_groups))
print("Testing groups:", len(test_groups))

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())

print("\nGroup overlap:",
      len(set(train_groups) & set(test_groups)))

# Save dataset splitting information
split_df = pd.DataFrame({
    "Group": groups,
    "Label": y,
    "Split": np.where(train_mask, "Train", "Test")
})

split_df.to_csv(
    RESULT_DIR / "dataset_split.csv",
    index=False
)

# ==================================================
# 5. Define machine learning models
# ==================================================

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    ))
])

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

models = {
    "SVM": svm_model,
    "Random Forest": rf_model
}

results = []
confusion_matrices = {}
roc_results = {}

# ==================================================
# 6. Train and evaluate models
# ==================================================

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if name == "SVM":
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    specificity = tn / (tn + fp)

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_test, y_score)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "AUC": auc
    })

    confusion_matrices[name] = cm

    fpr, tpr, _ = roc_curve(y_test, y_score)

    roc_results[name] = (fpr, tpr, auc)

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Non-seizure", "Seizure"],
        digits=4,
        zero_division=0
    ))

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("Specificity:", round(specificity, 4))
    print("F1-score:", round(f1, 4))
    print("AUC:", round(auc, 4))

    # Save trained model
    model_filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        MODEL_DIR / f"{model_filename}_model.pkl"
    )

# ==================================================
# 7. Save performance results
# ==================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("Model Performance Comparison")
print("=" * 50)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    RESULT_DIR / "model_performance.csv",
    index=False
)

# ==================================================
# 8. Plot confusion matrices
# ==================================================

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5)
)

for ax, (name, cm) in zip(
    axes, confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-seizure", "Seizure"],
        yticklabels=["Non-seizure", "Seizure"],
        ax=ax
    )

    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()

plt.savefig(
    FIG_DIR / "09_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

# ==================================================
# 9. Plot ROC curves
# ==================================================

plt.figure(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_results.items():

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.xlim(0, 1)
plt.ylim(0, 1.02)

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "10_roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

# ==================================================
# 10. Plot model performance comparison
# ==================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1-score",
    "AUC"
]

positions = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))

for i, row in results_df.iterrows():

    values = row[metrics].astype(float).to_numpy()

    bars = plt.bar(
        positions + (i - 0.5) * width,
        values,
        width,
        label=row["Model"]
    )

    plt.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9
    )

plt.xticks(positions, metrics)
plt.ylabel("Score")
plt.title("Machine Learning Performance Comparison")
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "11_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# ==================================================
# 11. Random Forest feature importance
# ==================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    RESULT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 10 important features:")

print(
    importance_df.head(10).to_string(index=False)
)

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    color="royalblue"
)

plt.gca().invert_yaxis()

plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "12_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

# ==================================================
# 12. Finish
# ==================================================

print("\n" + "=" * 50)
print("Machine learning completed successfully!")
print("Models, results and figures saved.")
print("=" * 50)

plt.show()
