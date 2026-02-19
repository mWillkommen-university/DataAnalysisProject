#!/usr/bin/env python3
"""
train_binary_severe_with_risk.py
Binary severity model with feature engineering:
Severe = fatal OR serious, Non-severe = slight

Adds:
- 5 base binary risk features
- 4 composite risk indicators

Trains:
1) Baseline RandomForest with class_weight
2) SMOTE + RandomForest (default threshold)
3) SMOTE + threshold override optimized for recall with a precision floor

Outputs in ./model_outputs_binary_risk:
- Models:
  - severe_baseline_classweight_rf.joblib
  - severe_smote_rf.joblib
- Tables:
  - severe_metrics_summary.csv
  - severe_metrics_summary.tex
  - severe_threshold_tradeoff.csv
- PDFs (each plot separate):
  - severe_class_distribution.pdf
  - severe_confusion_baseline.pdf
  - severe_confusion_smote_default.pdf
  - severe_confusion_smote_thresholded.pdf
  - severe_confusion_compare_baseline_vs_thresholded.pdf
  - severe_pr_curve_compare.pdf
  - severe_roc_curve_compare.pdf
  - severe_threshold_optimization.pdf
  - severe_feature_importance_compare.pdf

Requirements:
- pandas, numpy, matplotlib, scikit-learn, joblib
- imbalanced-learn (SMOTE)
  conda install -c conda-forge imbalanced-learn
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# --- SMOTE imports ---
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Missing dependency: imbalanced-learn.\n"
        "Install with:\n"
        "  conda install -c conda-forge imbalanced-learn\n"
        "or\n"
        "  pip install imbalanced-learn"
    )

# =======================
# Config
# =======================
CSV_PATH = "/Users/marcelwillkommen/Coding/DataForAI/DataAnalysisProject/DataAnalysisProject/data/processed/dataset-20251215_2.csv"
TARGET_ORIG = "Accident_severity"
SEVERE_LABELS = ["fatal", "serious"]  # confirmed in your dataset
NONSEVERE_LABEL = "slight"

BASE_FEATURES = [
    "Time",
    "Day_of_week",
    "Age_band_of_driver",
    "Vehicle_driver_relation",
    "Driving_experience",
    "Type_of_vehicle",
    "Owner_of_vehicle",
    "Area_accident_occured",
    "Lanes_or_Medians",
    "Road_allignment",
    "Types_of_Junction",
    "Road_surface_type",
    "Road_surface_conditions",
    "Light_conditions",
    "Weather_conditions",
    "Cause_of_accident",
]

OUT_DIR = Path("model_outputs_binary_risk")
OUT_DIR.mkdir(exist_ok=True)

PRECISION_FLOOR = 0.20  # policy choice for threshold optimization

# =======================
# Helpers
# =======================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: trim spaces + remove leading '# ' (e.g. '# Time' -> 'Time')."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"^#\s*", "", regex=True)
    )
    return df

def save_pdf(path: Path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def aggregate_importance(pipe, feature_cols):
    """Aggregate one-hot importances back to original columns for interpretability."""
    ohe = pipe.named_steps["preprocessing"].named_transformers_["cat"].named_steps["onehot"]
    fnames = ohe.get_feature_names_out(feature_cols)
    imps = pipe.named_steps["model"].feature_importances_

    agg = {c: 0.0 for c in feature_cols}
    for name, imp in zip(fnames, imps):
        base = name.split("_", 1)[0]
        if base in agg:
            agg[base] += float(imp)
        else:
            # fallback prefix match
            for col in feature_cols:
                if name.startswith(col + "_"):
                    agg[col] += float(imp)
                    break
    return pd.Series(agg).sort_values(ascending=False)

def summarize_binary(y_true, y_pred, y_proba, name: str):
    """Binary metrics for severe(1) vs non-severe(0)."""
    return {
        "model": name,
        "precision_severe": precision_score(y_true, y_pred, zero_division=0),
        "recall_severe": recall_score(y_true, y_pred, zero_division=0),
        "f1_severe": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "accuracy": float((y_true == y_pred).mean()),
        "support_severe": int(np.sum(y_true)),
        "support_total": int(len(y_true)),
    }

# =======================
# Load
# =======================
df = normalize_headers(pd.read_csv(CSV_PATH))

missing = [c for c in BASE_FEATURES + [TARGET_ORIG] if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}\nAvailable columns:\n{df.columns.tolist()}")

# =======================
# Binary target
# =======================
df["Severe_accident"] = df[TARGET_ORIG].isin(SEVERE_LABELS).astype(int)

# =======================
# Feature engineering (AS GIVEN)
# Base binary risk features
# -----------------------------
df["risk_night"] = df["Light_conditions"].isin(
    ["darkness", "lights"]
)

df["risk_wet"] = df["Weather_conditions"].isin(
    ["raining", "snow", "fog"]
)

df["risk_speed"] = df["Cause_of_accident"].eq("speeding")

df["risk_young_driver"] = df["Age_band_of_driver"].isin(["young", "minor"])

df["risk_unlit_road"] = df["Light_conditions"].eq(
    "darkness"
)

# -----------------------------
# Define composite risk indicators
# -----------------------------
df["risk_speed_wet"] = df["risk_speed"] & df["risk_wet"]
df["risk_night_rain"] = df["risk_night"] & df["risk_wet"]
df["risk_young_night"] = df["risk_young_driver"] & df["risk_night"]
df["risk_night_rain_unlit"] = df["risk_night"] & df["risk_wet"] & df["risk_unlit_road"]

RISK_FEATURES = [
    "risk_night",
    "risk_wet",
    "risk_speed",
    "risk_young_driver",
    "risk_unlit_road",
    "risk_speed_wet",
    "risk_night_rain",
    "risk_young_night",
    "risk_night_rain_unlit",
]

# Convert booleans to int (0/1) for modeling
for col in RISK_FEATURES:
    df[col] = df[col].astype(int)

FEATURES_EXTENDED = BASE_FEATURES + RISK_FEATURES

# =======================
# Split
# =======================
X = df[FEATURES_EXTENDED].copy()
y = df["Severe_accident"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =======================
# Preprocessing
# - Base features are categorical -> one-hot
# - Risk features are numeric binary -> pass through (impute if needed)
# =======================
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipe, BASE_FEATURES),
        ("num", num_pipe, RISK_FEATURES),
    ],
    remainder="drop",
)

# =======================
# Models
# =======================
baseline_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight={0: 1, 1: 4},   # same idea as before
    min_samples_leaf=2,
)

smote_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight=None,
    min_samples_leaf=2,
)

baseline_pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", baseline_model),
])

smote_pipe = ImbPipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", smote_model),
])

# =======================
# Train
# =======================
baseline_pipe.fit(X_train, y_train)
smote_pipe.fit(X_train, y_train)

# =======================
# Predict + probabilities
# =======================
# Baseline
y_pred_base = baseline_pipe.predict(X_test)
y_proba_base = baseline_pipe.predict_proba(X_test)[:, 1]  # P(severe)

# SMOTE default
y_pred_sm_default = smote_pipe.predict(X_test)            # threshold 0.5 internally
y_proba_sm = smote_pipe.predict_proba(X_test)[:, 1]       # P(severe)

# =======================
# Threshold optimization on SMOTE probabilities
# =======================
thresholds = np.linspace(0.01, 0.90, 90)
rows = []
for t in thresholds:
    pred = (y_proba_sm >= t).astype(int)
    rows.append({
        "threshold": float(t),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "predicted_severe_rate": float(pred.mean()),
    })

thr_df = pd.DataFrame(rows)
thr_df.to_csv(OUT_DIR / "severe_threshold_tradeoff.csv", index=False)

# Choose best threshold: maximize recall subject to precision floor
cand = thr_df[thr_df["precision"] >= PRECISION_FLOOR].sort_values("recall", ascending=False).head(1)
if len(cand) == 0:
    best_t = float(thr_df.sort_values("f1", ascending=False).iloc[0]["threshold"])
    print(f"\nNo threshold met precision >= {PRECISION_FLOOR:.2f}; fallback to best F1.")
else:
    best_t = float(cand.iloc[0]["threshold"])

print(f"\nChosen severe threshold t = {best_t:.3f} (precision floor = {PRECISION_FLOOR:.2f})")

# Apply thresholded prediction
y_pred_sm_thresholded = (y_proba_sm >= best_t).astype(int)

# =======================
# Print reports
# =======================
print("\n=== Baseline (class_weight) ===")
print(classification_report(y_test, y_pred_base, zero_division=0, target_names=["non-severe", "severe"]))

print("\n=== SMOTE (default threshold) ===")
print(classification_report(y_test, y_pred_sm_default, zero_division=0, target_names=["non-severe", "severe"]))

print("\n=== SMOTE + severe-threshold override ===")
print(classification_report(y_test, y_pred_sm_thresholded, zero_division=0, target_names=["non-severe", "severe"]))

# =======================
# Save table-ready metrics (CSV + LaTeX)
# =======================
metrics_rows = [
    summarize_binary(y_test, y_pred_base, y_proba_base, "Baseline (class_weight)"),
    summarize_binary(y_test, y_pred_sm_default, y_proba_sm, "SMOTE (default threshold)"),
    summarize_binary(y_test, y_pred_sm_thresholded, y_proba_sm, "SMOTE + threshold override"),
]
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / "severe_metrics_summary.csv", index=False)
with open(OUT_DIR / "severe_metrics_summary.tex", "w", encoding="utf-8") as f:
    f.write(metrics_df.to_latex(index=False, float_format="%.3f"))

# =======================
# Plots (each plot -> separate PDF)
# =======================

# Plot 1: binary class distribution (test set)
plt.figure(figsize=(6, 4))
test_dist = y_test.value_counts(normalize=True) * 100
test_dist = test_dist.rename(index={0: "non-severe (slight)", 1: "severe (fatal+serious)"})
test_dist.plot(kind="bar")
plt.ylabel("Percentage (%)")
plt.title("Binary Class Distribution (Test set)")
save_pdf(OUT_DIR / "severe_class_distribution.pdf")

# Plot 2/3/4: confusion matrices (baseline, SMOTE default, SMOTE thresholded)
labels = ["non-severe", "severe"]

cm_base = confusion_matrix(y_test, y_pred_base, labels=[0, 1])
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cm_base, display_labels=labels).plot(values_format="d")
plt.title("Confusion Matrix — Baseline (class_weight)")
save_pdf(OUT_DIR / "severe_confusion_baseline.pdf")

cm_sm_def = confusion_matrix(y_test, y_pred_sm_default, labels=[0, 1])
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cm_sm_def, display_labels=labels).plot(values_format="d")
plt.title("Confusion Matrix — SMOTE (default threshold)")
save_pdf(OUT_DIR / "severe_confusion_smote_default.pdf")

cm_sm_thr = confusion_matrix(y_test, y_pred_sm_thresholded, labels=[0, 1])
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cm_sm_thr, display_labels=labels).plot(values_format="d")
plt.title("Confusion Matrix — SMOTE + Threshold override")
save_pdf(OUT_DIR / "severe_confusion_smote_thresholded.pdf")

# Plot 5: confusion compare (baseline vs SMOTE thresholded) side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrix Comparison (Counts)")

for ax, cm, title in [(axes[0], cm_base, "Baseline"), (axes[1], cm_sm_thr, "SMOTE + Threshold")]:
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

save_pdf(OUT_DIR / "severe_confusion_compare_baseline_vs_thresholded.pdf")

# Plot 6: PR curves compare (baseline vs SMOTE)
p_b, r_b, _ = precision_recall_curve(y_test, y_proba_base)
p_s, r_s, _ = precision_recall_curve(y_test, y_proba_sm)

ap_b = average_precision_score(y_test, y_proba_base)
ap_s = average_precision_score(y_test, y_proba_sm)

plt.figure(figsize=(7, 5))
plt.plot(r_b, p_b, label=f"Baseline (AP={ap_b:.3f})")
plt.plot(r_s, p_s, label=f"SMOTE (AP={ap_s:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Severe = fatal+serious)")
plt.legend()
save_pdf(OUT_DIR / "severe_pr_curve_compare.pdf")

# Plot 7: ROC curves compare (baseline vs SMOTE)
fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_base)
fpr_s, tpr_s, _ = roc_curve(y_test, y_proba_sm)

auc_b = roc_auc_score(y_test, y_proba_base)
auc_s = roc_auc_score(y_test, y_proba_sm)

plt.figure(figsize=(7, 5))
plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC={auc_b:.3f})")
plt.plot(fpr_s, tpr_s, label=f"SMOTE (AUC={auc_s:.3f})")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Severe = fatal+serious)")
plt.legend()
save_pdf(OUT_DIR / "severe_roc_curve_compare.pdf")

# Plot 8: threshold optimization plot
plt.figure(figsize=(7, 5))
plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
plt.axvline(best_t, linestyle="--", label=f"Chosen t={best_t:.3f}")
plt.xlabel("Severe probability threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization (SMOTE) — Severe accidents")
plt.legend()
save_pdf(OUT_DIR / "severe_threshold_optimization.pdf")

# Plot 9: feature importance compare (aggregated) — baseline vs SMOTE
imp_base = aggregate_importance(baseline_pipe, BASE_FEATURES)  # aggregated only for categorical base features
imp_sm = aggregate_importance(smote_pipe, BASE_FEATURES)

# Add risk features to importance plot using RF direct importances:
# We'll show two panels:
# - Aggregated base feature importance
# - Risk feature importances from RF (direct columns, no one-hot)

# Extract risk importances:
# The transformed matrix is [onehot(base features) | risk features], so risk importances are at the end.
ohe = baseline_pipe.named_steps["preprocessing"].named_transformers_["cat"].named_steps["onehot"]
n_ohe = len(ohe.get_feature_names_out(BASE_FEATURES))

rf_base_imp = baseline_pipe.named_steps["model"].feature_importances_
rf_sm_imp = smote_pipe.named_steps["model"].feature_importances_

risk_imp_base = pd.Series(rf_base_imp[n_ohe:], index=RISK_FEATURES).sort_values(ascending=False)
risk_imp_sm = pd.Series(rf_sm_imp[n_ohe:], index=RISK_FEATURES).sort_values(ascending=False)

topn_base = 10
topn_risk = 9  # all risk features

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Feature Importance Comparison")

# Base (aggregated) - Baseline
imp_base.head(topn_base).sort_values().plot(kind="barh", ax=axes[0, 0])
axes[0, 0].set_title("Base Features (Aggregated) — Baseline")
axes[0, 0].set_xlabel("Importance")

# Base (aggregated) - SMOTE
imp_sm.head(topn_base).sort_values().plot(kind="barh", ax=axes[0, 1])
axes[0, 1].set_title("Base Features (Aggregated) — SMOTE")
axes[0, 1].set_xlabel("Importance")

# Risk features - Baseline
risk_imp_base.head(topn_risk).sort_values().plot(kind="barh", ax=axes[1, 0])
axes[1, 0].set_title("Risk Features — Baseline")
axes[1, 0].set_xlabel("Importance")

# Risk features - SMOTE
risk_imp_sm.head(topn_risk).sort_values().plot(kind="barh", ax=axes[1, 1])
axes[1, 1].set_title("Risk Features — SMOTE")
axes[1, 1].set_xlabel("Importance")

save_pdf(OUT_DIR / "severe_feature_importance_compare.pdf")

# =======================
# Save models
# =======================
joblib.dump(baseline_pipe, OUT_DIR / "severe_baseline_classweight_rf.joblib")
joblib.dump(smote_pipe, OUT_DIR / "severe_smote_rf.joblib")

print("\nDone. Outputs saved to:", OUT_DIR.resolve())
print("Tables:", (OUT_DIR / "severe_metrics_summary.csv").resolve(), "and .tex")
print("Threshold tradeoff:", (OUT_DIR / "severe_threshold_tradeoff.csv").resolve())
