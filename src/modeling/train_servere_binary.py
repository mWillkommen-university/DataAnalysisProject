#!/usr/bin/env python3
"""
train_severe_binary.py — Binary severity model:
Severe = (fatal OR serious), Non-severe = slight

Outputs (in ./model_outputs_severe):
Models:
- severe_baseline_classweight_rf.joblib
- severe_smote_rf.joblib

Metrics:
- severe_metrics_summary.csv
- severe_metrics_summary.tex
- severe_threshold_tradeoff.csv

PDFs (each plot its own PDF):
- severe_class_distribution.pdf
- severe_confusion_baseline.pdf
- severe_confusion_smote.pdf
- severe_confusion_compare.pdf
- severe_pr_curve_compare.pdf
- severe_roc_curve_compare.pdf
- severe_threshold_optimization.pdf
- severe_feature_importance_compare.pdf

Requirements:
- pandas, numpy, matplotlib, scikit-learn, joblib
- imbalanced-learn (SMOTE) -> conda install -c conda-forge imbalanced-learn
"""

import json
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

from sklearn.ensemble import RandomForestClassifier

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
SEVERE_LABELS = ["fatal", "serious"]  # your dataset uses these exact strings
NONSEVERE_LABEL = "slight"

FEATURES = [
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

OUT_DIR = Path("model_outputs_binary")
OUT_DIR.mkdir(exist_ok=True)

# =======================
# Helpers
# =======================
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"^#\s*", "", regex=True)  # "# Time" -> "Time"
    )
    return df

def save_pdf(path: Path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def aggregate_importance(pipe, feature_cols):
    """Aggregate one-hot importances back to original columns."""
    ohe = pipe.named_steps["preprocessing"].named_transformers_["cat"].named_steps["onehot"]
    fnames = ohe.get_feature_names_out(feature_cols)
    imps = pipe.named_steps["model"].feature_importances_
    agg = {c: 0.0 for c in feature_cols}
    for name, imp in zip(fnames, imps):
        base = name.split("_", 1)[0]
        if base in agg:
            agg[base] += float(imp)
        else:
            for col in feature_cols:
                if name.startswith(col + "_"):
                    agg[col] += float(imp)
                    break
    return pd.Series(agg).sort_values(ascending=False)

# =======================
# Load
# =======================
df = normalize_headers(pd.read_csv(CSV_PATH))

missing = [c for c in FEATURES + [TARGET_ORIG] if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}\nAvailable: {df.columns.tolist()}")

# Create binary target: 1=severe, 0=non-severe
df["Severe_accident"] = df[TARGET_ORIG].isin(SEVERE_LABELS).astype(int)

X = df[FEATURES].copy()
y = df["Severe_accident"].copy()  # 1 severe, 0 non-severe

# =======================
# Split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =======================
# Preprocessing (categorical)
# =======================
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[("cat", cat_pipe, FEATURES)],
    remainder="drop"
)

# =======================
# Models
# =======================
# Baseline: class_weight (emphasize severe)
baseline_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight={0: 1, 1: 4},  # tune this if needed
    min_samples_leaf=2,
)

# SMOTE model: no class_weight (usually)
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
# Predict + probs
# =======================
y_pred_base = baseline_pipe.predict(X_test)
y_proba_base = baseline_pipe.predict_proba(X_test)[:, 1]  # P(severe)

y_pred_sm = smote_pipe.predict(X_test)
y_proba_sm = smote_pipe.predict_proba(X_test)[:, 1]       # P(severe)

# =======================
# Metrics (binary)
# =======================
def summarize_binary(y_true, y_pred, y_proba, name: str):
    ap = average_precision_score(y_true, y_proba)         # PR-AUC
    auc = roc_auc_score(y_true, y_proba)                 # ROC-AUC
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "model": name,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": ap,
        "roc_auc": auc,
        "accuracy": float((y_true == y_pred).mean()),
        "support_severe": int(y_true.sum()),
        "support_total": int(len(y_true)),
    }

row_base = summarize_binary(y_test, y_pred_base, y_proba_base, "Baseline (class_weight)")
row_sm = summarize_binary(y_test, y_pred_sm, y_proba_sm, "SMOTE")

metrics_df = pd.DataFrame([row_base, row_sm])
metrics_df.to_csv(OUT_DIR / "severe_metrics_summary.csv", index=False)
with open(OUT_DIR / "severe_metrics_summary.tex", "w", encoding="utf-8") as f:
    f.write(metrics_df.to_latex(index=False, float_format="%.3f"))

print("\n=== Baseline (class_weight) ===")
print(classification_report(y_test, y_pred_base, zero_division=0, target_names=["non-severe", "severe"]))
print("\n=== SMOTE ===")
print(classification_report(y_test, y_pred_sm, zero_division=0, target_names=["non-severe", "severe"]))
print("\nSaved metrics to:", OUT_DIR)

# =======================
# Plot: Severe class distribution (whole dataset)
# =======================
plt.figure(figsize=(6, 4))
dist = df["Severe_accident"].value_counts(normalize=True) * 100
dist = dist.rename(index={0: "non-severe (slight)", 1: "severe (fatal+serious)"})
dist.plot(kind="bar")
plt.ylabel("Percentage (%)")
plt.title("Binary Class Distribution")
save_pdf(OUT_DIR / "severe_class_distribution.pdf")

# =======================
# Plot: Confusion matrices (separate + compare)
# =======================
cm_base = confusion_matrix(y_test, y_pred_base, labels=[0, 1])
cm_sm = confusion_matrix(y_test, y_pred_sm, labels=[0, 1])
labels = ["non-severe", "severe"]

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cm_base, display_labels=labels).plot(values_format="d")
plt.title("Confusion Matrix — Baseline (class_weight)")
save_pdf(OUT_DIR / "severe_confusion_baseline.pdf")

plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay(cm_sm, display_labels=labels).plot(values_format="d")
plt.title("Confusion Matrix — SMOTE")
save_pdf(OUT_DIR / "severe_confusion_smote.pdf")

# Side-by-side compare (one PDF, two panels)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrix Comparison (Counts)")
for ax, cm, title in [(axes[0], cm_base, "Baseline"), (axes[1], cm_sm, "SMOTE")]:
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
save_pdf(OUT_DIR / "severe_confusion_compare.pdf")

# =======================
# Plot: PR curve compare
# =======================
p_base, r_base, _ = precision_recall_curve(y_test, y_proba_base)
p_sm, r_sm, _ = precision_recall_curve(y_test, y_proba_sm)

plt.figure(figsize=(7, 5))
plt.plot(r_base, p_base, label=f"Baseline (AP={row_base['pr_auc']:.3f})")
plt.plot(r_sm, p_sm, label=f"SMOTE (AP={row_sm['pr_auc']:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Severe = fatal+serious)")
plt.legend()
save_pdf(OUT_DIR / "severe_pr_curve_compare.pdf")

# =======================
# Plot: ROC curve compare
# =======================
fpr_b, tpr_b, _ = roc_curve(y_test, y_proba_base)
fpr_s, tpr_s, _ = roc_curve(y_test, y_proba_sm)

plt.figure(figsize=(7, 5))
plt.plot(fpr_b, tpr_b, label=f"Baseline (AUC={row_base['roc_auc']:.3f})")
plt.plot(fpr_s, tpr_s, label=f"SMOTE (AUC={row_sm['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Severe = fatal+serious)")
plt.legend()
save_pdf(OUT_DIR / "severe_roc_curve_compare.pdf")

# =======================
# Threshold optimization on SMOTE (optimize severe recall explicitly)
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

# Plot threshold tradeoff
plt.figure(figsize=(7, 5))
plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")
plt.xlabel("Severe probability threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization (SMOTE) — Severe accidents")
plt.legend()
save_pdf(OUT_DIR / "severe_threshold_optimization.pdf")

# Choose threshold by policy: maximize recall with a precision floor
PRECISION_FLOOR = 0.20  # adjust to your report goal
cand = thr_df[thr_df["precision"] >= PRECISION_FLOOR].sort_values("recall", ascending=False).head(1)
if len(cand) == 0:
    best_t = float(thr_df.sort_values("f1", ascending=False).iloc[0]["threshold"])
    print(f"\nNo threshold met precision >= {PRECISION_FLOOR:.2f}; fallback to best F1.")
else:
    best_t = float(cand.iloc[0]["threshold"])

print(f"\nChosen severe threshold t = {best_t:.3f} (precision floor = {PRECISION_FLOOR:.2f})")

from sklearn.metrics import classification_report

# Apply threshold override to SMOTE predictions
y_pred_sm_thresholded = (y_proba_sm >= best_t).astype(int)

print("\n=== SMOTE + severe-threshold override ===")
print(classification_report(
    y_test,
    y_pred_sm_thresholded,
    zero_division=0,
    target_names=["non-severe", "severe"]
))


# =======================
# Feature importance compare (aggregated)
# =======================
imp_base = aggregate_importance(baseline_pipe, FEATURES)
imp_sm = aggregate_importance(smote_pipe, FEATURES)
topn = 12
top_features = list(dict.fromkeys(list(imp_base.head(topn).index) + list(imp_sm.head(topn).index)))[:topn]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Top Feature Importances (Aggregated) — Severe model")

imp_base.loc[top_features].sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("Baseline")
axes[0].set_xlabel("Importance")

imp_sm.loc[top_features].sort_values().plot(kind="barh", ax=axes[1])
axes[1].set_title("SMOTE")
axes[1].set_xlabel("Importance")

save_pdf(OUT_DIR / "severe_feature_importance_compare.pdf")

# =======================
# Save models
# =======================
joblib.dump(baseline_pipe, OUT_DIR / "severe_baseline_classweight_rf.joblib")
joblib.dump(smote_pipe, OUT_DIR / "severe_smote_rf.joblib")

print("\nDone. Outputs saved to:", OUT_DIR.resolve())
