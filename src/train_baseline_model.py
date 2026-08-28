"""
train_baseline_model.py

Day 12 — trains a baseline XGBoost classifier using ONLY standard
transaction features (no graph features). This is the comparison point
Day 13's enhanced model will be measured against.

Class imbalance handling: both SMOTE and class-weighting are tried, and
compared, rather than assuming one approach — per the original project
plan (docs/DECISIONS.md Day 2 committed to this comparison rather than
picking blindly).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42

# Standard transaction features only — NO graph_* columns, NO ring_id/ring_tier
BASELINE_FEATURES = [
    "transaction_amt", "has_device_identity",
    "product_cd_encoded", "p_email_domain_encoded",
    "v1", "v12", "v45", "v78", "v100", "v130", "v160", "v200", "v250", "v300",
]
# Note: card1/addr1/etc. are excluded as direct model features here since
# they are high-cardinality identifiers (not generalizable patterns) —
# their PREDICTIVE value is exactly what the graph features in Day 13
# are meant to capture in a generalizable form. product_cd and
# p_email_domain are low-cardinality categoricals, label-encoded below.


def load_data() -> pd.DataFrame:
    print("Loading modeling table...")
    df = pd.read_parquet(PROCESSED_DIR / "transactions_with_graph_features.parquet")
    print(f"Loaded {len(df):,} rows")
    return df


def prepare_split(df: pd.DataFrame, feature_cols: list):
    df = df.copy()

    # Label-encode low-cardinality categoricals
    df["product_cd_encoded"] = df["product_cd"].astype("category").cat.codes
    df["p_email_domain_encoded"] = df["p_email_domain"].astype("category").cat.codes

    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()

    if "has_device_identity" in X.columns:
        X["has_device_identity"] = X["has_device_identity"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train: {len(X_train):,} ({y_train.mean()*100:.3f}% fraud) | "
          f"Test: {len(X_test):,} ({y_test.mean()*100:.3f}% fraud)")
    return X_train, X_test, y_train, y_test


def train_with_class_weighting(X_train, y_train) -> xgb.XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n[Class weighting] scale_pos_weight = {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def train_with_smote(X_train, y_train) -> xgb.XGBClassifier:
    print("\n[SMOTE] Imputing missing values (median) — SMOTE cannot handle NaN,")
    print("unlike XGBoost's native missing-value handling used in the class-weighted")
    print("path. This imputation is applied ONLY for this SMOTE branch; the Day 2")
    print("decision to leave NaNs for XGBoost to handle natively is preserved")
    print("elsewhere. See docs/DECISIONS.md Day 12.")
    X_train_imputed = X_train.fillna(X_train.median())

    print("[SMOTE] Resampling training set...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)
    print(f"Resampled train: {len(X_resampled):,} "
          f"({y_resampled.mean()*100:.1f}% fraud)")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_resampled, y_resampled)
    return model


def evaluate(model, X_test, y_test, label: str) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "label": label,
        "auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    print(f"\n--- {label} ---")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legit", "fraud"]))

    return metrics


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_split(df, BASELINE_FEATURES)

    model_weighted = train_with_class_weighting(X_train, y_train)
    metrics_weighted = evaluate(model_weighted, X_test, y_test, "Baseline (class-weighted)")

    model_smote = train_with_smote(X_train, y_train)
    X_test_imputed = X_test.fillna(X_train.median())  # use TRAIN median, avoid test-set leakage
    metrics_smote = evaluate(model_smote, X_test_imputed, y_test, "Baseline (SMOTE)")

    # Pick whichever approach scored higher AUC as "the" baseline going
    # forward into Day 13's comparison — document the choice either way.
    best = metrics_weighted if metrics_weighted["auc"] >= metrics_smote["auc"] else metrics_smote
    print(f"\n>>> Selected approach for baseline comparison: {best['label']} (AUC {best['auc']:.4f})")

    results = {
        "class_weighted": metrics_weighted,
        "smote": metrics_smote,
        "selected_for_comparison": best["label"],
    }
    with open(DOCS_DIR / "day12_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to docs/day12_baseline_results.json")


if __name__ == "__main__":
    main()