"""
train_enhanced_model.py

Day 13 — trains XGBoost with baseline features PLUS graph-derived features
(Day 11), using the same train/test split and class-weighting approach that
won in Day 12, for a fair apples-to-apples comparison. Runs SHAP to show
which features actually drove the improvement.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

RANDOM_SEED = 42

BASELINE_FEATURES = [
    "transaction_amt", "has_device_identity",
    "product_cd_encoded", "p_email_domain_encoded",
    "v1", "v12", "v45", "v78", "v100", "v130", "v160", "v200", "v250", "v300",
]

GRAPH_FEATURES = [
    "graph_degree", "graph_weighted_degree",
    "graph_community_size", "graph_community_fraud_density",
]
# NOTE: graph_community_id is intentionally excluded — it's an arbitrary
# integer label, not an ordinal or meaningful numeric feature. Including it
# would let the model "memorize" specific community IDs rather than learn
# generalizable structural patterns.

ENHANCED_FEATURES = BASELINE_FEATURES + GRAPH_FEATURES


def load_data() -> pd.DataFrame:
    print("Loading modeling table...")
    df = pd.read_parquet(PROCESSED_DIR / "transactions_with_graph_features.parquet")
    print(f"Loaded {len(df):,} rows")
    return df


def prepare_split(df: pd.DataFrame, feature_cols: list):
    df = df.copy()
    df["product_cd_encoded"] = df["product_cd"].astype("category").cat.codes
    df["p_email_domain_encoded"] = df["p_email_domain"].astype("category").cat.codes

    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()

    if "has_device_identity" in X.columns:
        X["has_device_identity"] = X["has_device_identity"].astype(int)

    # Same random_state and stratify as Day 12 — ensures an IDENTICAL
    # train/test split, which is essential for a fair before/after comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_enhanced(X_train, y_train) -> xgb.XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

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


def run_shap_analysis(model, X_test, out_path: Path) -> pd.DataFrame:
    print("\nRunning SHAP analysis...")

    # Workaround for a known XGBoost 2.x / SHAP compatibility bug: XGBoost
    # serializes base_score as a bracketed scientific-notation string
    # (e.g. '[5E-1]'), which SHAP's config parser cannot convert to float.
    # Fix: strip the brackets directly in the booster's saved config before
    # SHAP reads it.
    import json
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    base_score = config["learner"]["learner_model_param"]["base_score"]
    if base_score.startswith("["):
        config["learner"]["learner_model_param"]["base_score"] = base_score.strip("[]")
        booster.load_config(json.dumps(config))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nTop 10 features by mean |SHAP value|:")
    print(importance_df.head(10).to_string(index=False))

    n_graph_in_top10 = importance_df.head(10)["feature"].isin(GRAPH_FEATURES).sum()
    print(f"\nGraph features in top 10: {n_graph_in_top10} / {len(GRAPH_FEATURES)} total graph features")

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved SHAP summary plot to: {out_path}")

    return importance_df


def load_baseline_results() -> dict:
    with open(DOCS_DIR / "day12_baseline_results.json") as f:
        return json.load(f)


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_split(df, ENHANCED_FEATURES)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    model = train_enhanced(X_train, y_train)
    enhanced_metrics = evaluate(model, X_test, y_test, "Enhanced (baseline + graph features)")

    importance_df = run_shap_analysis(model, X_test, DOCS_DIR / "day13_shap_summary.png")

    # Direct comparison against Day 12's selected baseline
    baseline_results = load_baseline_results()
    baseline_metrics = baseline_results["class_weighted"]  # the selected one from Day 12

    print("\n" + "=" * 60)
    print("DAY 12 vs DAY 13 — DIRECT COMPARISON")
    print("=" * 60)
    comparison = pd.DataFrame([
        {"metric": "AUC", "baseline": baseline_metrics["auc"], "enhanced": enhanced_metrics["auc"]},
        {"metric": "Precision", "baseline": baseline_metrics["precision"], "enhanced": enhanced_metrics["precision"]},
        {"metric": "Recall", "baseline": baseline_metrics["recall"], "enhanced": enhanced_metrics["recall"]},
        {"metric": "F1", "baseline": baseline_metrics["f1"], "enhanced": enhanced_metrics["f1"]},
    ])
    comparison["improvement"] = comparison["enhanced"] - comparison["baseline"]
    comparison["improvement_pct"] = (comparison["improvement"] / comparison["baseline"] * 100).round(1)
    print(comparison.to_string(index=False))

    comparison.to_csv(DOCS_DIR / "day13_comparison.csv", index=False)
    importance_df.to_csv(DOCS_DIR / "day13_shap_importance.csv", index=False)

    results = {"enhanced": enhanced_metrics, "comparison": comparison.to_dict(orient="records")}
    with open(DOCS_DIR / "day13_enhanced_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved all Day 13 results to docs/")


if __name__ == "__main__":
    main()