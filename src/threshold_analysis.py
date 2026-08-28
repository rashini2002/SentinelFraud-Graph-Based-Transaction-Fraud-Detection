"""
threshold_analysis.py

Day 14 — precision-recall/threshold analysis for the Day 13 enhanced model,
framed as a business cost tradeoff rather than a pure ML metrics exercise.

Framing: a false positive costs investigator time (reviewing a legitimate
transaction that was flagged). A false negative costs actual fraud loss
(a fraudulent transaction that went undetected). These costs are NOT
equal, so the "best" threshold depends on their relative weight — this
script computes a range of thresholds so that tradeoff is explicit rather
than hidden behind a single default 0.5 cutoff.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import xgboost as xgb
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
ENHANCED_FEATURES = BASELINE_FEATURES + GRAPH_FEATURES

# Illustrative cost assumptions — these are ASSUMPTIONS made explicit for
# demonstration, not derived from real operational data (this is a
# portfolio project on synthetic data, not a live fraud system). Documented
# clearly so anyone reading this understands these are illustrative, not
# empirical claims.
COST_FALSE_POSITIVE = 5       # ~5 minutes of investigator review time, illustrative
COST_FALSE_NEGATIVE = 150     # illustrative average fraud loss per missed transaction


def load_and_prepare():
    df = pd.read_parquet(PROCESSED_DIR / "transactions_with_graph_features.parquet")
    df["product_cd_encoded"] = df["product_cd"].astype("category").cat.codes
    df["p_email_domain_encoded"] = df["p_email_domain"].astype("category").cat.codes

    X = df[ENHANCED_FEATURES].copy()
    X["has_device_identity"] = X["has_device_identity"].astype(int)
    y = df["is_fraud"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)
    return model


def analyze_thresholds(model, X_test, y_test) -> pd.DataFrame:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc:.4f}")

    # Evaluate a specific set of candidate thresholds for the business table
    candidate_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_fraud = y_test.sum()
    n_legit = len(y_test) - n_fraud

    rows = []
    for t in candidate_thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        total_cost = fp * COST_FALSE_POSITIVE + fn * COST_FALSE_NEGATIVE
        alerts_raised = tp + fp
        pct_of_legit_flagged = fp / n_legit * 100

        rows.append({
            "threshold": t,
            "precision": prec,
            "recall": rec,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "alerts_raised": alerts_raised,
            "pct_legit_flagged": pct_of_legit_flagged,
            "illustrative_total_cost": total_cost,
        })

    results_df = pd.DataFrame(rows)
    print("\n--- Threshold analysis ---")
    print(results_df.to_string(index=False))

    best_cost_row = results_df.loc[results_df["illustrative_total_cost"].idxmin()]
    print(f"\nLowest illustrative-cost threshold: {best_cost_row['threshold']} "
          f"(cost={best_cost_row['illustrative_total_cost']:.0f}, "
          f"recall={best_cost_row['recall']:.3f}, "
          f"{best_cost_row['pct_legit_flagged']:.2f}% of legit transactions flagged)")

    return results_df, precision, recall, thresholds


def plot_pr_curve(precision, recall, out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Enhanced Model (Day 13 features)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve to: {out_path}")


def main():
    X_train, X_test, y_train, y_test = load_and_prepare()
    model = train_model(X_train, y_train)
    results_df, precision, recall, thresholds = analyze_thresholds(model, X_test, y_test)
    plot_pr_curve(precision, recall, DOCS_DIR / "day14_pr_curve.png")

    results_df.to_csv(DOCS_DIR / "day14_threshold_analysis.csv", index=False)
    print(f"\nSaved threshold analysis to docs/day14_threshold_analysis.csv")


if __name__ == "__main__":
    main()