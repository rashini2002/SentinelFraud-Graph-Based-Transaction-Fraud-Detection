"""
profile_data.py
Initial profiling of the IEEE-CIS Fraud Detection seed dataset.
Run this before any cleaning/imputation decisions — the output here
drives which columns get dropped, imputed, or kept as-is.
"""

import pandas as pd
from pathlib import Path

# Use parents[1] since this script lives in src/, but data/ and docs/ are at repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)


def profile_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a summary table: null %, dtype, unique count per column."""
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "n_unique": df.nunique(),
    })
    summary = summary.sort_values("null_pct", ascending=False)
    summary.to_csv(OUT_DIR / f"{name}_profile.csv")
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns with >90% nulls: {(summary['null_pct'] > 90).sum()}")
    print(f"Columns with >50% nulls: {(summary['null_pct'] > 50).sum()}")
    return summary


def main():
    print("Loading transaction data...")
    txn = pd.read_csv(RAW_DIR / "train_transaction.csv")

    print("Loading identity data...")
    identity = pd.read_csv(RAW_DIR / "train_identity.csv")

    txn_summary = profile_dataframe(txn, "transaction")
    identity_summary = profile_dataframe(identity, "identity")

    # Class imbalance — this number drives your SMOTE/class-weighting decision later
    fraud_rate = txn["isFraud"].mean() * 100
    print(f"\nFraud rate: {fraud_rate:.3f}%")
    print(f"Fraud count: {txn['isFraud'].sum()} / {len(txn)}")

    # Merge rate — how many transactions actually have identity data joined
    match_rate = identity.shape[0] / txn.shape[0] * 100
    print(f"\nIdentity match rate: {match_rate:.2f}% of transactions have identity data")

    # Identity/linkage-relevant columns worth checking sparsity on specifically
    # (these are your candidate fields for fraud-ring graph edges later)
    linkage_candidates = ["card1", "card2", "card3", "card4", "card5", "card6",
                           "addr1", "addr2", "P_emaildomain", "R_emaildomain",
                           "dist1", "dist2"]
    print("\n--- Linkage candidate columns (for graph edges) ---")
    for col in linkage_candidates:
        if col in txn.columns:
            null_pct = txn[col].isnull().mean() * 100
            n_unique = txn[col].nunique()
            print(f"  {col}: {null_pct:.1f}% null, {n_unique} unique values")

    if "DeviceType" in identity.columns:
        print(f"\n  DeviceType unique values: {identity['DeviceType'].nunique()}")
    if "DeviceInfo" in identity.columns:
        print(f"  DeviceInfo unique values: {identity['DeviceInfo'].nunique()}")

    with open(OUT_DIR / "profiling_summary.md", "w") as f:
        f.write("# Data Profiling Summary — IEEE-CIS Fraud Detection\n\n")
        f.write(f"- Transaction rows: {txn.shape[0]:,}, columns: {txn.shape[1]}\n")
        f.write(f"- Identity rows: {identity.shape[0]:,}, columns: {identity.shape[1]}\n")
        f.write(f"- Fraud rate: {fraud_rate:.3f}%\n")
        f.write(f"- Identity match rate: {match_rate:.2f}%\n")
        f.write(f"- Transaction columns >90% null: {(txn_summary['null_pct'] > 90).sum()}\n")
        f.write(f"- Identity columns >90% null: {(identity_summary['null_pct'] > 90).sum()}\n")
        f.write("\n## Linkage candidate columns\n")
        f.write("These are candidate fields for building the shared-attribute graph\n")
        f.write("(fraud ring detection) in the analytics engineering phase:\n")
        f.write("card1-6, addr1-2, P_emaildomain, R_emaildomain, dist1-2, DeviceType, DeviceInfo\n")

    print(f"\nDone. Full profiles saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
