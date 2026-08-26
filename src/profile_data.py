"""
profile_data.py
Initial profiling of the ULB Credit Card Fraud Detection seed dataset.
Run this before any cleaning/imputation decisions.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "docs"
OUT_DIR.mkdir(exist_ok=True)


def find_dataset_path(raw_dir: Path) -> Path:
    """Prefer the transaction dataset, then fall back to any CSV in the raw folder."""
    preferred_names = ["train_transaction.csv", "creditcard.csv", "transactions.csv"]
    for name in preferred_names:
        candidate = raw_dir / name
        if candidate.exists():
            return candidate

    csv_files = sorted(raw_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    available = ", ".join(sorted(p.name for p in raw_dir.glob("*"))) or "none"
    raise FileNotFoundError(
        f"No CSV dataset found in {raw_dir}. Available files: {available}"
    )


def resolve_label_column(df: pd.DataFrame) -> str:
    """Return the fraud label column name for either legacy or IEEE-CIS datasets."""
    for candidate in ("Class", "isFraud"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No fraud label column found. Expected 'Class' or 'isFraud'.")


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
    print(f"Columns with any nulls: {(summary['null_pct'] > 0).sum()}")
    return summary


def main():
    dataset_path = find_dataset_path(RAW_DIR)
    print(f"Loading credit card transaction data from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    dataset_name = dataset_path.stem
    summary = profile_dataframe(df, dataset_name)

    label_col = resolve_label_column(df)

    # Class imbalance — drives SMOTE/class-weighting decision later
    fraud_rate = df[label_col].mean() * 100
    fraud_count = df[label_col].sum()
    print(f"\nFraud rate: {fraud_rate:.4f}%")
    print(f"Fraud count: {fraud_count} / {len(df)}")

    amount_col = "Amount" if "Amount" in df.columns else "TransactionAmt"
    time_col = "Time" if "Time" in df.columns else "TransactionDT"

    # Amount distribution — relevant since fraud amounts often differ from legit
    print(f"\nAmount stats (legit): \n{df[df[label_col] == 0][amount_col].describe()}")
    print(f"\nAmount stats (fraud): \n{df[df[label_col] == 1][amount_col].describe()}")

    # Time span (seconds elapsed from first transaction in dataset)
    time_span_hours = df[time_col].max() / 3600
    print(f"\nTime span: ~{time_span_hours:.1f} hours of transaction data")

    with open(OUT_DIR / "profiling_summary.md", "w") as f:
        f.write("# Data Profiling Summary — ULB Credit Card Fraud Dataset\n\n")
        f.write(f"- Total rows: {df.shape[0]:,}, columns: {df.shape[1]}\n")
        f.write(f"- Fraud rate: {fraud_rate:.4f}%\n")
        f.write(f"- Fraud count: {fraud_count} / {len(df)}\n")
        f.write(f"- Time span: ~{time_span_hours:.1f} hours\n")
        f.write(f"- Columns with nulls: {(summary['null_pct'] > 0).sum()} (expect 0 — this dataset is pre-cleaned)\n")
        f.write("\n## Note\n")
        f.write("This dataset has NO identity/device columns (V1-V28 are PCA-anonymized).\n")
        f.write("Identity/device linkage fields for fraud-ring simulation will be\n")
        f.write("synthetically generated in the data generation step (Day 3-4), not\n")
        f.write("sourced from this seed dataset.\n")

    print(f"\nDone. Full profile saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
