"""
clean_data.py

Day 2 — applies the column disposition and missing-data rules documented in
docs/DECISIONS.md to produce a cleaned, merged dataset ready for Day 3's
SDV-based synthetic data generation.

This script does NOT impute high-null feature columns (V1-V339 etc.) — those
are left as NaN intentionally, since the downstream XGBoost classifier (Day 12)
handles missing values natively. Imputing them here would add noise without
benefit and would contradict the documented decision.

What this script DOES do:
  1. Merge transaction + identity data (left join, since only 24.42% of
     transactions have identity data — we keep all transactions).
  2. Drop linkage columns confirmed too sparse to be reliable (R_emaildomain,
     dist1, dist2) — dropped only from the linkage-key set, not necessarily
     from the full feature set used later for modeling.
  3. Flag which linkage keys are "primary" vs "secondary" vs "high-confidence"
     so Day 3-4's ring-injection logic can reference them directly instead of
     re-deriving this every time.
  4. Output a cleaned parquet file (faster to reload than CSV for the next
     several days of work).
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)

# --- Linkage column tiers, per docs/DECISIONS.md Day 2 ---
PRIMARY_LINKAGE_KEYS = ["card1", "card2", "addr1"]
SECONDARY_LINKAGE_KEYS = ["card3", "card4", "card5", "card6", "addr2", "P_emaildomain"]
HIGH_CONFIDENCE_LINKAGE_KEYS = ["DeviceInfo", "DeviceType"]  # only present for ~24% of rows

# Columns confirmed too sparse to be reliable for linkage (kept in the dataset
# for potential modeling use, just excluded from the linkage-key set)
DROPPED_LINKAGE_CANDIDATES = ["R_emaildomain", "dist1", "dist2"]


def load_and_merge() -> pd.DataFrame:
    print("Loading transaction data...")
    txn = pd.read_csv(RAW_DIR / "train_transaction.csv")

    print("Loading identity data...")
    identity = pd.read_csv(RAW_DIR / "train_identity.csv")

    print("Merging (left join on TransactionID — keeps all transactions)...")
    df = txn.merge(identity, on="TransactionID", how="left")

    print(f"Merged shape: {df.shape}")
    return df


def add_linkage_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean flag for whether each transaction has high-confidence
    (device-level) identity data available. This drives the two-tier
    fraud-ring injection logic in Day 3-4: Tier 1 rings can only be planted
    among rows where has_device_identity == True.
    """
    if "DeviceInfo" in df.columns:
        df["has_device_identity"] = df["DeviceInfo"].notna()
    else:
        df["has_device_identity"] = False

    n_with_device = df["has_device_identity"].sum()
    pct_with_device = n_with_device / len(df) * 100
    print(f"Rows with device identity available: {n_with_device:,} ({pct_with_device:.2f}%)")

    return df


def report_dropped_columns(df: pd.DataFrame) -> None:
    """
    We don't physically drop DROPPED_LINKAGE_CANDIDATES from the dataframe —
    they may still have modeling value even though they're unreliable for
    linkage. This just confirms they exist and reports their sparsity so
    it's visible in the pipeline output, matching what's documented.
    """
    print("\nLinkage candidates excluded from graph-edge construction (kept in dataset for modeling):")
    for col in DROPPED_LINKAGE_CANDIDATES:
        if col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            print(f"  {col}: {null_pct:.1f}% null — excluded from linkage keys, retained as a feature")


def validate_primary_keys(df: pd.DataFrame) -> None:
    """
    Sanity check: primary linkage keys should have very low null rates,
    per the Day 2 profiling. If this assumption breaks (e.g. a future data
    refresh has much higher nulls), we want to know immediately rather than
    silently building a weaker graph later.
    """
    print("\nValidating primary linkage key completeness...")
    for col in PRIMARY_LINKAGE_KEYS:
        null_pct = df[col].isnull().mean() * 100
        status = "OK" if null_pct < 15 else "WARNING — higher than expected"
        print(f"  {col}: {null_pct:.1f}% null [{status}]")


def main():
    df = load_and_merge()
    df = add_linkage_metadata(df)
    report_dropped_columns(df)
    validate_primary_keys(df)

    out_path = PROCESSED_DIR / "transactions_cleaned.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved cleaned dataset to: {out_path}")
    print(f"Final shape: {df.shape}")

    # Small metadata file so Day 3's SDV script and Day 4's ring-injection
    # script don't need to re-derive which columns are which tier.
    metadata_path = PROCESSED_DIR / "linkage_key_metadata.txt"
    with open(metadata_path, "w") as f:
        f.write("PRIMARY_LINKAGE_KEYS=" + ",".join(PRIMARY_LINKAGE_KEYS) + "\n")
        f.write("SECONDARY_LINKAGE_KEYS=" + ",".join(SECONDARY_LINKAGE_KEYS) + "\n")
        f.write("HIGH_CONFIDENCE_LINKAGE_KEYS=" + ",".join(HIGH_CONFIDENCE_LINKAGE_KEYS) + "\n")
        f.write("DROPPED_LINKAGE_CANDIDATES=" + ",".join(DROPPED_LINKAGE_CANDIDATES) + "\n")
    print(f"Saved linkage key metadata to: {metadata_path}")


if __name__ == "__main__":
    main()