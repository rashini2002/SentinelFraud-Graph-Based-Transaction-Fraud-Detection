"""
generate_synthetic.py

Day 3 — fits an SDV GaussianCopula synthesizer on a curated subset of the
cleaned IEEE-CIS distribution and generates a synthetic transaction dataset.

SCOPE DECISION (see docs/DECISIONS.md):
We do NOT synthesize all 435 columns. The V1-V339 Vesta engineered features
are anonymized and not interpretable — synthesizing all of them would be slow,
memory-heavy, and adds no value for this project's purpose (graph-based fraud
ring detection + classification). Instead we synthesize a curated column set:
core transaction fields, the primary/secondary/high-confidence linkage keys,
and a small sample of V-columns to preserve some behavioral-feature signal
for the classifier stage.

GaussianCopula (not CTGAN) is used deliberately — CTGAN is far slower and
is overkill for this dataset's column types; GaussianCopula fits in minutes
rather than hours on a laptop and is sufficient for preserving marginal
distributions and correlations for this project's purposes.
"""

import pandas as pd
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_DIR.mkdir(exist_ok=True, parents=True)

N_SYNTHETIC_ROWS = 400_000  # per Day 1 scoping decision (300-500K range)

# Curated column set — see docstring above for rationale
CORE_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD",
]
LINKAGE_COLUMNS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain",
    "DeviceType", "DeviceInfo", "has_device_identity",
]
# Small sample of V-columns retained for the classifier stage — not all 339.
# These are arbitrarily spaced picks across the V-block to get some spread
# of whatever latent signal they encode, without full computational cost.
SAMPLE_V_COLUMNS = [f"V{i}" for i in [1, 12, 45, 78, 100, 130, 160, 200, 250, 300]]

SYNTHESIS_COLUMNS = CORE_COLUMNS + LINKAGE_COLUMNS + SAMPLE_V_COLUMNS


def load_curated_data() -> pd.DataFrame:
    print("Loading cleaned dataset...")
    df = pd.read_parquet(PROCESSED_DIR / "transactions_cleaned.parquet")

    available_cols = [c for c in SYNTHESIS_COLUMNS if c in df.columns]
    missing_cols = set(SYNTHESIS_COLUMNS) - set(available_cols)
    if missing_cols:
        print(f"WARNING — expected columns not found, skipping: {missing_cols}")

    df_subset = df[available_cols].copy()
    print(f"Curated subset shape: {df_subset.shape}")
    return df_subset


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    print("Detecting metadata...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    # TransactionID should not be treated as a modeled feature — it's an
    # identifier. Mark it as a primary key so SDV generates fresh unique IDs
    # rather than trying to learn its distribution.
    if "TransactionID" in df.columns:
        metadata.update_column(column_name="TransactionID", sdtype="id")
        metadata.set_primary_key("TransactionID")

    return metadata


def fit_and_sample(df: pd.DataFrame, metadata: SingleTableMetadata) -> pd.DataFrame:
    print("Fitting GaussianCopula synthesizer (this may take several minutes)...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)

    print(f"Sampling {N_SYNTHETIC_ROWS:,} synthetic rows...")
    synthetic_data = synthesizer.sample(num_rows=N_SYNTHETIC_ROWS)

    return synthetic_data


def validate_synthetic(real: pd.DataFrame, synthetic: pd.DataFrame, metadata: SingleTableMetadata) -> None:
    # NOTE: SDV's evaluate_quality() is skipped here due to a version
    # mismatch between SingleTableMetadata and the evaluation module in the
    # currently installed SDV release (see docs/DECISIONS.md Day 3 notes).
    # The fraud-rate check below is the most important validation for this
    # project's purposes and doesn't depend on SDV's internal metadata API.
    print("\nComparing real vs. synthetic fraud rate...")

    real_fraud_rate = real["isFraud"].mean() * 100
    synth_fraud_rate = synthetic["isFraud"].mean() * 100
    print(f"\nReal fraud rate: {real_fraud_rate:.3f}%")
    print(f"Synthetic fraud rate: {synth_fraud_rate:.3f}%")
    print(f"Difference: {abs(real_fraud_rate - synth_fraud_rate):.3f} percentage points")

    # Quick distributional sanity check on a couple of key numeric columns,
    # as a lightweight substitute for the full SDV quality report.
    print("\nTransactionAmt — real vs synthetic (describe):")
    print(pd.concat(
        [real["TransactionAmt"].describe().rename("real"),
         synthetic["TransactionAmt"].describe().rename("synthetic")],
        axis=1
    ))

    print("\ncard1 — unique value count, real vs synthetic:")
    print(f"  real: {real['card1'].nunique()}, synthetic: {synthetic['card1'].nunique()}")


def main():
    df = load_curated_data()
    metadata = build_metadata(df)
    synthetic_data = fit_and_sample(df, metadata)

    out_path = SYNTHETIC_DIR / "synthetic_transactions_base.parquet"
    synthetic_data.to_parquet(out_path, index=False)
    print(f"\nSaved synthetic dataset to: {out_path}")

    validate_synthetic(df, synthetic_data, metadata)


if __name__ == "__main__":
    main()