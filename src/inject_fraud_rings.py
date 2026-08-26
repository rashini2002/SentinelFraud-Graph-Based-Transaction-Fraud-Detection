"""
inject_fraud_rings.py

Day 4 — plants synthetic collusive fraud rings into the SDV-generated
transaction data, per the two-tier design documented in docs/DECISIONS.md:

  Tier 1 (high confidence): ring members share BOTH a card1/addr1 pattern
    AND a DeviceInfo value. Only eligible among transactions where
    has_device_identity == True (~20% of the data).
  Tier 2 (weak signal): ring members share ONLY a card1/addr1 pattern,
    no device overlap. Eligible across the full dataset.

Ring membership is planted primarily among isFraud == 1 transactions (the
realistic case: a fraud ring's transactions are fraudulent), with an
optional small number of "mule" accounts — legitimate-labeled transactions
recruited into a ring to simulate rings that partially evade upstream
fraud labeling. This is a deliberate realism choice: real fraud rings often
include some transactions that individually look clean.

Ground truth (ring_id, ring_tier) is saved so Day 10's community detection
results can be scored against it — but this ground truth must NEVER be fed
into the classifier or graph-construction features, only used for evaluation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# --- Ring generation parameters ---
N_TIER1_SMALL_RINGS = 25    # size 3-6, high-confidence
N_TIER1_LARGE_RINGS = 5     # size 10-18, high-confidence
N_TIER2_SMALL_RINGS = 50    # size 2-5, weak-signal
N_TIER2_LARGE_RINGS = 8     # size 10-20, weak-signal

MULE_FRACTION = 0.10  # fraction of each ring's members drawn from non-fraud rows


def load_data() -> pd.DataFrame:
    print("Loading synthetic base dataset...")
    df = pd.read_parquet(SYNTHETIC_DIR / "synthetic_transactions_base.parquet")
    df["ring_id"] = np.nan
    df["ring_tier"] = None
    print(f"Loaded {len(df):,} rows")
    return df


def generate_fake_identity_value(prefix: str) -> str:
    """Generate a synthetic-looking shared identity string for ring injection."""
    return f"{prefix}_{rng.integers(100000, 999999)}"


def pick_ring_members(df: pd.DataFrame, size: int, tier: str, used_indices: set) -> list:
    """
    Select row indices for a ring. Primarily draws from isFraud==1 rows;
    a small fraction are drawn from isFraud==0 rows to simulate mule
    accounts. For Tier 1, all members must have has_device_identity==True.
    """
    if tier == "tier1":
        pool = df[(df["has_device_identity"] == True) & (~df.index.isin(used_indices))]
    else:
        pool = df[~df.index.isin(used_indices)]

    fraud_pool = pool[pool["isFraud"] == 1]
    legit_pool = pool[pool["isFraud"] == 0]

    n_mules = max(0, round(size * MULE_FRACTION))
    n_fraud = size - n_mules

    if len(fraud_pool) < n_fraud or len(legit_pool) < n_mules:
        return []  # not enough eligible rows left, skip this ring

    fraud_members = rng.choice(fraud_pool.index.values, size=n_fraud, replace=False)
    mule_members = rng.choice(legit_pool.index.values, size=n_mules, replace=False) if n_mules > 0 else []

    return list(fraud_members) + list(mule_members)


def inject_ring(df: pd.DataFrame, member_idx: list, ring_id: int, tier: str) -> None:
    """Overwrite linkage columns for the given row indices to force overlap."""
    shared_card1 = rng.integers(1000, 20000)
    shared_addr1 = rng.integers(100, 500)

    df.loc[member_idx, "card1"] = shared_card1
    df.loc[member_idx, "addr1"] = shared_addr1
    df.loc[member_idx, "ring_id"] = ring_id
    df.loc[member_idx, "ring_tier"] = tier

    if tier == "tier1":
        shared_device = generate_fake_identity_value("DEV")
        df.loc[member_idx, "DeviceInfo"] = shared_device


def inject_all_rings(df: pd.DataFrame) -> pd.DataFrame:
    used_indices = set()
    ring_id_counter = 0
    ring_log = []

    ring_plan = (
        [("tier1", size) for size in rng.integers(3, 7, N_TIER1_SMALL_RINGS)] +
        [("tier1", size) for size in rng.integers(10, 19, N_TIER1_LARGE_RINGS)] +
        [("tier2", size) for size in rng.integers(2, 6, N_TIER2_SMALL_RINGS)] +
        [("tier2", size) for size in rng.integers(10, 21, N_TIER2_LARGE_RINGS)]
    )
    rng.shuffle(ring_plan)

    for tier, size in ring_plan:
        member_idx = pick_ring_members(df, int(size), tier, used_indices)
        if not member_idx:
            print(f"  Skipped a {tier} ring of size {size} — not enough eligible rows left")
            continue

        inject_ring(df, member_idx, ring_id_counter, tier)
        used_indices.update(member_idx)
        ring_log.append({"ring_id": ring_id_counter, "tier": tier, "size": len(member_idx)})
        ring_id_counter += 1

    print(f"\nInjected {ring_id_counter} rings total ({len(used_indices)} transactions involved)")
    return df, pd.DataFrame(ring_log)


def summarize(df: pd.DataFrame, ring_log: pd.DataFrame) -> None:
    print("\n--- Ring injection summary ---")
    print(ring_log.groupby("tier")["size"].agg(["count", "sum", "mean", "max"]))

    n_mule_rows = df[(df["ring_id"].notna()) & (df["isFraud"] == 0)].shape[0]
    print(f"\nMule (non-fraud-labeled) rows recruited into rings: {n_mule_rows}")

    total_in_rings = df["ring_id"].notna().sum()
    print(f"Total transactions involved in a ring: {total_in_rings} "
          f"({total_in_rings / len(df) * 100:.3f}% of dataset)")


def main():
    df = load_data()
    df, ring_log = inject_all_rings(df)
    summarize(df, ring_log)

    out_path = SYNTHETIC_DIR / "synthetic_transactions_with_rings.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved dataset with injected rings to: {out_path}")

    ring_log_path = SYNTHETIC_DIR / "ring_ground_truth_log.csv"
    ring_log.to_csv(ring_log_path, index=False)
    print(f"Saved ring ground-truth log to: {ring_log_path}")
    print("\nREMINDER: ring_id / ring_tier columns are ground truth for evaluation")
    print("ONLY (Day 10 community detection scoring). Do not feed them into the")
    print("classifier or use them to construct graph edges — that would leak the answer.")


if __name__ == "__main__":
    main()