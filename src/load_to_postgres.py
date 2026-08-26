"""
load_to_postgres.py

Day 5 — loads the ring-injected synthetic dataset into Postgres as a raw
table, so dbt has a source to build staging models on top of.

This intentionally loads the data AS-IS (no cleaning/transformation here) —
that's dbt's job starting with the staging layer. This script's only
responsibility is getting the data into the warehouse.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "sentinelfraud")
DB_USER = os.getenv("DB_USER", "sentinel")
DB_PASSWORD = os.getenv("DB_PASSWORD", "sentinel_dev")

RAW_TABLE_NAME = "raw_transactions"


def main():
    print("Loading synthetic dataset with rings...")
    df = pd.read_parquet(SYNTHETIC_DIR / "synthetic_transactions_with_rings.parquet")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str)

    print(f"Writing to Postgres table '{RAW_TABLE_NAME}'...")
    df.to_sql(RAW_TABLE_NAME, engine, if_exists="replace", index=False, chunksize=10000)

    print("Done. Verifying row count in Postgres...")
    with engine.connect() as conn:
        result = conn.exec_driver_sql(f"SELECT COUNT(*) FROM {RAW_TABLE_NAME}")
        count = result.scalar()
        print(f"Row count in Postgres: {count:,}")


if __name__ == "__main__":
    main()