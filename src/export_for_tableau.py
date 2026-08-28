"""
export_for_tableau.py

Day 16 — Tableau Public cannot connect directly to Postgres (no database
connector in the free edition — only file-based sources). This script
exports the tables/views needed for the executive dashboard as CSV files
that Tableau Public can open directly.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLEAU_DIR = PROJECT_ROOT / "dashboards" / "tableau_data"
TABLEAU_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR = PROJECT_ROOT / "docs"


def get_engine():
    conn_str = (
        f"postgresql://{os.getenv('DB_USER', 'sentinel')}:"
        f"{os.getenv('DB_PASSWORD', 'sentinel_dev')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5433')}/"
        f"{os.getenv('DB_NAME', 'sentinelfraud')}"
    )
    return create_engine(conn_str)


def main():
    engine = get_engine()

    print("Exporting fact_transactions (joined with dim_date for real dates)...")
    query = """
        SELECT
            ft.transaction_id,
            dd.calendar_date,
            dd.day_name,
            dd.is_weekend,
            ft.transaction_amt,
            ft.product_cd,
            ft.p_email_domain,
            ft.is_fraud,
            ft.has_device_identity,
            ft.ring_id,
            ft.ring_tier
        FROM fact_transactions ft
        LEFT JOIN dim_date dd ON ft.date_key = dd.date_key
    """
    fact_df = pd.read_sql(query, engine)
    fact_df.to_csv(TABLEAU_DIR / "fact_transactions.csv", index=False)
    print(f"  Exported {len(fact_df):,} rows to fact_transactions.csv")

    print("Exporting fact_entity_edges...")
    edges_df = pd.read_sql("SELECT * FROM fact_entity_edges", engine)
    edges_df.to_csv(TABLEAU_DIR / "fact_entity_edges.csv", index=False)
    print(f"  Exported {len(edges_df):,} rows to fact_entity_edges.csv")

    # Copy over the Day 10-14 result files already generated (already CSV/JSON,
    # just need to be alongside the main export for a single Tableau workbook)
    import shutil
    for fname in [
        "day10_ring_recovery.csv",
        "day10_community_purity.csv",
        "day13_comparison.csv",
        "day13_shap_importance.csv",
        "day14_threshold_analysis.csv",
    ]:
        src = DOCS_DIR / fname
        if src.exists():
            shutil.copy(src, TABLEAU_DIR / fname)
            print(f"  Copied {fname}")
        else:
            print(f"  WARNING: {fname} not found in docs/, skipping")

    print(f"\nAll files ready in: {TABLEAU_DIR}")
    print("Open Tableau Public -> Connect -> Text file -> point to this folder")


if __name__ == "__main__":
    main()