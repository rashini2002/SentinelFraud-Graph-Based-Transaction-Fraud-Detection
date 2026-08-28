"""
build_graph_features.py

Day 11 — engineers graph-derived features for EVERY transaction (not just
the 27,562 that appear in the graph). Transactions with no shared-attribute
connections get explicit zero/default values rather than being dropped —
a production fraud model must score all transactions, connected or not.

Features engineered:
  - degree: number of distinct transactions this one shares an attribute with
  - weighted_degree: sum of edge weights (device edges count double, per Day 7)
  - community_id: Louvain community assignment (Day 10); -1 for unconnected
    transactions (their own singleton "community")
  - community_size: size of the transaction's community; 1 for unconnected
  - community_fraud_density: fraction of the transaction's community
    (EXCLUDING itself) that is fraud-labeled — this is the single feature
    most directly capturing "am I sitting in a suspicious cluster?"

IMPORTANT: ring_id / ring_tier are NOT included as features here — per
docs/DECISIONS.md, they are ground truth for evaluation only and must
never leak into the classifier.
"""

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_graph_and_communities():
    print("Loading graph...")
    with open(PROCESSED_DIR / "transaction_graph.pkl", "rb") as f:
        G = pickle.load(f)

    print("Running Louvain (same seed as Day 10, for consistent community IDs)...")
    communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)

    node_to_community = {}
    for community_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = community_id

    return G, communities, node_to_community


def compute_community_fraud_density(G: nx.Graph, communities: list) -> dict:
    """
    For each community, fraction of members that are fraud-labeled.
    Computed once per community (not per-node) for efficiency, then broadcast.
    """
    density_by_community = {}
    for community_id, members in enumerate(communities):
        fraud_count = sum(1 for m in members if G.nodes[m].get("is_fraud") == 1)
        density_by_community[community_id] = fraud_count / len(members)
    return density_by_community


def build_feature_table(G: nx.Graph, node_to_community: dict,
                          community_sizes: dict, density_by_community: dict) -> pd.DataFrame:
    print("\nBuilding per-transaction graph feature table...")

    rows = []
    for node in G.nodes():
        degree = G.degree(node)
        weighted_degree = G.degree(node, weight="weight")
        community_id = node_to_community.get(node, -1)
        community_size = community_sizes.get(community_id, 1)

        # Exclude self when computing "how suspicious is my neighborhood" —
        # a fraud transaction's own label shouldn't inflate its own feature.
        density = density_by_community.get(community_id, 0.0)
        if community_size > 1:
            # adjust to exclude self's own fraud label from the density calc
            self_is_fraud = G.nodes[node].get("is_fraud") == 1
            raw_fraud_count = density * community_size
            adjusted_fraud_count = raw_fraud_count - (1 if self_is_fraud else 0)
            density = adjusted_fraud_count / (community_size - 1)

        rows.append({
            "transaction_id": node,
            "graph_degree": degree,
            "graph_weighted_degree": weighted_degree,
            "graph_community_id": community_id,
            "graph_community_size": community_size,
            "graph_community_fraud_density": density,
        })

    graph_features_df = pd.DataFrame(rows)
    print(f"Built graph features for {len(graph_features_df):,} connected transactions")
    return graph_features_df


def merge_with_all_transactions(graph_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join graph features onto ALL 400,000 transactions. Unconnected
    transactions get default values: degree 0, community_size 1 (a
    singleton "community" of just themselves), fraud_density 0 (no
    neighbors to be suspicious about).
    """
    print("\nLoading full transaction set from Postgres...")
    from sqlalchemy import create_engine
    from dotenv import load_dotenv
    import os
    load_dotenv()

    conn_str = (
        f"postgresql://{os.getenv('DB_USER', 'sentinel')}:"
        f"{os.getenv('DB_PASSWORD', 'sentinel_dev')}@"
        f"{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5433')}/"
        f"{os.getenv('DB_NAME', 'sentinelfraud')}"
    )
    engine = create_engine(conn_str)
    all_txns = pd.read_sql("SELECT * FROM fact_transactions", engine)
    print(f"Loaded {len(all_txns):,} total transactions")

    merged = all_txns.merge(graph_features_df, on="transaction_id", how="left")

    # Fill defaults for the ~93% with no graph connections
    merged["graph_degree"] = merged["graph_degree"].fillna(0)
    merged["graph_weighted_degree"] = merged["graph_weighted_degree"].fillna(0.0)
    merged["graph_community_id"] = merged["graph_community_id"].fillna(-1)
    merged["graph_community_size"] = merged["graph_community_size"].fillna(1)
    merged["graph_community_fraud_density"] = merged["graph_community_fraud_density"].fillna(0.0)

    n_connected = (merged["graph_degree"] > 0).sum()
    print(f"\nTransactions with graph connections: {n_connected:,} "
          f"({n_connected/len(merged)*100:.2f}%)")
    print(f"Transactions with no connections (defaulted): {len(merged) - n_connected:,}")

    return merged


def main():
    G, communities, node_to_community = load_graph_and_communities()
    community_sizes = {i: len(c) for i, c in enumerate(communities)}
    density_by_community = compute_community_fraud_density(G, communities)

    graph_features_df = build_feature_table(G, node_to_community, community_sizes, density_by_community)
    final_df = merge_with_all_transactions(graph_features_df)

    out_path = PROCESSED_DIR / "transactions_with_graph_features.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"\nSaved final modeling table to: {out_path}")
    print(f"Shape: {final_df.shape}")

    # Quick sanity check — do fraud transactions have higher graph feature
    # values on average? This is a preview of whether these features will
    # actually help the classifier before we even train it.
    print("\n--- Quick sanity check: mean feature values by fraud label ---")
    print(final_df.groupby("is_fraud")[
        ["graph_degree", "graph_community_size", "graph_community_fraud_density"]
    ].mean())


if __name__ == "__main__":
    main()