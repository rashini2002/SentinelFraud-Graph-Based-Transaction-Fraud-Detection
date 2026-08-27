"""
build_graph.py

Day 9 — loads fact_entity_edges from Postgres, builds a NetworkX graph, and
computes basic structural statistics. This is the foundation Day 10's
community detection (Louvain) will run on top of.

Note on scope: only transactions that appear in at least one edge become
nodes in this graph. The other ~99.8% of the 400K transactions have no
shared-attribute connections at all and are correctly excluded — a graph
of isolated singleton nodes carries no analytical value.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "sentinelfraud")
DB_USER = os.getenv("DB_USER", "sentinel")
DB_PASSWORD = os.getenv("DB_PASSWORD", "sentinel_dev")


def get_engine():
    conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


def load_edges_and_labels(engine) -> tuple:
    print("Loading fact_entity_edges...")
    edges_df = pd.read_sql("SELECT * FROM fact_entity_edges", engine)
    print(f"Loaded {len(edges_df):,} edges")

    print("Loading transaction labels (is_fraud, ring_id, ring_tier)...")
    labels_df = pd.read_sql(
        "SELECT transaction_id, is_fraud, ring_id, ring_tier FROM fact_transactions",
        engine
    )
    print(f"Loaded {len(labels_df):,} transaction labels")

    return edges_df, labels_df


def build_graph(edges_df: pd.DataFrame, labels_df: pd.DataFrame) -> nx.Graph:
    print("\nBuilding NetworkX graph...")
    G = nx.Graph()

    for _, row in edges_df.iterrows():
        a, b = row["transaction_id_a"], row["transaction_id_b"]
        weight = row["edge_weight"]
        attr_type = row["shared_attribute_type"]

        # If an edge already exists between this pair (e.g. they share BOTH
        # a card/addr pattern AND a device), accumulate the weight rather
        # than overwrite — a pair connected two ways is a stronger signal.
        if G.has_edge(a, b):
            G[a][b]["weight"] += weight
            G[a][b]["shared_attribute_types"].add(attr_type)
        else:
            G.add_edge(a, b, weight=weight, shared_attribute_types={attr_type})

    # Attach node attributes (is_fraud, ring_id, ring_tier) for nodes that
    # actually made it into the graph
    labels_lookup = labels_df.set_index("transaction_id").to_dict(orient="index")
    for node in G.nodes():
        info = labels_lookup.get(node, {})
        G.nodes[node]["is_fraud"] = info.get("is_fraud")
        G.nodes[node]["ring_id"] = info.get("ring_id")
        G.nodes[node]["ring_tier"] = info.get("ring_tier")

    return G


def report_graph_stats(G: nx.Graph) -> None:
    print(f"\n--- Graph stats ---")
    print(f"Nodes: {G.number_of_nodes():,}")
    print(f"Edges: {G.number_of_edges():,}")

    components = list(nx.connected_components(G))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"\nConnected components: {len(components)}")
    print(f"Largest component size: {component_sizes[0]}")
    print(f"Top 10 component sizes: {component_sizes[:10]}")
    print(f"Number of components with size >= 3: {sum(1 for s in component_sizes if s >= 3)}")
    print(f"Number of isolated pairs (size 2): {sum(1 for s in component_sizes if s == 2)}")

    degrees = [d for _, d in G.degree()]
    print(f"\nDegree stats — min: {min(degrees)}, max: {max(degrees)}, "
          f"mean: {sum(degrees)/len(degrees):.2f}")

    # How many components contain at least one true ring member?
    ring_component_count = 0
    for comp in components:
        ring_ids_in_comp = {G.nodes[n]["ring_id"] for n in comp if G.nodes[n]["ring_id"] is not None}
        if ring_ids_in_comp:
            ring_component_count += 1
    print(f"\nComponents containing at least one true ring member: {ring_component_count}")


def visualize_sample_subgraph(G: nx.Graph, out_path: Path) -> None:
    """
    Pick the largest connected component that contains at least one true
    fraud-ring member, and plot it with ring members highlighted — a
    sanity check that injected rings are visible as dense clusters, per
    the Day 1 plan.
    """
    components = list(nx.connected_components(G))
    ring_components = [
        c for c in components
        if any(G.nodes[n]["ring_id"] is not None for n in c)
    ]

    if not ring_components:
        print("No components with ring members found — skipping visualization.")
        return

    # Pick the largest ring-containing component for the most visually
    # interesting plot
    target_component = max(ring_components, key=len)
    subG = G.subgraph(target_component)

    node_colors = [
        "red" if G.nodes[n]["ring_id"] is not None else "lightblue"
        for n in subG.nodes()
    ]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(subG, seed=42)
    nx.draw(
        subG, pos,
        node_color=node_colors,
        node_size=100,
        with_labels=False,
        edge_color="gray",
        width=0.5,
    )
    plt.title(f"Sample fraud-ring subgraph (n={len(target_component)} nodes)\n"
              f"Red = injected ring member, Blue = coincidental sharing only")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved sample subgraph visualization to: {out_path}")


def main():
    engine = get_engine()
    edges_df, labels_df = load_edges_and_labels(engine)
    G = build_graph(edges_df, labels_df)
    report_graph_stats(G)
    visualize_sample_subgraph(G, DOCS_DIR / "sample_fraud_subgraph.png")

    # Save the graph itself for Day 10's community detection to load directly,
    # avoiding a full rebuild from SQL each time.
    graph_path = PROJECT_ROOT / "data" / "processed" / "transaction_graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    print(f"\nSaved graph object to: {graph_path}")


if __name__ == "__main__":
    main()