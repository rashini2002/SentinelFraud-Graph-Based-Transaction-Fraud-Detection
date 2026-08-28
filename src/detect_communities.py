"""
detect_communities.py

Day 10 — runs Louvain community detection on the full transaction graph and
evaluates how well it recovers the known fraud rings injected in Day 4.

Two complementary evaluation angles, because a single "recall" number would
hide the Day 9 finding (noisy DeviceInfo cliques diluting otherwise-clean
rings):

  1. RING RECOVERY (recall-oriented): for each of the 88 true rings, do all
     its members end up in the same detected community? Fully recovered
     if yes; otherwise a majority-fraction score.

  2. COMMUNITY PURITY (precision-oriented): for each detected community
     that contains any true ring members, what fraction of the ENTIRE
     community (including non-ring noise nodes) actually belongs to the
     dominant ring? A community that perfectly recovers a ring but is
     diluted by a large attached noise clique (as seen in the Day 9
     visualization) will score low on purity even if recall is high.

Both are broken down by tier (tier1/device vs tier2/card_addr) since Day 6
found device-sharing to be a noisier signal than originally assumed.
"""

import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


def load_graph() -> nx.Graph:
    print("Loading graph from data/processed/transaction_graph.pkl...")
    with open(PROCESSED_DIR / "transaction_graph.pkl", "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def run_louvain(G: nx.Graph) -> dict:
    print("\nRunning Louvain community detection (weighted)...")
    communities = nx.algorithms.community.louvain_communities(G, weight="weight", seed=42)
    print(f"Detected {len(communities)} communities")

    node_to_community = {}
    for community_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = community_id

    community_sizes = sorted([len(c) for c in communities], reverse=True)
    print(f"Largest community size: {community_sizes[0]}")
    print(f"Communities with size >= 3: {sum(1 for s in community_sizes if s >= 3)}")

    return node_to_community, communities


def evaluate_ring_recovery(G: nx.Graph, node_to_community: dict) -> pd.DataFrame:
    print("\n--- Ring Recovery (recall-oriented) ---")

    ring_members = defaultdict(list)
    ring_tier_lookup = {}
    for node, attrs in G.nodes(data=True):
        if attrs.get("ring_id") is not None:
            ring_members[attrs["ring_id"]].append(node)
            ring_tier_lookup[attrs["ring_id"]] = attrs.get("ring_tier")

    results = []
    for ring_id, members in ring_members.items():
        community_ids = [node_to_community.get(m) for m in members]
        community_counts = Counter(community_ids)
        majority_community, majority_count = community_counts.most_common(1)[0]
        majority_fraction = majority_count / len(members)
        fully_recovered = majority_fraction == 1.0

        results.append({
            "ring_id": ring_id,
            "tier": ring_tier_lookup[ring_id],
            "ring_size": len(members),
            "majority_fraction": majority_fraction,
            "fully_recovered": fully_recovered,
        })

    results_df = pd.DataFrame(results)

    overall_full_recovery_rate = results_df["fully_recovered"].mean()
    overall_mean_majority_fraction = results_df["majority_fraction"].mean()
    print(f"Rings fully recovered: {results_df['fully_recovered'].sum()} / {len(results_df)} "
          f"({overall_full_recovery_rate*100:.1f}%)")
    print(f"Mean majority-fraction across all rings: {overall_mean_majority_fraction:.3f}")

    print("\nBy tier:")
    print(results_df.groupby("tier")[["fully_recovered", "majority_fraction"]].mean())

    return results_df


def evaluate_community_purity(G: nx.Graph, node_to_community: dict, communities: list) -> pd.DataFrame:
    print("\n--- Community Purity (precision-oriented) ---")

    results = []
    for community_id, members in enumerate(communities):
        ring_ids_in_community = [
            G.nodes[m]["ring_id"] for m in members if G.nodes[m].get("ring_id") is not None
        ]
        if not ring_ids_in_community:
            continue  # this community has no ring members at all — not relevant here

        dominant_ring, dominant_count = Counter(ring_ids_in_community).most_common(1)[0]
        purity = dominant_count / len(members)  # relative to WHOLE community, including noise

        results.append({
            "community_id": community_id,
            "community_size": len(members),
            "dominant_ring": dominant_ring,
            "ring_members_in_community": len(ring_ids_in_community),
            "purity": purity,
        })

    results_df = pd.DataFrame(results)
    print(f"Communities containing ring members: {len(results_df)}")
    print(f"Mean purity: {results_df['purity'].mean():.3f}")
    print(f"Communities with purity < 0.5 (majority noise): {(results_df['purity'] < 0.5).sum()}")

    low_purity = results_df[results_df["purity"] < 0.5].sort_values("community_size", ascending=False)
    if len(low_purity) > 0:
        print("\nLowest-purity communities (likely noise-diluted rings, e.g. the Day 9 DeviceInfo case):")
        print(low_purity.head(5).to_string(index=False))

    return results_df


def main():
    G = load_graph()
    node_to_community, communities = run_louvain(G)

    ring_recovery_df = evaluate_ring_recovery(G, node_to_community)
    purity_df = evaluate_community_purity(G, node_to_community, communities)

    ring_recovery_df.to_csv(DOCS_DIR / "day10_ring_recovery.csv", index=False)
    purity_df.to_csv(DOCS_DIR / "day10_community_purity.csv", index=False)
    print(f"\nSaved evaluation results to docs/day10_ring_recovery.csv and docs/day10_community_purity.csv")


if __name__ == "__main__":
    main()