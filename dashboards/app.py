"""
app.py

Day 15 — Streamlit app: interactive fraud-ring network explorer.

Lets a user browse detected communities (from Day 10's Louvain results),
filter by fraud-density, and click into a specific cluster to see its
connected accounts and shared attributes rendered as a network graph.

Run with: streamlit run dashboards/app.py
"""

import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import plotly.graph_objects as go
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

st.set_page_config(page_title="SentinelFraud — Network Explorer", layout="wide")


@st.cache_data
def load_graph():
    with open(PROCESSED_DIR / "transaction_graph.pkl", "rb") as f:
        G = pickle.load(f)
    return G


@st.cache_data
def load_communities(_G):
    communities = nx.algorithms.community.louvain_communities(_G, weight="weight", seed=42)
    return communities


@st.cache_data
def build_community_summary(_G, communities) -> pd.DataFrame:
    rows = []
    for community_id, members in enumerate(communities):
        if len(members) < 3:
            continue  # skip isolated pairs — low information value, per Day 9 finding

        fraud_count = sum(1 for m in members if _G.nodes[m].get("is_fraud") == 1)
        ring_ids_present = {_G.nodes[m].get("ring_id") for m in members if _G.nodes[m].get("ring_id") is not None}

        rows.append({
            "community_id": community_id,
            "size": len(members),
            "fraud_count": fraud_count,
            "fraud_density": fraud_count / len(members),
            "contains_true_ring": len(ring_ids_present) > 0,
            "true_ring_ids": ", ".join(str(r) for r in ring_ids_present) if ring_ids_present else "—",
        })

    return pd.DataFrame(rows).sort_values("fraud_density", ascending=False)


def plot_community_network(G: nx.Graph, members: set):
    subG = G.subgraph(members)
    pos = nx.spring_layout(subG, seed=42)

    edge_x, edge_y = [], []
    for u, v in subG.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.5, color="#888"), hoverinfo="none"
    )

    node_x, node_y, node_color, node_text = [], [], [], []
    for node in subG.nodes():
        node_x.append(pos[node][0])
        node_y.append(pos[node][1])
        is_fraud = G.nodes[node].get("is_fraud")
        node_color.append("red" if is_fraud == 1 else "lightblue")
        node_text.append(
            f"Transaction: {node}<br>Fraud: {is_fraud}<br>"
            f"Ring: {G.nodes[node].get('ring_id', 'none')}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text",
        text=node_text,
        marker=dict(size=12, color=node_color, line=dict(width=1, color="black"))
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False, hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


def main():
    st.title("SentinelFraud — Fraud Ring Network Explorer")
    st.caption(
        "Interactive exploration of transaction clusters detected via graph "
        "community detection (Louvain). Red nodes = fraud-labeled transactions."
    )

    G = load_graph()
    communities = load_communities(G)
    summary_df = build_community_summary(G, communities)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total communities (size >= 3)", len(summary_df))
    col2.metric("Communities with known fraud rings", int(summary_df["contains_true_ring"].sum()))
    col3.metric("Mean fraud density", f"{summary_df['fraud_density'].mean()*100:.1f}%")

    st.divider()

    st.subheader("Filter clusters")
    min_density = st.slider("Minimum fraud density", 0.0, 1.0, 0.3, 0.05)
    min_size = st.slider("Minimum cluster size", 3, int(summary_df["size"].max()), 3)

    filtered = summary_df[
        (summary_df["fraud_density"] >= min_density) &
        (summary_df["size"] >= min_size)
    ]

    st.write(f"Showing {len(filtered)} clusters matching filters")
    st.dataframe(filtered, width="stretch", hide_index=True)

    st.divider()

    st.subheader("Inspect a cluster")
    if len(filtered) > 0:
        selected_id = st.selectbox("Select a community to visualize", filtered["community_id"].tolist())
        members = communities[selected_id]

        st.plotly_chart(plot_community_network(G, members), width="stretch")

        row = filtered[filtered["community_id"] == selected_id].iloc[0]
        st.write(f"**Cluster size:** {row['size']} | **Fraud density:** {row['fraud_density']*100:.1f}% | "
                 f"**Contains known ring:** {row['contains_true_ring']} ({row['true_ring_ids']})")
    else:
        st.info("No clusters match the current filters — try lowering the minimum density or size.")


if __name__ == "__main__":
    main()