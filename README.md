# SentinelFraud — Graph-Based Transaction Fraud Detection

An end-to-end fraud detection pipeline combining a modern analytics engineering stack (dbt + Postgres), graph analytics (NetworkX + Louvain community detection), machine learning (XGBoost + SHAP), and multi-audience BI reporting (Streamlit + Tableau) — built to demonstrate how collusive fraud rings can be detected through relationship structure, not just transaction-level features.

## The Problem

Standard fraud detection treats each transaction independently: does *this* transaction look suspicious? But real fraud often operates as **networks** — groups of accounts sharing a device, card, or address, working together. A transaction-only model is blind to this structure. This project builds a system that detects fraud both ways, and — just as importantly — **honestly measures how much the network view actually adds** on top of a standard classifier.

## Data & Ethics

Real financial fraud data cannot legally be shared due to privacy and regulatory constraints. This project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) (public, anonymized, released via Vesta Corporation on Kaggle) as a **seed distribution**, not as the working dataset itself.

A synthetic transaction dataset (400,000 rows) is generated from this seed using [SDV](https://sdv.dev/)'s GaussianCopula synthesizer. Collusive fraud-ring patterns — shared devices, cards, and addresses across accounts — are then deliberately **injected** into the synthetic data to create a controllable ground-truth signal for evaluating graph-based detection, since real fraud rings aren't natively labeled in the source dataset.

**No real transaction or customer data is used or stored in this repository.**

## Architecture

```
IEEE-CIS (seed) → SDV synthesis → Ring injection → Postgres
                                                        ↓
                                            dbt: staging → intermediate → marts
                                                        ↓
                                    NetworkX graph → Louvain community detection
                                                        ↓
                              Feature engineering → XGBoost (baseline vs. enhanced)
                                                        ↓
                                    Streamlit network explorer + Tableau dashboards
```

**Tech stack:** Python · PostgreSQL · dbt · NetworkX · XGBoost · SHAP · SDV · Streamlit · Tableau · Docker

## Key Findings

This project's value isn't a single polished number — it's a coherent chain of honest diagnosis, each finding informing the next step.

**1. Graph-based ring detection works essentially perfectly on this data.**
Running Louvain community detection on a transaction graph built from shared card/address/device attributes recovered **88 of 88 injected fraud rings (100%)**, with **94.8% mean cluster purity**. The few imperfect cases (3 of 88) were traced to a specific, explainable cause: generic, frequently-recurring `DeviceInfo` values creating coincidental noise cliques — a real limitation of device fingerprinting as a "high confidence" signal, discovered through the data rather than assumed.

**2. The tabular baseline classifier was weak — and the reason is diagnosable.**
A standard XGBoost model on transaction-level features achieved only **AUC 0.52** — barely above random. Root cause: SDV's GaussianCopula synthesis preserves simple statistics (fraud rate matched real data within 0.03 percentage points) but does not preserve the complex, nonlinear feature relationships that make the *real* IEEE-CIS dataset predictable. This is a known, documented cost of synthetic tabular data — not a modeling error.

**3. Graph-derived features added a real, SHAP-validated lift on top of that weak baseline.**
Adding graph features (degree, community size, community fraud-density) improved AUC to **0.536** (+3.2%) and F1 by **+8.6%**. SHAP analysis independently confirmed the specific feature hypothesized to matter most (`community_fraud_density`, which showed a 12x fraud/legit separation in exploratory analysis) as the only graph feature to reach the top-10 most important features overall — two independent methods agreeing on the same signal.

**4. A naive cost-minimization threshold would have recommended an unusable policy.**
Optimizing a simple false-positive/false-negative cost formula recommended flagging **99.8% of all legitimate transactions** — mathematically "optimal" but operationally absurd, since it ignores investigator capacity constraints. Reframing around a realistic review volume (1-2% of transactions) surfaced an honest limitation instead: no threshold achieves both a workable alert volume and meaningful recall given the current feature set.

## Dashboards

Four Tableau dashboards, each demonstrating a different analytical skill set, built on the same underlying data:

| Dashboard | Purpose |
|---|---|
| **Executive Overview** | KPI summary (transactions, fraud rate, rings detected) + daily fraud rate trend + ring composition by tier |
| **Transaction Segmentation & EDA** | Fraud rate by product code, weekend/weekday, device identity presence, and amount distribution |
| **Model Performance & Explainability** | Baseline vs. enhanced metrics, SHAP feature importance, precision-recall tradeoff by threshold |
| **Fraud Ring Network Analysis** | Ring recovery by tier, community purity distribution, ring size vs. purity (visualizes the noise-dilution finding directly) |

Plus an interactive **Streamlit app** (`dashboards/app.py`) for live exploration of individual detected clusters, with filters by fraud density and cluster size.

## Repository Structure

```
sentinelfraud/
├── data/
│   ├── raw/                  # IEEE-CIS seed data (gitignored)
│   ├── processed/            # cleaned + graph-featured tables (gitignored)
│   └── synthetic/            # SDV-generated + ring-injected data (gitignored)
├── dbt/
│   └── models/
│       ├── staging/          # stg_transactions
│       ├── intermediate/     # int_shared_card_addr, int_shared_device
│       └── marts/            # fact_transactions, fact_entity_edges, dim_*
├── src/                      # Python pipeline scripts (Days 1-14)
├── dashboards/
│   ├── app.py                 # Streamlit network explorer
│   └── *.twbx                 # Tableau packaged workbook (4 dashboards)
├── docs/
│   ├── DECISIONS.md            # full day-by-day build log and reasoning
│   └── *.png, *.csv            # generated charts and evaluation results
└── docker-compose.yml         # Postgres container
```

## Running the Pipeline

```bash
# 1. Environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker compose up -d

# 2. Data generation
python src/profile_data.py
python src/clean_data.py
python src/generate_synthetic.py
python src/inject_fraud_rings.py
python src/load_to_postgres.py

# 3. Analytics engineering
cd dbt && dbt run && dbt test && cd ..

# 4. Graph analytics
python src/build_graph.py
python src/detect_communities.py

# 5. Modeling
python src/build_graph_features.py
python src/train_baseline_model.py
python src/train_enhanced_model.py
python src/threshold_analysis.py

# 6. Dashboards
streamlit run dashboards/app.py
# Open dashboards/sentinelfraud_tableau_dashboards.twbx in Tableau
```

## Known Limitations & Future Work

- **Synthetic data ceiling:** GaussianCopula synthesis capped baseline classifier performance well below what's achievable on real data. Future work: regenerate with CTGAN (neural-network-based synthesis) to better preserve nonlinear feature relationships, at the cost of significantly longer fit time.
- **DeviceInfo noise:** device fingerprints proved noisier than assumed as a "high confidence" signal due to generic recurring values. A production system would need device fingerprinting with higher entropy (e.g., canvas fingerprinting, behavioral biometrics) for genuinely high-confidence device-based linkage.
- **dim_card degeneracy:** the card dimension table is a near-degenerate dimension (399,981 rows for 400,000 facts) due to high natural cardinality in the synthetic card fields — a real dataset characteristic, documented rather than artificially "fixed."

Full day-by-day reasoning, every design decision, and every honest dead-end are documented in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Resume Summary

> Built an end-to-end graph-based fraud detection pipeline (dbt, PostgreSQL, NetworkX, XGBoost) achieving 100% recall and 94.8% mean cluster purity in recovering synthetically injected fraud rings; diagnosed a synthetic-data limitation constraining baseline classifier performance, engineered graph-derived features that delivered a SHAP-validated 8.6% F1 improvement, and identified a naive cost-minimization pitfall in threshold selection — communicated across four Tableau dashboards and an interactive Streamlit network explorer.