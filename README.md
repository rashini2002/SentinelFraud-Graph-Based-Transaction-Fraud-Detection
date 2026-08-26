# SentinelFraud — Graph-Based Transaction Fraud Detection

## Overview
An end-to-end fraud detection pipeline combining supervised classification (XGBoost)
with graph analytics (community detection on shared-attribute networks) to identify
both individual fraudulent transactions and collusive fraud rings.

## Data & Ethics Note
Real financial fraud data is not legally shareable due to privacy and regulatory
constraints. This project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection)
(public, anonymized, released by Vesta Corporation via Kaggle) as a **seed distribution**.

A synthetic transaction dataset is generated from this seed using the
[Synthetic Data Vault (SDV)](https://sdv.dev/) library, preserving realistic statistical
properties without exposing any real individual's data. Collusive fraud-ring patterns
(shared devices, cards, or addresses across accounts) are then deliberately injected
into the synthetic data to create a ground-truth signal for evaluating graph-based
detection methods — a pattern not naturally labeled in the original dataset.

No real transaction or customer data is used or stored in this repository.

## Project Status
🚧 In progress — see `docs/DECISIONS.md` for build log and design decisions.

## Architecture
1. **Data Generation** — synthetic transactions + injected fraud rings (SDV)
2. **Analytics Engineering** — dbt models: staging → entity-linkage → star schema
3. **Graph Analytics** — NetworkX community detection (Louvain) on shared-attribute graph
4. **Modeling** — XGBoost classifier, baseline vs. graph-feature-enhanced comparison
5. **Visualization** — Streamlit network explorer + Tableau executive dashboard

## Tech Stack
Python · PostgreSQL · dbt · NetworkX · XGBoost · SDV · Streamlit · Tableau · Docker

## Repo Structure
```
sentinelfraud/
├── data/
│   ├── raw/          # IEEE-CIS seed data (gitignored)
│   └── synthetic/    # Generated synthetic transactions
├── dbt/              # Analytics engineering models
├── notebooks/        # EDA and modeling notebooks
├── src/              # Pipeline scripts
├── dashboards/       # Streamlit app
└── docs/             # Profiling reports, decisions log
```
