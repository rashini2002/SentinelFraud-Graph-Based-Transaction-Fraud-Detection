# SentinelFraud — Build Decisions Log

This log tracks key design decisions made during the build, along with the
reasoning behind them. Kept for transparency and as an interview talking point.

---

## Day 1 — Environment & Data Sourcing

**Repo structure:** `data/`, `dbt/`, `notebooks/`, `src/`, `dashboards/`, `docs/`
set up from day one rather than growing organically, to keep the analytics
engineering layer (dbt) cleanly separated from exploratory work (notebooks).

**Postgres via Docker Compose, port 5433:** Used `5433:5432` instead of the
default port mapping to avoid conflicting with a possible existing local
Postgres instance from a prior project. Container: `sentinelfraud_pg`,
database: `sentinelfraud`.

**Dataset sourcing — IEEE-CIS Fraud Detection (Kaggle):**
Real financial fraud data cannot legally be shared due to privacy/regulatory
constraints, so this project uses IEEE-CIS as a seed distribution rather than
using it directly as "the" dataset. IEEE-CIS is public, anonymized data
released by Vesta Corporation via a Kaggle research competition.

Accessing this dataset required two gates beyond the usual Kaggle download:
1. Accepting the competition's "Late Submission" rules (the competition
   closed years ago, so the normal "Join Competition" flow is replaced with
   this).
2. Persona-based identity verification (webcam check), which some gated
   Kaggle competition datasets now require.

A fallback plan (switching to the ungated ULB Credit Card Fraud dataset) was
prepared in case verification failed, but verification succeeded, so the
project proceeds with IEEE-CIS as originally planned. The ULB fallback plan
is kept in mind as a documented contingency, not used.

**venv issue:** Initial `python3 -m venv venv` creation was interrupted
(KeyboardInterrupt) partway through pip bootstrap, leaving a broken venv with
no `bin/activate`. Fixed by deleting and recreating the venv from scratch,
then upgrading pip before installing dependencies (to avoid resolver slowness
with `sdv`'s dependency tree, which includes `torch`).

**Data ethics note:** No real transaction or customer data is used or stored
in this repository. IEEE-CIS is itself already anonymized; on top of that,
this project layers in *synthetically generated* fraud-ring patterns
(shared device/card/address linkages) that do not correspond to any real
individuals, to create a controllable ground-truth signal for evaluating
graph-based fraud detection — a signal not natively labeled in the original
dataset.

---

## Day 2 — Column Disposition & Linkage Strategy

**Profiling results:**
- Transaction data: 590,540 rows, 394 columns. 2 columns >90% null, 174
  columns >50% null.
- Identity data: 144,233 rows, 41 columns. Only joins to 24.42% of
  transactions.
- Fraud rate: 3.499% (20,663 / 590,540) — moderate imbalance. Will compare
  SMOTE vs. class-weighting in the modeling phase rather than assuming one
  approach upfront.

**Linkage candidate column decisions:**

| Column | Null % | Decision | Reason |
|---|---|---|---|
| card1 | 0.0% | Primary linkage key | Near-complete, high cardinality (13,553 unique) |
| card2 | 1.5% | Primary linkage key | Near-complete, useful secondary grouping |
| card3-6 | <1.5% | Secondary linkage key | Near-complete, low cardinality — supporting signal only |
| addr1 | 11.1% | Primary linkage key | Good coverage, moderate cardinality (332) |
| addr2 | 11.1% | Secondary linkage key | Good coverage, low cardinality (74) |
| P_emaildomain | 16.0% | Secondary linkage key | Coarse (59 unique domains) — weak signal alone, useful combined with others |
| R_emaildomain | 76.8% | Dropped from linkage | Too sparse to be reliable |
| dist1 | 59.7% | Dropped from linkage | Too sparse |
| dist2 | 93.6% | Dropped from linkage | Almost entirely missing |
| DeviceType | — (only in 24.42% subset) | Secondary linkage key | Only 2 values — too coarse alone |
| DeviceInfo | — (only in 24.42% subset) | High-confidence linkage key | High cardinality (1,786), strong signal where present |

**Missing-data strategy:**
- V1-V339 (Vesta engineered features) and other high-null columns are NOT
  imputed. XGBoost handles missing values natively via its split-on-missing
  mechanism, so imputation would add noise without benefit. This is a
  deliberate choice, documented here to make that explicit rather than leaving
  it implicit.
- Linkage-key columns with nulls (e.g., addr1, P_emaildomain) are treated as
  "missing = no edge contribution" rather than imputed with a placeholder
  value, to avoid creating false shared-attribute edges between unrelated
  accounts that both happen to have missing data.

**Identity coverage gap (75.58% of transactions have no identity/device data):**
Decision: build the fraud-ring graph primarily on card1/card2/addr1
(near-complete coverage), with DeviceInfo layered in as a high-confidence
signal only for the subset where it exists. This mirrors a realistic
production constraint (identity/device data is often incomplete in practice)
rather than assuming full coverage.

**Fraud-ring injection design (for Day 3-4):**
Two-tier ring structure planned:
- **Tier 1 (high confidence):** accounts sharing both a card1/addr1 pattern
  AND DeviceInfo — planted only within the ~24% subset with identity data.
- **Tier 2 (weak signal):** accounts sharing only a card1/addr1 pattern, no
  device overlap — planted across the full dataset.

This gives two ring types to separately evaluate against during community
detection (Day 10), testing whether the graph correctly assigns higher
confidence/density to Tier 1 rings than Tier 2 rings — a more nuanced
evaluation than a single flat "recovered ring" metric.
