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

## Day 3 — Synthetic Data Generation Scope & Method

Scope decision — curated column subset, not all 435 columns: Fitting SDV on the full merged dataset (435 columns, 590,540 rows) was deliberately avoided. The V1-V339 Vesta engineered features are anonymized and not individually interpretable; synthesizing all of them would be slow, memory-heavy, and adds little value for this project's actual goals (fraud-ring graph detection + classification comparison). Instead, a curated subset was synthesized:

Core transaction fields: TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD
Full linkage-key set from Day 2 (primary, secondary, and high-confidence tiers), plus the has_device_identity flag
A sample of 10 V-columns (arbitrarily spaced across the V1-V339 block) to retain some behavioral-feature signal for the Day 12-13 classifier, without the cost of modeling all 339

Synthesizer choice — GaussianCopula over CTGAN: GaussianCopula was chosen over CTGAN (SDV's other common single-table option) because CTGAN's training time is substantially longer (often hours rather than minutes on this scale of data on a laptop) and its main advantage — better modeling of complex non-linear relationships — is not essential here. This project's priority is preserving linkage-key distributions and the overall fraud rate faithfully enough to plant realistic fraud rings on top, not achieving maximum generative fidelity. GaussianCopula fits in minutes and is sufficient for that purpose.

TransactionID handling: Marked as an id-type primary key in SDV metadata rather than a modeled feature, so the synthesizer generates fresh unique IDs instead of trying to learn a distribution over what is really just a row identifier.

Target synthetic row count: 400,000 (within the 300-500K range scoped on Day 1).

Known risk flagged before running: GaussianCopula can under-represent minority classes in imbalanced data. The script includes an explicit real-vs-synthetic fraud rate comparison to check for this. If synthetic fraud rate drops significantly below the real ~3.5%, the plan is to either oversample fraud rows before fitting, or fit two separate synthesizers (fraud / non-fraud) and combine — not yet needed unless the check fails.

[RESULTS — generate_synthetic.py, run completed]:

Real fraud rate: 3.499% | Synthetic fraud rate: 3.528% | Difference: 0.029pp. Excellent preservation of the minority class — no per-class refitting needed.
SDV's built-in evaluate_quality() was skipped due to a version mismatch between SingleTableMetadata and the evaluation module in the installed SDV release (AttributeError on _get_single_table_name). Replaced with a direct fraud-rate and distribution comparison instead of chasing SDV's internal API changes — sufficient for this project's validation needs.
Known limitation — TransactionAmt tail compression: real data has a much longer right tail (max ~$31,937, std ~239) than synthetic (max ~$1,985, std ~132). GaussianCopula's copula-based fitting compresses extreme outliers in heavily right-skewed columns — a known characteristic, not a bug. Accepted as a limitation for this project rather than switching synthesizers, since the graph/classification work depends more on linkage-key sharing patterns than on faithfully reproducing rare high-value outlier transactions. Documented here for transparency.
card1 cardinality is higher in synthetic (17,397 unique) than real (13,553) — expected, since GaussianCopula treats high-cardinality near-continuous columns somewhat continuously rather than resampling from the exact real value pool. Does not affect linkage/graph design, since fraud-ring detection depends on relative sharing patterns between synthetic transactions, not exact overlap with real-world card1 values.

## Day 4 — Fraud-Ring Injection

Design: planted synthetic collusive fraud rings on top of the SDV-generated dataset, following the two-tier structure from Day 2:

Tier 1 (high confidence): members share both a forced card1/addr1 pattern AND a forced DeviceInfo value. Eligibility restricted to rows with has_device_identity == True.
Tier 2 (weak signal): members share only a forced card1/addr1 pattern, no device overlap. Eligible across the full dataset.

Ring sizing: mix of small rings (Tier 1: 3-6 members, Tier 2: 2-5) and a smaller number of large rings (Tier 1: 10-18, Tier 2: 10-20), to reflect that most real fraud rings are small with occasional larger networks.

Mule accounts: ~10% of each ring's members are drawn from isFraud == 0 rows rather than fraud-labeled rows, simulating transactions that individually look clean but are structurally part of a ring. This is intended to create cases the label alone cannot catch, motivating the graph-based approach — if a ring were fully caught by isFraud already, the graph method would add no value over a plain classifier.

Ground truth handling: ring_id and ring_tier are stored both as columns in the main dataset and in a separate ring_ground_truth_log.csv. These fields are explicitly for scoring the Day 10 community detection results only. They are never used as classifier features or as inputs to graph construction — doing so would leak the answer into the method being evaluated.

[RESULTS — inject_fraud_rings.py, run completed]:

88 rings successfully injected (30 Tier 1, 58 Tier 2) — no rings skipped, confirming eligible-row pools (fraud rows, and device-identity fraud rows specifically for Tier 1) were large enough for the planned ring counts/sizes.
483 transactions (0.121% of the 400K synthetic dataset) are involved in a ring — a small, realistic fraction, consistent with fraud rings being a minority pattern even within fraud-labeled transactions.
Tier 1 mean ring size: 6.3 (max 17). Tier 2 mean ring size: 5.07 (max 20).
27 mule (non-fraud-labeled) rows recruited into rings — somewhat lower than the 10% target of ~48 across 483 members, because the mule count per ring is rounded down for small rings (many small rings round to 0 mules). Not corrected, since it doesn't affect the validity of the design — just noted here for accuracy.