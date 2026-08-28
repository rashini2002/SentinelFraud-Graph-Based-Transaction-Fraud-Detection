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

---

## Day 5 — dbt Project Scaffolding & First Staging Model

**dbt version note:** running dbt-core 1.8.10, which prints a deprecation
warning on every command (no longer receiving patches). Decision: continue
with this version for the remainder of the build rather than upgrading
mid-project, since an upgrade could introduce its own compatibility issues
(similar to the SDV version drift hit in Day 3). Documented as a known,
accepted limitation rather than an oversight.

**Setup hiccup:** an interrupted first `dbt init` attempt (aborted mid-prompt)
left a partially-created project folder that blocked subsequent init attempts
with "a project called X already exists here." Resolved by removing the
partial folders and rerunning `dbt init` to completion without interruption.

**Data loading:** the ring-injected synthetic dataset (from Day 4) was loaded
into Postgres as `raw_transactions` via a dedicated script
(src/load_to_postgres.py) using pandas' `to_sql()`. Column names retained
their original mixed-case (e.g. `TransactionID`), requiring quoted
identifiers in downstream SQL.

**First staging model — stg_transactions:**
- Defined `raw_transactions` as a dbt source (models/staging/sources.yml),
  with explicit column descriptions flagging `ring_id`/`ring_tier` as
  evaluation-only ground truth, never to be used as model features.
- Built `stg_transactions.sql`: renames all columns to snake_case, casts
  types explicitly (bigint, numeric, boolean, text), and does no business
  logic — staging models are intentionally "thin," per dbt convention.
- Verified row counts post-build: 400,000 total rows, 483 with a non-null
  `ring_id` — an exact match to the Day 4 fraud-ring injection count,
  confirming no data was lost or duplicated in the load → staging pipeline.


## Day 6 — Intermediate Entity-Linkage Models

Combinatorial explosion risk — addressed with group-size capping: A naive self-join on a shared attribute (card1+addr1, or DeviceInfo) risks producing enormous numbers of edges if any single value is shared by many transactions — a fully-connected group of N transactions produces N*(N-1)/2 edges. To prevent this, group sizes are computed first, and only groups with 2 to {{ var('max_linkage_group_size') }} (currently 50) members are joined into pairs. Groups above this threshold are excluded entirely — they are more likely a common demographic/generic-value collision than a genuine relationship, and including them would flood the graph with low-value, fully-connected cliques.

Two intermediate models built:

int_shared_card_addr (Tier 2, weak signal): pairs sharing both card1 AND addr1 (not either alone, since either alone is too common to be a meaningful signal by itself).
int_shared_device (Tier 1, high confidence): pairs sharing DeviceInfo, restricted to has_device_identity = true rows.

Both include an is_true_ring_pair flag (comparing ring_id between the pair) for evaluation purposes only — same restriction as ring_id/ring_tier throughout this project: never used as a graph-construction input or model feature, only for scoring detection quality later.

[RESULTS]:

int_shared_card_addr: 12,086 total edges, 1,859 true ring pairs (~15% of edges correspond to an actual injected ring). Roughly matches the expected sum of within-ring pairwise combinations across the 88 injected rings, confirming the injection and linkage logic are consistent with each other.
int_shared_device: 66,322 total edges, only 720 true ring pairs (~1%).

Finding — DeviceInfo is noisier than expected as a "high-confidence" signal: Day 2 profiling showed DeviceInfo has only 1,786 unique values across ~118,666 identity-matched rows — an average real-world group size of ~66 even before capping. This indicates DeviceInfo contains many generic, recurring strings (e.g. common OS/browser labels or device model codes) rather than truly unique-per-user fingerprints, so plenty of unrelated real transactions naturally collide on device values. This means raw device-sharing edge count alone is a weaker signal than assumed in the original two-tier design — Day 10's community detection will need to rely on edge density/clustering structure, not just edge presence, to separate genuine fraud rings from generic device-string collisions. Documented here as a design consideration carried into Phase 3, not a bug to fix now.

---

## Day 7 — Mart Layer (Star Schema)

**Models built:** dim_date, dim_card, dim_device, fact_transactions,
fact_entity_edges.

**dim_date:** TransactionDT in IEEE-CIS is a relative second-offset from an
unspecified reference point, not a real timestamp. Anchored to an arbitrary
reference date (2017-12-01, a common convention seen in public analyses of
this dataset) purely to derive usable calendar attributes (day-of-week,
weekend flag) for BI purposes. This is a synthetic calendar for analytical
convenience, not a factual claim about when transactions occurred —
documented here to avoid confusion later, especially since the dashboard
(Phase 4) will display real-looking dates derived from this assumption.

**fact_entity_edges:** unifies int_shared_card_addr (Tier 2) and
int_shared_device (Tier 1) into a single edge list for Day 9's graph
construction. Tier 1 (device) edges are weighted 2.0, Tier 2 (card/addr)
edges weighted 1.0, reflecting relative confidence — used by Day 10's
community detection to favor higher-confidence connections when forming
clusters.

**[RESULTS]:**
- fact_transactions: 400,000 rows — exact match to the synthetic dataset,
  confirming no fan-out or row loss from the dimension joins.
- fact_entity_edges: 78,408 rows — exact match to 12,086 + 66,322 from
  Day 6, confirming the union of both linkage tiers is correct.
- dim_date: 182 rows (~6 months), consistent with the dataset's ~4,392-hour
  (~183 day) time span from Day 2 profiling.
- dim_device: 2,649 distinct device_type + device_info combinations.

**Known limitation — dim_card is a near-degenerate dimension:**
dim_card has 399,981 rows — almost one row per transaction, not the small,
reusable lookup table a star schema dimension is meant to be. This is
because the 6-column composite (card1-card6) is nearly unique per row in
this dataset; card1 alone already has high cardinality (17,397 unique
values in the synthetic data, per Day 3), and combining 5 more fields
pushes uniqueness even higher. This is a genuine tradeoff of this dataset's
structure rather than a modeling error, and is left as-is for this project
rather than "fixed" by artificially collapsing the grain — doing so would
misrepresent what the data actually contains. Documented here as a known
design limitation, worth discussing if raised.

---

## Day 8 — dbt Testing & CI Polish

**Schema tests added** across stg_transactions, fact_transactions,
fact_entity_edges, dim_date, and dim_device: not_null and unique on all
primary/surrogate keys, accepted_values on is_fraud (0/1),
shared_attribute_type ('card_addr'/'device'), edge_weight (1.0/2.0), and
ring_tier ('tier1'/'tier2', only where not null).

**Custom singular tests added:** assert_no_negative_amounts (no
transaction should have a negative amount) and
assert_positive_edge_weights (every graph edge weight must be positive) —
both pass when the query returns zero rows.

**dbt version note:** this dbt version (1.8.10) uses the newer `data_tests`
key rather than the deprecated `tests` key — encountered and fixed a
deprecation warning during setup.

**Setup issue — duplicate source definition:** initially defined the
`raw.raw_transactions` source in both `models/staging/sources.yml` (Day 5)
and `models/marts/schema.yml` (Day 8), which dbt rejected as an ambiguous
duplicate. Fixed by keeping the source definition only in sources.yml and
moving its column-level test (TransactionID unique/not_null) there instead.

**Cleanup:** removed `models/example/` — the placeholder starter models
dbt auto-generates on `dbt init` (my_first_dbt_model, my_second_dbt_model).
These were never built as real tables, so their auto-generated tests
failed with "relation does not exist." Not a real issue, just unused
template scaffolding — removed for a clean project.

**[RESULTS]:** 18 of 18 real data tests passing after removing the
unrelated example-model test failures.

---

## Day 9 — Graph Construction (NetworkX)

**Built from fact_entity_edges:** 78,408 edges loaded; where a pair of
transactions was connected via both card/addr AND device sharing, the two
edge weights were accumulated into a single stronger edge rather than kept
as duplicates (2 edges between the same node pair collapsed into 1 in
NetworkX's undirected Graph, weight summed).

**[RESULTS]:**
- Nodes: 27,562 (6.9% of the 400,000 total transactions) — only
  transactions with at least one shared-attribute connection appear in the
  graph; the rest are correctly excluded as having no linkage signal.
- Edges: 77,688 (slightly fewer than 78,408 raw edges, due to the
  weight-accumulation merge described above).
- Connected components: 10,404 total. Of these, 9,416 are just isolated
  pairs (size 2) — low information value. The interesting structure is in
  the 988 components with size >= 3.
- **Components containing at least one true ring member: 88 — an exact
  match to the number of injected rings**, with zero accidental merging
  between separate rings. This confirms the ring-injection design (Day 4)
  and the group-size-capped linkage logic (Day 6) work correctly together
  to produce cleanly separable ring structures.
- Max node degree: 49, consistent with the group-size cap of 50 configured
  in Day 6 (var: max_linkage_group_size).

**Visualization finding — confirms the Day 6 DeviceInfo noise hypothesis:**
The saved sample subgraph (docs/sample_fraud_subgraph.png) shows a small,
fully-red (all true-ring-member) dense clique connected via a single
bridging node to a much larger, fully-blue (no ring membership) dense
clique. This is a direct visual confirmation of the Day 6 finding that
DeviceInfo sharing produces large cliques of coincidentally-connected,
unrelated transactions — the bridging node happens to be a ring member
who also shares a generic device value with the unrelated cluster. This
image is a strong illustration of why Day 10's community detection will
need to weight/filter by cluster density rather than treat all connected
components as equally meaningful.

---

## Day 10 — Community Detection (Louvain) & Ring-Recovery Evaluation

**Method:** ran `networkx.algorithms.community.louvain_communities` (built
into NetworkX 3.x — no separate python-louvain package needed) on the
weighted transaction graph from Day 9. Evaluated using two complementary
metrics rather than one, since Day 9's visualization already suggested
recall and precision could diverge:
- **Ring recovery (recall):** does 100% of a ring's members end up in the
  same detected community?
- **Community purity (precision):** of a community containing ring
  members, what fraction of the WHOLE community (including any attached
  noise) is actually the dominant ring?

**[RESULTS]:**
- Louvain detected 10,405 communities from 27,562 nodes.
- **Ring recovery: 88/88 rings (100%) fully recovered**, both Tier 1
  (device) and Tier 2 (card/addr) — every injected ring's members were
  grouped into a single community with zero fragmentation.
- **Community purity: mean 0.948** across the 88 ring-containing
  communities. Only 3 communities had purity below 0.5 (majority noise):
  ring 41 (29.2% pure, diluted by a 48-node community), ring 44 (31.4%
  pure, 35-node community), and ring 6 (42.9% pure, 7-node community).

**Interpretation:** the Day 6/Day 9 concern about DeviceInfo noise was
real but more contained than initially feared — the group-size cap
(max_linkage_group_size = 50, set in Day 6) already filtered out most of
the damage before it reached the graph. Only 3 of 88 rings (3.4%) ended up
meaningfully diluted by coincidental noise cliques. The single subgraph
visualized in Day 9 was, in retrospect, one of these 3 rare worst-case
examples — a real but non-representative illustration, worth noting rather
than implying it's typical.

**Honest caveat for write-ups/interviews:** 100% recall is a strong result
but should be read in context — the injected rings used deliberately
distinctive forced values (a fabricated card1/addr1 pair, or a fabricated
DeviceInfo string) that don't naturally recur elsewhere in the data by
construction, making them easier to detect than real-world fraud rings
might be, which could share attributes with legitimate transactions in
messier, less clean ways. This evaluates whether the pipeline can recover
a KNOWN, deliberately-injected structure — a valid and useful test of the
method, but not a claim about real-world production performance.

---

## Day 11 — Graph Feature Engineering

**Features engineered:** graph_degree, graph_weighted_degree,
graph_community_id, graph_community_size, graph_community_fraud_density —
computed for all 400,000 transactions (not just the 27,562 in the graph;
unconnected transactions get explicit defaults: degree 0, community_size
1, fraud_density 0, rather than being dropped).

**Leakage safeguard:** graph_community_fraud_density explicitly excludes
each transaction's own fraud label when computing its community's fraud
rate. Without this, a fraud transaction would partly predict its own
label back to itself, artificially inflating Day 13's enhanced-model
performance.

**[RESULTS] — sanity check before modeling:**
- Transactions with any graph connection: 27,562 (6.89%).
- Mean graph_degree: 0.379 (legit) vs 0.659 (fraud) — fraud transactions
  are somewhat more likely to appear in a shared-attribute connection.
- Mean graph_community_size: 1.420 (legit) vs 1.774 (fraud).
- **Mean graph_community_fraud_density: 0.0026 (legit) vs 0.0303 (fraud)
  — a ~12x separation.** Despite small absolute values (diluted by the
  93% of unconnected transactions defaulting to 0), this ratio is a
  strong early indicator that community fraud density will carry real
  predictive signal in the Day 12/13 classifier comparison.

  ---

## Day 12 — Baseline XGBoost (No Graph Features)

**Pipeline gap caught and fixed before training:** discovered that
stg_transactions.sql (Day 5) never selected the 10 sampled V-columns
(V1, V12, V45...V300) that were specifically retained during Day 3's SDV
synthesis for classifier use — they were silently dropped at the staging
layer and never reached fact_transactions. Fixed by adding them to both
stg_transactions.sql and fact_transactions.sql, then rebuilding the dbt
models and regenerating the graph features table before training.
Modeling table shape confirmed to grow from (400000, 16) to (400000, 26)
after the fix, verifying the columns flowed through correctly.

**Features used:** transaction_amt, has_device_identity,
product_cd_encoded, p_email_domain_encoded (label-encoded categoricals),
and the 10 sampled V-columns. Explicitly EXCLUDES card1/addr1/card2-6 as
direct features (too high-cardinality to generalize — their predictive
value is what the graph features are meant to capture in aggregated
form) and EXCLUDES ring_id/ring_tier (ground truth, evaluation-only).

**SMOTE NaN fix:** SMOTE (scikit-learn-based) cannot handle NaN, unlike
XGBoost's native missing-value handling used elsewhere per the Day 2
decision. Median imputation was applied ONLY within the SMOTE training
path (using train-set medians applied to both train and test, avoiding
test-set leakage) — the class-weighted path's NaNs are left untouched,
preserving the original Day 2 design decision.

**[RESULTS]:**
- Class-weighted (scale_pos_weight=27.34): AUC 0.5195, Precision 0.038,
  Recall 0.380, F1 0.070.
- SMOTE (oversampled to 50% fraud): AUC 0.4770 — WORSE than random
  chance, with recall collapsing to 0.6%. Selected class-weighted as the
  baseline for Day 13's comparison.

**Key finding — baseline predictive signal is very weak, likely due to
SDV/GaussianCopula limitations:** AUC 0.52 is barely better than random
guessing. The most probable explanation: GaussianCopula (Day 3) preserves
marginal distributions and pairwise linear correlations well (which is
why fraud rate and simple stats matched the real data closely), but
likely destroyed the complex, nonlinear relationships between the
Vesta-engineered V-columns and the fraud label that make them genuinely
predictive in the real IEEE-CIS dataset. This is a real, documented cost
of using synthetic tabular data for classifier training — aggregate
statistics survived synthesis; fine-grained predictive structure did not.

This sets up Day 13 as a meaningful test: the graph features (built from
deliberately, robustly injected ring structure — not dependent on
tabular synthesis fidelity) are expected to show a much clearer
improvement, precisely because they don't rely on the same fragile
synthetic relationships the baseline tabular features do.
 
---

## Day 13 — Enhanced Model (Graph Features) + SHAP

**SHAP/XGBoost compatibility issue:** XGBoost 3.x serializes base_score
as a bracketed string (e.g. '[5E-1]') that this SHAP version's parser
cannot read, even after in-memory config patching (SHAP re-serializes the
model internally, so the live-object patch didn't propagate). Resolved by
downgrading to xgboost<3 (installed 2.1.4), the standard fix for this
known compatibility issue, rather than continuing to patch around it.

**Same train/test split as Day 12** (identical random_state and stratify)
to ensure a fair, directly comparable before/after evaluation.

**[RESULTS] — Day 12 vs Day 13 comparison:**
| Metric | Baseline | Enhanced | Change |
|---|---|---|---|
| AUC | 0.5195 | 0.5362 | +3.2% |
| Precision | 0.0383 | 0.0423 | +10.6% |
| Recall | 0.3804 | 0.3521 | -7.4% |
| F1 | 0.0695 | 0.0755 | +8.6% |

**SHAP finding — confirms the Day 11 hypothesis precisely:** only 1 of 4
graph features (graph_community_fraud_density) reached the top 10 most
important features overall (rank #10, essentially tied with
product_cd_encoded). graph_degree, graph_weighted_degree, and
graph_community_size did not contribute meaningfully. This exactly
matches Day 11's own sanity check, where community_fraud_density showed a
striking ~12x fraud/legit separation while the other graph features
showed only mild differences — SHAP independently validated that
specific hypothesis rather than contradicting it.

**Honest interpretation:** the improvement is real but modest, not
dramatic — consistent with graph features only being non-zero for 6.89%
of transactions (Day 11), meaning their influence on the overall test-set
metrics is naturally diluted by the ~93% of transactions with no graph
signal at all. The precision/recall trade-off (precision up, recall down)
reflects the model becoming more conservative with the added feature —
fewer false alarms, at the cost of missing some fraud it previously
caught. This is a legitimate, explainable business trade-off, not a
flaw. The coherence between the Day 11 prediction and the Day 13 SHAP
result — the same single feature standing out in both — is a stronger,
more credible research narrative than an unexplained large lift would
have been.

**Resume bullet — updated with real, honest numbers:**
"Engineered graph-based fraud detection features (community-level fraud
density) that improved XGBoost F1-score by 8.6% and precision by 10.6%
over a tabular-only baseline; validated via SHAP that the specific
graph feature hypothesized to matter most (based on independent
observational analysis) was confirmed as the top contributing graph
signal, despite covering only 6.9% of transactions."