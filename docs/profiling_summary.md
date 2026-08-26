# Data Profiling Summary — ULB Credit Card Fraud Dataset

- Total rows: 590,540, columns: 394
- Fraud rate: 3.4990%
- Fraud count: 20663 / 590540
- Time span: ~4392.0 hours
- Columns with nulls: 342 (expect 0 — this dataset is pre-cleaned)

## Note
This dataset has NO identity/device columns (V1-V28 are PCA-anonymized).
Identity/device linkage fields for fraud-ring simulation will be
synthetically generated in the data generation step (Day 3-4), not
sourced from this seed dataset.
