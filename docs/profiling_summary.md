# Data Profiling Summary — IEEE-CIS Fraud Detection

- Transaction rows: 590,540, columns: 394
- Identity rows: 144,233, columns: 41
- Fraud rate: 3.499%
- Identity match rate: 24.42%
- Transaction columns >90% null: 2
- Identity columns >90% null: 9

## Linkage candidate columns
These are candidate fields for building the shared-attribute graph
(fraud ring detection) in the analytics engineering phase:
card1-6, addr1-2, P_emaildomain, R_emaildomain, dist1-2, DeviceType, DeviceInfo
