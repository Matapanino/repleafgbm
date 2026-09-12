# S6E9-scale split/leaf speed profile — 2026-09-12

## Setup

`experiments/2026_09_s6e9_speed_profile.py` generated a seeded synthetic bank
with 130,000 rows and 151 raw columns: 133 numerical and 18 ten-level
categoricals. The model used `RustSplitBackend`, constant leaves, 127 leaves,
50 rounds, learning rate 0.05, `min_samples_leaf=20`, `l2_leaf=2.0`, and
`max_bins=255`. Thread controls matched the probe at four. Three repeats used
the existing `REPLEAFGBM_PROFILE=1` / `PhaseProfiler` hooks.

Wall times were 8.2972, 8.2791, and 8.3035 seconds; median 8.2972 seconds, or
0.16594 seconds per round including one-time binning. This is consistent with
the probe's 0.16670-second 50-round timing guard.

## Phase medians and hotspots

| rank | phase | median seconds | share of median wall |
|---:|---|---:|---:|
| 1 | histogram construction | 5.3483 | 64.5% |
| 2 | split scan | 0.9876 | 11.9% |
| 3 | one-time binning | 0.8274 | 10.0% |
| 4 | partition | 0.2236 | 2.7% |
| 5 | leaf fit | 0.0365 | 0.4% |
| 6 | training-score update/eval | 0.0227 | 0.3% |

The top three account for 86.3% of median wall time. Constant-leaf fitting is
not a meaningful bottleneck at this shape; histogram construction dominates.

## Decision

No code change was attempted. The only obvious cheap-looking optimization—do
not construct histograms for columns excluded by `colsample_bytree`—would alter
the backend histogram contract and likely the Rust ABI/capability ladder. That
is not a safe one-hour refactor. The row/column-sampling PR therefore performs
column selection before split scan for compatibility, with the documented
tradeoff that histogram-build time is unchanged. A future native optimization
should target feature-masked histogram construction and retain NumPy/Rust
parity tests; the expected upper bound is the sampled-out portion of the 64.5%
histogram phase, not total wall time.

Because no optimization was landed, there is no before/after claim. The
numbers above are the reproducible baseline for a later focused native PR.
