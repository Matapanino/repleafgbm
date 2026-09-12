# Leaf-embedding control synthetic checks — 2026-09-12

This is the regression fixture used by `tests/test_leaf_embedding_control.py`,
run with `OMP_NUM_THREADS=1`. The default is unchanged; the purpose is to expose
and deconfound the released projection path.

## 665 → 64 reduction

Data: 1,200 seeded Gaussian rows × 133 numerical columns, PLR with four bins
plus its linear term (665 dimensions). The target sums 80 independent threshold
effects. A ridge read is fitted on 900 rows and scored on 300 rows.

| reduction | resolved width | test RMSE |
|---|---:|---:|
| none | 665 | 2.4085 |
| random projection | 64 | 3.7688 |
| target correlation | 64 | 2.6077 |

Both the uncapped and learned paths recover most of the signal lost by the
random projection. This is a controlled mechanism check, not a claim that an
uncapped 665-D leaf fit is cheap or always usable.

## Routed PLR on a noise-heavy bank

Data: 900 rows × 22 numerical columns; two carry the piecewise-linear signal
and 20 are irrelevant (representing a TE/noise-dominated wide bank). Twelve
trees, eight leaves, and `min_samples_leaf=20` are held fixed.

| encoder | width | test RMSE | linear-leaf fraction |
|---|---:|---:|---:|
| PLR on all columns | 110 | 0.3923 | 0.1979 |
| PLR routed to numerical indices `[0, 1]` | 10 | 0.2915 | 0.7188 |

Routing avoids the dimensionality-driven constant fallback and improves this
fixture. It does not justify changing the default encoder or reduction policy.
