# Dataset Abstraction and Memory Strategy

## Why RepLeafDataset exists

The naive pipeline

```python
Z = encoder.transform(X)          # (n_rows, n_features * emb_dim) — dense
booster.fit(X_raw=X, Z=Z, y=y)
```

materializes the full embedding matrix up front. With per-feature embeddings
(`n_features × n_bins` columns) this is the easiest way to OOM on a large
dataset. `RepLeafDataset` exists so that *who computes Z, when, and where it
lives* is a dataset policy, not something scattered through the booster.

## Responsibilities (v0, implemented)

- Hold the raw feature matrix (float64; categoricals ordinal-encoded).
- Hold the target and `FeatureMetadata` (names, types, category maps).
- Provide `get_raw_features()` (router view) and
  `get_numerical_features()` (encoder input view).
- Compute embeddings lazily via `get_embeddings(encoder)` and cache them
  (one encoder's cache at a time in v0; `clear_embedding_cache()` to drop).
  The cached encoder is held by strong reference and matched with `is`, so a
  garbage-collected encoder can never alias a new one's cache entry.
- Re-apply training metadata to prediction inputs
  (`RepLeafDataset(X, metadata=...)`).

## Memory levers (design, mostly future)

| Lever | Status |
|---|---|
| `max_leaf_emb_dim` + random projection (cap Z width) | implemented |
| Lazy transform (Z computed only if the leaf model needs it) | implemented |
| Embedding cache with explicit invalidation | implemented (minimal) |
| `transform_batch(dataset, rows)` row-chunked transforms | future (API reserved) |
| Low-rank / learned compression instead of random projection | future |
| float32 storage for Z | future (trivial, needs accuracy check) |
| GPU-resident Z with device transfer policy | future |
| Out-of-core raw features (memory-mapped / Arrow) | future |

## Buffer inventory

Training-time state is kept in distinct buffers with simple dtypes, which is
both a readability rule and the migration path to native/GPU backends:

```text
raw feature matrix    float64 (n_rows, n_features)   dataset
binned feature matrix uint16  (n_rows, n_features)   splitter (per fit)
embedding matrix Z    float64 (n_rows, d_z)          dataset cache
gradient buffer g     float64 (n_rows,)              booster (per round)
hessian buffer h      float64 (n_rows,)              booster (per round)
leaf index vector     int64   (n_rows,)              per tree
tree structure        int32/float64 flat arrays      Tree
leaf model params     float64 (n_leaves, 1 + d_z)    LeafValues
prediction cache F    float64 (n_rows,)              booster (train + per eval set)
```

## Scaling expectations

v0 is research-scale: in-memory NumPy, single process. Rough envelope: 1M
rows × 100 features raw ≈ 800 MB float64 (before Z), so v0 is comfortable in
the 10⁴–10⁵ row regime and usable at 10⁶ with narrow Z. Past that, the
out-of-core and GPU items in the roadmap apply; the dataset API is the seam
where they plug in.
