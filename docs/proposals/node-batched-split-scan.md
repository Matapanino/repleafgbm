# Proposal: node-batched split scan (`find_best_split_batched`) for the CUDA path

- **Status:** Approved interface design (core-reviewer GO with corrections);
  **implementation deferred to a GPU-in-the-loop session.** Design-only spec.
- **Date:** 2026-06-25
- **Author:** CUDA/perf overnight loop (driver), reviewed by `core-reviewer`.
- **Type:** Performance optimization (CUDA split-scan launch amortization) + a
  minimal backend-contract extension. Not a default change.
- **Thesis check:** PASS — raw-feature routing only, frozen encoder, Newton leaf
  fitting, NaN-left all unchanged; no `format_version` bump. CUDA-only speed path;
  the host (NumPy/Rust) trees are unchanged.
- **Companion:** `docs/perf-notes/research-node-batched-split-scan.md` (evidence +
  local bitwise math proof), ADR 0005 (CUDA backend), `docs/gpu_roadmap.md`.

---

## 1. Problem

`split_scan` is the dominant CUDA fit phase (48–85%; ~85% on multiclass K=5).
The settled null result ([[gpu-cuda-bottleneck-split-scan]]) is that a **per-node**
on-device scan is launch-bound and loses to the host bulk-copy + NumPy scan
(`threshold 0` is 3–5× slower on narrow). The roadmap's named next lever is to
**batch the scan across the frontier**: compute M nodes' independent best splits
in ONE kernel launch, amortizing the launch over M× the work.

Local proof (`scratchpad/batched_scan_parity.py`): stacking M nodes on a leading
axis over the real `_numeric_split_table` is **bitwise-identical** to looping the
per-node scan — so the kernel changes only launch count, never the result (the
existing allclose caveat stays confined to the histogram atomicAdd).

## 2. Backend contract extension

Add an **optional fast-path** method, mirroring the existing
`build_histograms_multioutput` / `find_best_split_multioutput` pattern
(`backends/base.py:121-179`): a concrete default that loops the per-node
reference (bitwise), which CUDA overrides.

```python
class BaseSplitBackend:
    def find_best_split_batched(
        self,
        hists,                    # (M, F, B, 3) stacked node histograms
        n_bins_per_feature,
        min_samples_leaf,
        l2,
        categorical_mask=None,
    ) -> list[SplitCandidate | None]:
        # default: [self.find_best_split(h, ...) for h in hists]  (bitwise)
```

- `Splitter.find_best_split_batched([...])` wrapper, mirroring
  `Splitter.find_best_split` / `find_best_level_split` (`splitter.py:108-143`).
- **NumPy and Rust references move together, bitwise** — the default loop reuses
  each backend's existing `find_best_split`, so numeric + **categorical subset** +
  the lowest-flat-index tie-break are covered by construction; the parity test
  must assert this across **both** backends (the local proof covered numeric only).

## 3. CUDA kernel (deferred to the GPU session — `native-optimizer`)

One RawKernel, grid over `M × F` (one block per (node, feature)): each block
cumsum-scans its gain row and block-reduce-argmaxes; a second tiny reduction
picks the per-node `(feature, bin)` winner. Reuses the resident histogram + the
adaptive threshold (now over `M·F·B` cells). Only M winners (M×32 bytes) cross
D2H. Categorical subset scan stays on host (parity-critical), as today. Grid/block
+ atomic-reduction depth → `native-optimizer` when a T4 is in the loop.

## 4. Grower wiring — the substantive work (`core/tree.py`)

**Correction (core-reviewer):** depthwise gathers a level's *histograms* but
scans **per-node-eagerly** in `_make_candidate` (`tree.py:532-552`), like
leafwise; the `frontier` deque holds already-scanned candidates. So v1 requires a
real refactor of `_grow_depthwise`:

1. defer `find_best_split` out of `_make_candidate`;
2. collect the level's sibling histograms (already built per level);
3. call `find_best_split_batched` **once** per level;
4. build the candidates from the M winners.

Keep `core/booster.py` / the grower readable. The batched path is taken **only
when** `(cuda backend)` ∧ `(batched gate on)` ∧ `(grow_policy == "depthwise")`;
otherwise the per-node path is unchanged. Leafwise frontier-batching deferred.
Symmetric (`find_best_level_split`, shared-rule) is left alone.

## 5. Env gate

`REPLEAFGBM_CUDA_BATCHED_SCAN`, resolved **once at construction** (the
`_resolve_mo_device_scan` / `_resolve_scan_min_cells` pattern,
`cuda_backend.py:158-217`), stored as `self._batched_scan`, never read per-node,
**default-off**, documented private/not-public-API. ADR-0005-consistent: `"auto"`
never selects cuda; a default-off device sub-feature changes no backend selection.

## 6. Parity contract

- **Local, bitwise (CI-gating, lands WITH the interface):** `find_best_split_batched`
  default == per-node loop on **NumPy and Rust**, numeric + **categorical** +
  tie-break, seeded small synthetic (extend `tests/test_rust_backend.py`).
- **Local, host-path e2e:** a depthwise fit with the batched reference active on
  NumPy/Rust must produce an **identical tree** to the per-node grower (no GPU ⇒
  batching must not change the tree). Guards the §4 refactor.
- **Colab T4 only (allclose + quality-equivalence, NOT rtol=1e-6):** device-batched
  off vs on, interleaved A/B ≥5 reps at wide-200f / multiclass-K5 / multioutput
  (`cuda_overnight_loop --mode ab` + `REPLEAFGBM_CUDA_BATCHED_SCAN`), asserting
  RMSE/quality-equivalence + CPU-path-unchanged (near-tied splits can flip via
  low-bit reductions — [[cuda-multioutput-device-scan]] gotcha).

## 7. Acceptance / rejection (Colab)

Accept iff device-batched (B) wins ≥ reps−1 ∧ Δ>1σ on `split_scan`/fit ∧ CPU
path unchanged ∧ quality-equivalent. Multiclass-K5 first (largest 85% share).
Reject if no signal, if the host path regresses, or if the grower refactor changes
any host tree (bitwise e2e must hold).

## 8. Serialization / compatibility

No format change: the batched scan changes only *which kernel launch* computes the
same `SplitCandidate`; host trees are identical, CUDA trees follow the existing
allclose-not-bitwise regime (ADR 0005). `format_version` does not move.

## 9. Sequencing

1. (this session) spec + corrected research note — **done**.
2. core-reviewer signs off the `core/tree.py` depthwise refactor interface.
3. GPU session: NumPy/Rust `find_best_split_batched` + bitwise parity + host e2e
   (CPU, no GPU needed) → then the CUDA kernel + Colab A/B → results-analyst verdict.
