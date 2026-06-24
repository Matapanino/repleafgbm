# Design — node-batched CUDA split scan (the GPU split_scan lever)

Status: **DESIGNED + math-validated locally; CUDA kernel queued for a GPU-in-the-loop
session.** (2026-06-24, iter 003). No local GPU → no blind CUDA kernel shipped.

## Why

`split_scan` is the dominant CUDA fit phase (48–85%; ~85% on multiclass K=5;
`docs/gpu_roadmap.md`, [[gpu-cuda-bottleneck-split-scan]]). Prior sweeps proved
the **per-node on-device scan is launch-bound** and never beats the host
bulk-copy + NumPy scan; `threshold 0` (all-GPU per-node) is 3–5× slower on narrow.
The roadmap's named next lever is therefore **batch the scan across nodes**, not
scan per node on GPU: amortise one kernel launch over M frontier nodes' work.

## What (target math — already validated)

Per-node numeric scan = `numpy_backend._numeric_split_table` (cumsum over bins +
Newton gain `G²/(H+l2)` + min_samples / range mask + argmax, missing-bin left,
lowest-flat-index tie-break). Batching stacks M nodes on a leading M axis and
takes argmax per node — **bitwise-identical** to looping the per-node scan
(same float64 elementwise ops, independent per M-slice).

- Local proof: `scratchpad/batched_scan_parity.py` — batched vs per-node loop on
  the real `_numeric_split_table`, **BITWISE PARITY: PASS** for all M nodes
  (feature, bin, and gain match exactly).
- So the CUDA kernel changes **only launch count**, never the result. The
  existing allclose caveat stays confined to the histogram atomicAdd; the scan is
  deterministic given the histogram.

Distinct from `numpy_backend.find_best_level_split` (symmetric/oblivious): that
chooses ONE shared `(feature, bin)` for the whole level by **summing** node gains
— a different objective. Node-batched scan keeps M independent winners.

## How (interface + kernel)

1. **Backend method** (additive to `BaseSplitBackend`):
   `find_best_split_batched(hists, n_bins_per_feature, min_samples_leaf, l2,
   categorical_mask) -> list[SplitCandidate | None]` for M stacked node
   histograms `(M, F, B, 3)`. NumPy/Rust reference = loop the existing per-node
   path (bitwise); CUDA = one kernel.
2. **CUDA kernel**: grid over `M × F` (one block per (node, feature)); each block
   cumsum-scans its gain row and block-reduce-argmaxes; a second tiny reduction
   picks the per-node `(feature, bin)` winner. Reuses the resident histogram +
   the adaptive-threshold gate, now applied to `M·F·B` cells. Only M winners
   (M×32 bytes) cross D2H.
3. **Grower wiring** (the real cost — architectural, keep readable): present a
   *frontier batch* to the backend. **CORRECTION (core-reviewer, 2026-06-25):**
   depthwise already gathers a whole level's **histograms**, but the split
   **scan** is *per-node-eager* — `_grow_depthwise` calls `find_best_split` one
   node at a time inside `_make_candidate` (`tree.py:532-552`), exactly like
   leafwise; the `frontier` deque holds candidates whose split is already
   computed. So node-batching is **not** free wiring even in depthwise: it needs
   a real `tree.py` refactor — defer the scan out of `_make_candidate`, collect
   the level's sibling histograms, call `find_best_split_batched` once, then build
   candidates. That grower refactor (on the readable hot path) is the substantive,
   reviewable work; the `BaseSplitBackend` method itself is the easy part.
   - **v1: depthwise** (level = the batch unit, after the refactor above).
   - **leafwise:** batch the priority-queue frontier — deferred (tangles with the
     pop ordering + the adaptive host/GPU crossover).
   - **symmetric** already scans a level at once but with a *different*
     (shared-rule, summed-gain) objective (`find_best_level_split`) — leave it.

## Validation plan

- Local (now): batched NumPy/Rust reference must be **bitwise** vs the per-node
  loop (extend `tests/test_rust_backend.py` style). The math proof above is the
  pre-req; the product reference test lands with the implementation.
- Colab T4 (queued): `cuda_overnight_loop --mode ab` device-batched off vs on at
  wide-200f + multiclass-K5 + multioutput, interleaved (`colab_scan_ab` pattern),
  ≥5 reps. **Assert quality-equivalence, NOT rtol=1e-6** (near-tied splits can
  flip via low-bit reductions — the [[cuda-multioutput-device-scan]] gotcha).
  Accept only if B (batched) wins ≥ reps-1 AND Δ>1σ AND CPU path unchanged.

## Risk / gate

- The grower frontier-batch interface is architectural → **core-reviewer** sign-off
  on the interface BEFORE the kernel. Keep `core/booster.py` / grower readable.
- HOLD until a GPU-in-the-loop session (or user pre-approval of >1 Colab pass):
  the kernel cannot be iterated without a T4, and blind CUDA is rejected by the
  loop's safety rules.
