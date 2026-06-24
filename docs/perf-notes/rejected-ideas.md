# Rejected / dead-end ideas — do not re-litigate without new evidence

Each entry: idea · why it failed · the measurement that killed it · the date.
Seeded from prior sessions (memory + `docs/gpu_roadmap.md`) so the loop does not
waste a night re-testing settled null results.

## Settled dead ends

- **CUDA grad/hess device cache** — null result (2026-06-19). Bytes fell but
  wall-clock unchanged: grad/hess H2D is <0.2% of fit time. Pick GPU targets by
  `phase_seconds`, not transfer bytes.
- **All-GPU numeric scan (threshold 0)** — 3–5× slower on narrow (30f). Per-node
  on-device scan is launch-bound; it loses to the host bulk-copy + NumPy scan.
  The lever is *batching across nodes/classes*, not scanning per-node on GPU.
- **Multiclass histogram row-block** — measured DEAD for mc (session 4).
- **Multiclass leaf-glue → native** — measured DEAD for mc (session 4).
- **Per-node on-device scalar scan as default** — flips near-tied splits via
  low-bit CuPy reductions; only ever quality-neutral, never rtol=1e-6, and not
  faster on narrow. Keep adaptive host fallback.

## Loop-rejected (this branch)

- **Forest-batched routing (standalone)** — REJECT 2026-06-24 (iter 001). Routing
  is already per-tree native (`apply_tree`); batching only removes per-call pyo3 +
  marshalling overhead, a fraction of the 4–7% `overhead_seconds`, while adding a
  whole forest API + concatenated-forest representation + parity surface. Evidence:
  `artifacts/predict_bench/exp1_baseline/` (200 trees: constant overhead 4–7%,
  embedded_linear overhead 10–16% but routing only 28–41%). NOTE: the full
  **forest-fused predictor** (route+leaf-eval+accumulate in one kernel) is HELD,
  not rejected — it could capture the overhead but is a large, layer-coupling change.
