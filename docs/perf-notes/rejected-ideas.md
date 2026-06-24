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

_(append REJECT verdicts here with the killing measurement)_
