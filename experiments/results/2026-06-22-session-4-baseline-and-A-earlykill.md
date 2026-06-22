# Perf Session 4 — multiclass baseline, candidate-A early-kill, B+C decomposition

Date: 2026-06-22
Branch: `perf/s4-multiclass-histogram` (off `main` @ #25 `ad2d418`)
Machine: darwin arm64, 8 logical / 4 perf cores. `OMP_NUM_THREADS=1 RAYON_NUM_THREADS=8`.
Backend: rust. Config: `--task multiclass --n-classes 5 --max-leaf-emb-dim 64`, n_estimators=100.
Reps: 5, `--seed 0` (deterministic model; reps measure timing noise only).

## Baseline (rebuilt main, native unchanged since #24)

| scale | fit mean / median / σ | histogram | leaf_fit | eval | partition | split_scan |
|---|---|---|---|---|---|---|
| medium (100k×100) | 38.04 / 37.60 / **1.6%** | 32.6% | 30.4% | 11.3% | 10.8% | 4.8% |
| large (500k×200) | 252.51 / 251.75 / **1.7%** | **42.4%** (107.1s) | 28.5% (72.0s) | 10.6% (26.8s) | 8.4% (21.3s) | 1.5% |

**σ ≈ 1.7%, not the feared 10–13%** — this machine is timing very stably, so ≥15% is
cleanly provable. (The prompt's "290–312 s" figure was a heavier config; the *relative*
breakdown is what matters and it holds: histogram is the #1 phase, even larger at scale.)

## Candidate A (histogram row-block / fused-K) — EARLY-KILLED (measured, no prototype needed)

A's only memory-traffic lever for multiclass is **fusing the K classes at the root**
(the sole node with a row-set shared across the K independently-grown trees). Measured
the root-build cost directly to bound A's ceiling:

- T_root (1 class, 500k rows) = 32.9 ms; T_Kroot (5 classes) = 172.7 ms (linear in K)
- root_total = 100 rounds × T_Kroot = **17.3 s = 16.1% of the 107.1 s histogram phase**
- A ceiling (fuse K ⇒ save ~80% of root binned-reads) = 13.8 s
  = **12.9% of histogram = 5.5% of large fit (UPPER bound)**

**Why so low:** `core/tree.py:211–225` uses **sibling subtraction** (only the smaller
child is built), so deeper per-class nodes are **84%** of histogram work — and those have
**no shared row-set across classes**, so A cannot touch them. The per-class "row-block"
variant is independently null: a single class's grad/hess (~4 MB) is already L3-resident
across the feature loop, so row-block only *adds* per-thread local-hist write+merge DRAM
traffic and loses to the current feature-parallel kernel (F=200 ≫ threads=8). This is the
same mechanism as the #23 batched-mc null. **A shelved.**

## B+C decomposition (real grown tree, 500k×200×K5, D=64)

| phase (large) | sub-component | measured | beatable? |
|---|---|---|---|
| leaf_fit 72.0s | native Gram (`leaf_linear_stats`, rayon leaf-par) | ~all | only load-balance |
|  | NumPy centering "glue" | ~0.6 s | no (negligible) |
|  | batched `np.linalg.solve` | ~0.3 s | no (negligible) |
| eval 26.8s | `predict` einsum (`weights[leaf_idx]` gather + dot) | 22.4 s (83%) | **yes → Rust kernel** |
|  | `leaf_idx` assign + `F[:,k]+=` | ~0.4 s | no |
| partition 21.3s | `binned[rows,f]` gather + bool mask + `rows[mask]` | all (pure NumPy) | **yes → Rust kernel** |

**Consequence:** the plan's original **B (glue/solve → native) is moot** (glue+solve
≈ 0.9 s of 72 s). The reliable levers are the two **pure-NumPy** phases:

- **C-eval — fused `predict_add` Rust kernel** (rayon over rows, no 256 MB gather
  materialization): ceiling ~6–7% of fit. Reused by prediction too.
- **C-partition — Rust partition kernel** (single pass, preserves row order ⇒ exact
  index parity): ceiling ~4% of fit.
- **leaf_fit cross-K pooling** — see below; the **#1 lever**.

## Real leaf-size distribution (round 0, real softmax gradients, 100k×100×K5)

Per-class leaf-parallel is **Amdahl-capped by one giant leaf** per tree:

| class | max leaf (% rows) | ideal 8-thread speedup (per-class) |
|---|---|---|
| 0 | 50.5% | 2.0× |
| 1 | 26.2% | 3.8× |
| 2 | 17.9% | 5.6× |
| 3 | 30.8% | 3.2× |
| 4 | 55.0% | 1.8× |

Current achieved (weighted) ≈ 2.76×. **Pooling all K=5 trees' leaves into one rayon pass**
dilutes the largest leaf to ~10% of total work (55k of 500k rows) ⇒ ideal ~8× (realistically
~5–7× on 4P+4E cores). Projected: leaf_fit 72.0s → ~28–40s = **save 13–17% of fit alone**.
Per-leaf accumulation is unchanged (each leaf sums its own rows in order, reading its class's
grad/hess column) ⇒ **bitwise-identical** to the per-class path — only the schedule changes.

## Decision (revised by measurement)
1. **Primary: leaf_fit cross-K pooling** — new native `leaf_linear_stats_mc` (pool K×leaves in
   one rayon pass) + `fit_leaves_multiclass` (grow-all-then-pool-fit). ~13–17% fit. Parity:
   native pooled == native per-class **bitwise**; end-to-end numpy⇄rust allclose.
2. **If margin thin: C-eval `predict_add` kernel** (~6%) and/or **C-partition kernel** (~4%).
Histogram (A) and glue→native (B) are both shelved as measured dead-ends.
