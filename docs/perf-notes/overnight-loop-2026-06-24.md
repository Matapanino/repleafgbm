# Overnight optimization loop — 2026-06-24

Daily narrative log for the CUDA/GPU perf loop. Terse running notes here; the
durable per-iteration records live in `experiment-log.md` (verdicts),
`reflections.md` (GEPA), `rejected-ideas.md`, and `harness-log.md`.

## Setup

- Work branch: `perf/cuda-overnight-loop-20260624` (off `cuda-multioutput-device-scan`).
- Baseline green: `tests/test_rust_backend.py` + `tests/test_cuda_scan_threshold.py`
  (22 passed). Native ext built (`apply_tree`, `predict_linear`, …). CuPy absent
  locally (GPU work batched to Colab T4).
- Fleet (hybrid): +cuda-researcher, perf-profiler, harness-optimizer,
  experiment-strategist; reuse native-optimizer / qa-verifier+core-reviewer /
  results-analyst.
- Harness: `benchmarks/cuda_overnight_loop.py` (median+spread, interleaved A/B,
  `benchmarks/results/latest.jsonl`), `scripts/perf_loop.sh`.

## Plan for the night (first 3 experiments)

1. Forest-batched Rust predictor (local; bitwise parity; `predict_profile.py`).
2. float32 embedding-cache option (local; default-off; allclose-gated).
3. Node-batched split_scan CUDA (code-prep; logical parity local → Colab T4 A/B).
Cadence: GEPA reflection each iteration; harness iteration every 5 product iters;
1 Colab T4 pass at the morning checkpoint.

## Running notes

- [setup] scaffolding + orchestrator created; infra committed (5de1ca8 fleet, 640618a harness).
- [iter 001] forest-batched predictor → REJECT (routing already native, leaf-eval
  already fused; batchable overhead <3%) + HOLD forest-fused single-kernel as a
  dedicated-PR scaffold. Evidence: `artifacts/predict_bench/exp1_baseline/`.
  Pivoting the night to FIT levers (more headroom).
- [iter 002] float32 wide-emb leaf-fit → HOLD (strong evidence). leaf_fit=69% of
  wide-emb fit; float32 Gram+float64 solve = 1.6-1.9x leaf_fit (~30% total fit) at
  rel deviation ~1e-6, thread-robust. Blocked by public-API param (human-gated) +
  allclose tolerance decision. Top promote-to-proposal candidate. Evidence:
  `artifacts/gpu_bench/exp2_probe/` + scratchpad `f32_leaf_ceiling.py`.
- [iter 003] node-batched CUDA split scan → DESIGNED + math-validated (bitwise
  parity of batched-vs-per-node scan, `scratchpad/batched_scan_parity.py`); CUDA
  kernel + grower frontier-batch wiring queued for a GPU-in-the-loop session
  (no local GPU → no blind kernel). Design:
  `docs/perf-notes/research-node-batched-split-scan.md`.

## Morning checkpoint — for the user

- **Shipped (committed on the branch):** loop infra (4 agents, orchestrator,
  ledgers, schema, `perf_loop.sh`); baseline; 3 evidence-backed iterations.
- **Needs your decision:** (iter 002) float32 wide-emb leaf-fit is a strong lever
  (~30% of wide-emb fit) but needs a **public opt-in param** — approve turning it
  into a `docs/proposals/` spec? It is human-gated by design (§8).
- **Queued for the next Colab T4 pass:** (iter 003) node-batched split scan —
  ready to implement during a GPU-in-the-loop session; A/B harness already exists.
- **No regressions:** no product code changed; baseline suite stays green.
