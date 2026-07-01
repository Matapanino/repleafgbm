# Resume runbook — 5-seed numerical leaderboard (PAUSED 2026-06-26)

Production run of the fair leaderboard (deliverable A of the benchmark overhaul),
running **locally** on the dev Mac (Rust backend, `OMP_NUM_THREADS=1`). Paused at
the user's request; fully resumable via the per-suite ledgers — re-running the
same command **skips already-completed `(dataset, model, seed)` cells**.

## State at pause

| suite | cells done | target | note |
|---|---|---|---|
| `grinsztajn_num_reg` | **86** | 475 (19×5×5) | cpu_act/pol/elevators complete; wine_quality partial |
| `grinsztajn_num_cls` | 0 | 400 (16×5×5) | not started |

Ledgers: `benchmarks/results/leaderboard_grinsztajn_num_reg.jsonl`,
`…_num_cls.jsonl`. Do **not** delete them — they are the resume state.
Per-cell time on full-size Grinsztajn datasets is ~80–90 s (not the ~29 s of the
tiny abalone probe). Remaining ≈ 19 h sequential, **≈ 10 h if run concurrently**
(8-core Mac; the GBMs already multi-thread, so two suites ≈ saturate it).

## Resume (concurrent — recommended)

```bash
cd /path/to/repleafgbm   # repo root
caffeinate -i env OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/leaderboard.py \
    --suite grinsztajn_num_reg --seeds 5 --n-trials 50 \
    --out experiments/results/leaderboard-grinsztajn-num-reg.md &
caffeinate -i env OMP_NUM_THREADS=1 PYTHONPATH=src python3 benchmarks/leaderboard.py \
    --suite grinsztajn_num_cls --seeds 5 --n-trials 50 \
    --out experiments/results/leaderboard-grinsztajn-num-cls.md &
```

`caffeinate -i` keeps the Mac awake. Safe to Ctrl-C / re-run any time.

## Check progress

```bash
PYTHONPATH=src python3 -c "from benchmarks.ledger import Ledger; from pathlib import Path; \
[print(s, len(Ledger(Path(f'benchmarks/results/leaderboard_{s}.jsonl'), write_meta=False)), 'cells') \
 for s in ['grinsztajn_num_reg','grinsztajn_num_cls']]"
```

## When both reach their targets

1. Each run writes its report (with CD diagram + Friedman / Wilcoxon / win-tie-loss)
   to its `--out` at the end. Re-run once more with the same command to force a
   final clean report if interrupted.
2. Run **`results-analyst`** on the reports for the evidence verdict. Model
   defaults change **only** via that report (project rule).

## Notes / gotchas

- **Single environment.** Cells are computed with the local Rust backend; don't
  merge with cells from another machine (the manifest pins the environment).
- The Colab path was abandoned (VM recycles on long execs; local is ~8× faster).
  Fallback orchestrator kept at `scripts/colab_cpu_bench_loop.py` if ever needed.
- This is the **5-seed first pass** (valid Friedman/Wilcoxon/CD). Extend to more
  seeds later by raising `--seeds`; the ledger keeps prior seeds.
