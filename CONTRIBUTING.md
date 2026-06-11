# Contributing to RepLeafGBM

Thanks for your interest! RepLeafGBM is an experimental research library;
contributions are welcome as long as they respect the core idea:

> Tree routing uses **raw features only**; leaf prediction may use learned
> representations. Changes that blur this asymmetry will not be accepted.

## Development setup

```bash
git clone <repo-url> && cd repleafgbm
pip install -e ".[dev]"            # add ,external or ,bench as needed
bash scripts/check.sh              # lint + tests + examples — must pass
```

## Ground rules

- **Read `CLAUDE.md` first.** It is the canonical statement of architecture
  rules, priorities, and things to avoid (no hidden global state, no
  notebook-only code, no premature GPU/distributed work, ...).
- **Tests are required** for new behavior: fit/predict, save/load
  round-trip, and failure-path coverage. Keep synthetic test data small
  (hundreds of rows) and seeded — the suite must stay fast.
- **Determinism**: the same `random_state` must produce the same model.
  Never call global NumPy random functions; thread RNGs through
  `utils.random.check_random_state`.
- **Style**: `ruff check src tests examples benchmarks` must pass (config in
  pyproject). Type hints on public functions; docstrings on public classes;
  error messages should tell the user what to do.
- **Model-behavior defaults are evidence-based.** Don't change a default
  (e.g. encoder settings, regularization) without an experiment under
  `experiments/` whose markdown report justifies it. See
  `experiments/results/` for the established pattern.
- **Docs move with code**: update `docs/` when architecture changes, add an
  ADR under `docs/adr/` for significant decisions, and keep
  `docs/roadmap.md` honest about implemented vs planned.
- **Serialization changes** need a `format_version` decision (see
  docs/serialization.md) and a backward-compat test.
- Optional dependencies (lightgbm, torch) must never be imported by the
  native path; guard them at call time with actionable messages.

## Pull request checklist

- [ ] `bash scripts/check.sh` passes
- [ ] New behavior is tested
- [ ] Docs / roadmap updated if user-visible
- [ ] No new required dependencies without prior discussion
